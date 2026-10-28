import gc,json,os,subprocess,sys,time
from pathlib import Path
import torch
from .common import ROOT,atomic,read_jsonl,digest,file_hash
from .experiment import rollout,summary,collect_online
from .worker import replay_rows,terminal_calibration,choose_controller
from .gui_world import GUIWorld,train_world,checkpoint_reusable

def run():
    protocol=json.loads((ROOT/'protocol.json').read_text());splits=protocol['splits'];initial=read_jsonl(ROOT/'data/initial_world.jsonl')
    pilot=json.loads((ROOT/'results/opf_seed0/complete.json').read_text());version=pilot['promotion']['incumbent_version']
    incumbent_world=Path(pilot['promotion']['incumbent']);incumbent_policy=None
    incumbent_controller=pilot['outcomes'][version]['controller']['controller'];incumbent_pure={'alpha':'pure','terminal':pilot['outcomes'][version]['terminal']}
    folders=[]
    for generation in [1,2]:
        folder=ROOT/'results'/f'rsi_g{generation}';folder.mkdir(exist_ok=True)
        old_world,old_policy=incumbent_world,incumbent_policy;old_controller=dict(incumbent_controller);old_pure=dict(incumbent_pure)
        model=GUIWorld.load_gui(old_world);model.checkpoint_path=str(old_world)
        rollout(splits[f'rsi_update{generation}'],folder/'update.jsonl','hybrid',model,old_policy,old_controller)
        new=collect_online(folder/'update.jsonl',folder/'update.branches.jsonl')
        del model;gc.collect();torch.cuda.empty_cache()
        policy_path=ROOT/'checkpoints'/f'rsi_g{generation}'/'policy'
        result=policy_path/'result.json'
        if not result.exists():
            gpu=protocol['resource_plan']['rsi_policy_gpu']
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>500:raise RuntimeError('RSI policy GPU occupied by another job')
            command=[sys.executable,'-m','rsi_anything_gui.policy_train','--new',str(folder/'update.transitions.jsonl'),
                     '--replay',str(ROOT/'data/initial_world_policy.jsonl'),'--output',str(policy_path),'--seed',str(200+generation)]
            if old_policy:command+=['--parent',str(old_policy)]
            with (ROOT/'logs'/f'rsi_policy_g{generation}.log').open('a') as log:
                subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        adapter=json.loads(result.read_text())['adapter'];new_policy=Path(adapter) if adapter else old_policy
        policy_metadata=policy_path/('training.json' if (policy_path/'training.json').exists() else 'not_updated.json')
        policy_request=json.loads(policy_metadata.read_text())
        if policy_request['input_new_hash']!=digest([r['pair_hash'] for r in read_jsonl(folder/'update.transitions.jsonl')]) or policy_request['input_replay_hash']!=digest([r['pair_hash'] for r in read_jsonl(ROOT/'data/initial_world_policy.jsonl')]) or policy_request['parent']!=(str(old_policy) if old_policy else None):
            raise RuntimeError('Policy checkpoint/result belongs to another training request')
        for filename,sha in policy_request.get('files',{}).items():
            if file_hash(policy_path/filename)!=sha:raise RuntimeError('Policy checkpoint file corrupted')
        new_world=ROOT/'checkpoints'/f'rsi_g{generation}'/'world';old_data=replay_rows(initial,new,300+generation)
        atomic(folder/'replay.json',{'new_pair_hashes':[r['pair_hash'] for r in new],'replay_pair_hashes':[r['pair_hash'] for r in old_data]})
        if not checkpoint_reusable(new_world,new+old_data,'opf',300+generation,parent=old_world,replay_rows=old_data,epochs=protocol['update_epochs']):
            model=train_world(new+old_data,new_world,'opf',300+generation,parent=old_world,replay_rows=old_data,epochs=protocol['update_epochs'])
            del model;gc.collect();torch.cuda.empty_cache()
        old_model=GUIWorld.load_gui(old_world);old_model.checkpoint_path=str(old_world)
        olddev=summary(rollout(splits['dev'],folder/'old_old_selected_dev.jsonl','hybrid',old_model,old_policy,old_controller))
        del old_model;gc.collect();torch.cuda.empty_cache()
        model=GUIWorld.load_gui(new_world);model.checkpoint_path=str(new_world)
        terminal=terminal_calibration(model,folder/'terminal_calibration.json')
        selected=choose_controller(model,folder/'selection',terminal,new_policy)
        newdev=summary(rollout(splits['dev'],folder/'new_new_selected_dev.jsonl','hybrid',model,new_policy,selected['controller']))
        del model;gc.collect();torch.cuda.empty_cache()
        promoted=newdev['success_rate']-olddev['success_rate']>=protocol['promotion_gain'] and newdev['effective_seconds']<=olddev['effective_seconds']*protocol['promotion_cost_ratio']
        manifest={'generation':generation,'old_world':str(old_world),'new_world':str(new_world),'old_policy':str(old_policy) if old_policy else None,
                  'new_policy':str(new_policy) if new_policy else None,'old_controller':old_controller,'new_controller':selected['controller'],
                  'old_pure':old_pure,'new_pure':{'alpha':'pure','terminal':terminal},'dev':{'old_old':olddev,'new_new':newdev},
                  'selection':selected,'promoted':promoted,'new_rows':len(new),'replay_rows':len(old_data),'test_not_opened_yet':True,'time':time.time()}
        atomic(folder/'versions_frozen.json',manifest);folders.append(folder)
        if promoted:incumbent_world=new_world;incumbent_policy=new_policy;incumbent_controller=selected['controller'];incumbent_pure=manifest['new_pure']
    atomic(ROOT/'results/rsi_versions_frozen.json',{'generations':2,'time':time.time(),'all_versions_frozen_before_test':True})
    for folder in folders:
        m=json.loads((folder/'versions_frozen.json').read_text());outcomes={}
        combinations=[('old_old','old_world','old_policy','old_controller','old_pure'),('old_new','new_world','old_policy','new_controller','new_pure'),
                      ('new_old','old_world','new_policy','old_controller','old_pure'),('new_new','new_world','new_policy','new_controller','new_pure')]
        for arm,w,p,c,pure in combinations:
            model=GUIWorld.load_gui(m[w]);model.checkpoint_path=m[w];outcomes[arm]={}
            for kind,mode,control in [('selected','hybrid',m[c]),('pure','world',m[pure])]:
                outcomes[arm][kind]=summary(rollout(splits['rsi_test'],folder/f'{arm}_{kind}_test.jsonl',mode,model,m[p],control))
            del model;gc.collect();torch.cuda.empty_cache()
        for arm,p in [('old_policy',m['old_policy']),('new_policy',m['new_policy'])]:
            outcomes[arm]=summary(rollout(splits['rsi_test'],folder/f'{arm}_test.jsonl','policy',adapter=p))
        atomic(folder/'complete.json',{'versions':m,'outcomes':outcomes,'completed':time.time(),'raw_rejected_candidates_reported':True})
    atomic(ROOT/'results/rsi_complete.json',{'generations':2,'completed':time.time(),'training_seeds':1})

if __name__=='__main__':
    try:run()
    except Exception as error:
        import traceback
        atomic(ROOT/'logs/rsi_failure.json',{'error':repr(error),'traceback':traceback.format_exc()});raise
