import argparse,gc,json,random,time
from pathlib import Path
import torch
from .common import ROOT,atomic,atomic_jsonl,read_jsonl,context,digest,legal_actions
from .experiment import rollout,summary,collect_online,rank,branch,select,sigmoid
from .gui_env import GUIEnv
from .gui_world import GUIWorld,train_world,checkpoint_reusable

def replay_rows(initial,new,seed):
    rng=random.Random(seed);old=rng.sample(initial,min(len(initial),len(new)))
    if len(old)<len(new):old+=rng.choices(initial,k=len(new)-len(old))
    return old

def terminal_calibration(model,path):
    data=read_jsonl(ROOT/'data/terminal_calibration.jsonl')
    inputs=[r for r in data if r['action']['kind']=='submit'];predictions=[]
    for i in range(0,len(inputs),32):predictions+=model.forecast([r['context'] for r in inputs[i:i+32]])[0]
    protocol=json.loads((ROOT/'protocol.json').read_text());grid=protocol['terminal_calibration_grid'];choices=[]
    for temperature in grid['temperature']:
        for bias in grid['bias']:
            brier=sum((sigmoid(p['submit_success_logit']/temperature+bias)-r['success'])**2 for p,r in zip(predictions,inputs))/len(inputs)
            choices.append({'temperature':temperature,'bias':bias,'brier':brier})
    best=min(choices,key=lambda c:(c['brier'],abs(c['bias']),abs(c['temperature']-1)))
    atomic(path,{'selected':best,'grid':choices,'n':len(inputs),'instances':len({r['seed'] for r in inputs}),
                 'positive':sum(r['success'] for r in inputs),'fit_split':'terminal_calibration','data_hash':digest([r['pair_hash'] for r in inputs])})
    atomic_jsonl(Path(path).with_suffix('.predictions.jsonl'),[{'seed':r['seed'],'pair_hash':r['pair_hash'],'success':r['success'],'logit':p['submit_success_logit']} for r,p in zip(inputs,predictions)])
    return {'temperature':best['temperature'],'bias':best['bias']}

def choose_controller(model,folder,terminal,adapter=None,baseline=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);protocol=json.loads((ROOT/'protocol.json').read_text());seeds=protocol['splits']['dev']
    baseline=baseline or summary(rollout(seeds,folder/'baseline_dev.jsonl','policy',adapter=adapter))
    candidates=[]
    for alpha in protocol['hybrid_alpha_grid']:
        controller={'alpha':alpha,'terminal':terminal}
        result=summary(rollout(seeds,folder/f'alpha{alpha}_dev.jsonl','hybrid',model,adapter,controller))
        gain=result['success_rate']-baseline['success_rate'];ratio=result['effective_seconds']/max(baseline['effective_seconds'],1e-9)
        candidates.append({'controller':controller,'result':result,'gain':gain,'cost_ratio':ratio,'eligible':gain>=protocol['controller_enable_gain'] and ratio<=protocol['controller_enable_cost_ratio']})
    eligible=[c for c in candidates if c['eligible']]
    chosen=max(eligible,key=lambda c:(c['result']['success_rate'],-c['result']['effective_seconds'],-c['controller']['alpha'])) if eligible else None
    controller=chosen['controller'] if chosen else {'alpha':0.,'terminal':terminal}
    result=chosen['result'] if chosen else baseline
    output={'controller':controller,'baseline':baseline,'candidates':candidates,'selected_dev':result,'fallback':chosen is None,'data_split':'dev'}
    atomic(folder/'controller.json',output);return output

def diagnostics(model,seeds,path,controller,adapter=None):
    path=Path(path);folder=path.with_suffix('.instances');folder.mkdir(parents=True,exist_ok=True);env=GUIEnv();records=[]
    try:
        for seed in seeds:
            complete=folder/f'{seed}.json';prediction_file=folder/f'{seed}.predictions.json'
            if complete.exists():record=json.loads(complete.read_text())
            else:
                obs=env.reset(seed);prefix=[]
                for _ in range(seed%3):
                    action=rank(obs,adapter)['actions'][0];obs,_=env.step(action);prefix.append(action)
                    if obs['done']:obs=env.reset(seed);prefix=[];break
                proposal=rank(obs,adapter);predictions,_=model.forecast([context(obs,a) for a in proposal['actions']])
                committed={'seed':seed,'before':obs,'prefix':prefix,'proposal':proposal,'forecasts':predictions,'time':time.time()}
                if prediction_file.exists() and json.loads(prediction_file.read_text())['forecasts']!=predictions:raise RuntimeError('Diagnostic forecast changed on resume')
                atomic(prediction_file,committed)
                branches=[branch(env,seed,prefix,obs,a,len(prefix),'diagnostic_not_training') for a in proposal['actions']]
                index,_=select(proposal,predictions,controller)
                results=[]
                for actual,prediction in zip(branches,predictions):
                    bits=[bool(x) for x in actual['checked']];before=[e['checked'] for e in obs['elements']]
                    predicted=[p>=.5 for p in prediction['checked_probabilities'][:len(bits)]];slot=actual['action'].get('slot')
                    results.append({'actual':actual,'prediction':prediction,'state_exact':int(bits==predicted),'persistence_exact':int(before==bits),
                                    'clicked_correct':int(predicted[slot]==bits[slot]) if slot is not None else None,
                                    'utility_mse':(prediction['utility']-actual['utility'])**2,'value_mse':(prediction['value']-actual['return_target'])**2})
                record={'seed':seed,'before':obs,'prefix':prefix,'proposal':proposal,'forecasts':predictions,'branches':results,
                        'regret':max(r['return_target'] for r in branches)-branches[index]['return_target'],
                        'prediction_time':committed['time'],'completed':time.time(),'replay_verified':True}
                atomic(complete,record)
            records.append(record)
    finally:env.close()
    atomic_jsonl(path,records)
    branches=[b for r in records for b in r['branches']];clicks=[b for b in branches if b['clicked_correct'] is not None]
    result={'instances':len(records),'branches':len(branches),'state_exact':sum(b['state_exact'] for b in branches)/len(branches),
            'persistence_exact':sum(b['persistence_exact'] for b in branches)/len(branches),
            'clicked_accuracy':sum(b['clicked_correct'] for b in clicks)/len(clicks),'utility_mse':sum(b['utility_mse'] for b in branches)/len(branches),
            'value_mse':sum(b['value_mse'] for b in branches)/len(branches),'mean_regret':sum(r['regret'] for r in records)/len(records)}
    atomic(path.with_suffix('.summary.json'),result);return result

def prepare_version(model,path,folder,version):
    protocol=json.loads((ROOT/'protocol.json').read_text());splits=protocol['splits']
    if version=='frozen':
        fitted=json.loads((folder/'collection_terminal.json').read_text())
        terminal={k:fitted['selected'][k] for k in ['temperature','bias']}
        atomic(folder/f'{version}_terminal.json',fitted)
        controller=json.loads((folder/'collection_selection/controller.json').read_text())
    else:
        terminal=terminal_calibration(model,folder/f'{version}_terminal.json')
        controller=choose_controller(model,folder/f'{version}_selection',terminal)
    pure={'alpha':'pure','terminal':terminal}
    pure_dev=summary(rollout(splits['dev'],folder/f'{version}_pure_dev.jsonl','world',model,controller=pure))
    prepared={'world':str(path),'pure':pure,'pure_dev':pure_dev,'controller':controller,'terminal':terminal,'frozen_before_test':True,'time':time.time()}
    atomic(folder/f'{version}_frozen.json',prepared);return prepared

def evaluate_version(model,path,folder,version):
    protocol=json.loads((ROOT/'protocol.json').read_text());splits=protocol['splits'];output={}
    prepared=json.loads((folder/f'{version}_frozen.json').read_text());pure=prepared['pure'];controller=prepared['controller'];terminal=prepared['terminal']
    for name,mode,c in [('pure','world',pure),('selected','hybrid',controller['controller'])]:
        output[name]={'dev':prepared['pure_dev'] if name=='pure' else controller['selected_dev'],
                      'test':summary(rollout(splits['test'],folder/f'{version}_{name}_test.jsonl',mode,model,controller=c))}
    output['diagnostic']=diagnostics(model,splits['diagnostic'],folder/f'{version}_diagnostic.jsonl',pure)
    output['controller']=controller;output['terminal']=terminal;return output

def run(mode,seed):
    name=f'{mode}_seed{seed}';folder=ROOT/'results'/name;folder.mkdir(exist_ok=True);protocol=json.loads((ROOT/'protocol.json').read_text())
    initial=read_jsonl(ROOT/'data/initial_world.jsonl');p0=ROOT/'checkpoints'/name/'w0';p1=ROOT/'checkpoints'/name/'w1'
    started=time.time()
    if not checkpoint_reusable(p0,initial,mode,seed,epochs=protocol['initial_epochs']):
        model=train_world(initial,p0,mode,seed,epochs=protocol['initial_epochs']);del model;gc.collect();torch.cuda.empty_cache()
    model=GUIWorld.load_gui(p0);model.checkpoint_path=str(p0)
    # Development/calibration determines the data-collection controller without opening test.
    terminal=terminal_calibration(model,folder/'collection_terminal.json')
    collection=choose_controller(model,folder/'collection_selection',terminal)
    rollout(protocol['splits']['update'],folder/'update.jsonl','hybrid',model,controller=collection['controller'])
    new=collect_online(folder/'update.jsonl',folder/'update.branches.jsonl');old=replay_rows(initial,new,seed)
    atomic(folder/'replay.json',{'new_pair_hashes':[r['pair_hash'] for r in new],'replay_pair_hashes':[r['pair_hash'] for r in old],'seed':seed})
    del model;gc.collect();torch.cuda.empty_cache()
    if not checkpoint_reusable(p1,new+old,mode,seed+100,parent=p0,replay_rows=old,epochs=protocol['update_epochs']):
        model=train_world(new+old,p1,mode,seed+100,parent=p0,replay_rows=old,epochs=protocol['update_epochs']);del model;gc.collect();torch.cuda.empty_cache()
    outcomes={}
    for version,path in [('frozen',p0),('updated',p1)]:
        model=GUIWorld.load_gui(path);model.checkpoint_path=str(path)
        prepare_version(model,path,folder,version)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic(folder/'all_versions_frozen.json',{'time':time.time(),'both_versions_frozen_before_test':True})
    for version,path in [('frozen',p0),('updated',p1)]:
        model=GUIWorld.load_gui(path);model.checkpoint_path=str(path)
        outcomes[version]=evaluate_version(model,path,folder,version)
        del model;gc.collect();torch.cuda.empty_cache()
    olddev=outcomes['frozen']['selected']['dev'];newdev=outcomes['updated']['selected']['dev']
    promoted=newdev['success_rate']-olddev['success_rate']>=protocol['promotion_gain'] and newdev['effective_seconds']<=olddev['effective_seconds']*protocol['promotion_cost_ratio']
    atomic(folder/'complete.json',{'mode':mode,'seed':seed,'outcomes':outcomes,'initial_rows':len(initial),'new_rows':len(new),'replay_rows':len(old),
                                  'promotion':{'promoted':promoted,'incumbent':str(p1 if promoted else p0),'incumbent_version':'updated' if promoted else 'frozen'},'started':started,'completed':time.time()})

def baseline():
    splits=json.loads((ROOT/'protocol.json').read_text())['splits'];result={}
    for method in ['policy','rule']:
        for split in ['dev','test']:
            result[method+'_'+split]=summary(rollout(splits[split],ROOT/f'results/baseline_{method}_{split}.jsonl',method))
    atomic(ROOT/'results/baseline_complete.json',result)

def confirm(mode,seed):
    folder=ROOT/'results'/f'{mode}_seed{seed}';outcomes={};splits=json.loads((ROOT/'protocol.json').read_text())['splits']
    for version,checkpoint in [('frozen','w0'),('updated','w1')]:
        path=ROOT/'checkpoints'/f'{mode}_seed{seed}'/checkpoint;model=GUIWorld.load_gui(path);model.checkpoint_path=str(path)
        frozen=json.loads((folder/f'{version}_frozen.json').read_text())
        outcomes[version]=summary(rollout(splits['confirmation'],folder/f'{version}_confirmation.jsonl','world',model,controller=frozen['pure']))
        del model;gc.collect();torch.cuda.empty_cache()
    atomic(folder/'confirmation_complete.json',outcomes)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','baseline','confirm']);parser.add_argument('--mode',default='opf');parser.add_argument('--seed',type=int,default=0);a=parser.parse_args()
    try:{'run':lambda:run(a.mode,a.seed),'baseline':baseline,'confirm':lambda:confirm(a.mode,a.seed)}[a.phase]()
    except Exception as error:
        import traceback
        atomic(ROOT/'logs'/f'{a.phase}_{a.mode}_{a.seed}_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'time':time.time()});raise
