"""Behavioral preflight: real browser, interrupted resume, learning and checkpoint recovery."""
import argparse,gc,json,time
from pathlib import Path
import torch
from .common import ROOT,atomic,context,target,label_transition,legal_actions,read_jsonl
from .gui_env import GUIEnv,rule_action
from .experiment import collect_initial,select,rollout
from .gui_world import GUIWorld,train_world,batch_indices
from .policy_train import positive_rows,stratified

def browser():
    from . import experiment
    import random
    protocol=json.loads((ROOT/'protocol.json').read_text());seeds=protocol['splits']['probe'];checks={};env=GUIEnv()
    try:
        before=env.reset(seeds[0]);action={'kind':'click','slot':0};after,_=env.step(action)
        checks['actual_chromium']=bool(env.browser.version)
        checks['one_bit_changed']=after['elements'][0]['checked']!=before['elements'][0]['checked'] and after['elements'][1:]==before['elements'][1:]
        again,_=env.step(action);checks['double_click_restores']=again['elements']==before['elements']
        env.replay(seeds[0],[action],after);checks['exact_visible_budget_replay']=True
        encoded=json.loads(target(after));checks['public_goal_and_budget_in_target']=encoded['instruction']==after['instruction'] and encoded['remaining_actions']==after['remaining_actions']
        checks['scorer_has_no_evaluator']=all(k not in json.loads(context(before,action)) for k in ['raw_reward','reward_reason','target_bits','success'])
    finally:env.close()
    proposal={'candidate_ids':[0,1],'probabilities':[.6,.4],'actions':[{'kind':'click','slot':0},{'kind':'submit'}]}
    forecasts=[{'submit_success_logit':100.,'value':-.5},{'submit_success_logit':5.,'value':-1.}]
    checks['terminal_success_used_only_for_submit']=select(proposal,forecasts,{'alpha':'pure'})[0]==1
    checks['alpha_zero_reverts_to_policy']=select(proposal,forecasts,{'alpha':0.})[0]==0
    candidate={'utility':1.,'success':1,'action':{'kind':'submit'},'pair_hash':'a','episode_success':0}
    checks['failed_episode_positive_feedback_rejected']=not positive_rows([candidate])
    candidate['episode_success']=1;checks['successful_submit_retained']=len(positive_rows([candidate]))==1
    p=ROOT/'results/contracts_resume.jsonl';original=experiment.episode;calls=[]
    def interrupted(env,seed,*args,**kwargs):
        if len(calls)==1:raise RuntimeError('INJECTED_INTERRUPTION')
        calls.append(seed);return original(env,seed,*args,**kwargs)
    if not p.with_suffix('.complete.json').exists():
        experiment.episode=interrupted
        try:rollout(seeds,p,'rule')
        except RuntimeError as error:
            if str(error)!='INJECTED_INTERRUPTION':raise
        finally:experiment.episode=original
    before_commits={x.name for x in p.with_suffix('.episodes').glob('*.json')}
    resumed=[]
    def observed(env,seed,*args,**kwargs):resumed.append(seed);return original(env,seed,*args,**kwargs)
    experiment.episode=observed
    try:outcomes=rollout(seeds,p,'rule')
    finally:experiment.episode=original
    checks['resume_skips_fully_committed_episode']=all(f'{s}.json' not in before_commits for s in resumed)
    checks['resume_preserves_all_transitions']=sum(r['steps'] for r in outcomes)==len(read_jsonl(p.with_suffix('.transitions.jsonl')))
    checks['strict_rule_success']=all(r['success']==1 for r in outcomes)
    atomic(ROOT/'results/browser_contracts.json',{'passed':all(checks.values()),'checks':checks,'time':time.time()})
    if not all(checks.values()):raise RuntimeError('Browser/selection/recovery contracts failed')
    collect_initial(seeds,ROOT/'data/probe_world.jsonl')

def world():
    rows=read_jsonl(ROOT/'data/probe_world.jsonl');path=ROOT/'checkpoints/probe_excluded'
    protocol=json.loads((ROOT/'protocol.json').read_text());model=train_world(rows,path,'opf',999,epochs=protocol['probe_epochs'])
    training_modes={'online':model.online.backbone.training,'teacher_eval':not model.target.training}
    forecasts=model.forecast([r['context'] for r in rows])[0]
    clicked=[(r,p) for r,p in zip(rows,forecasts) if r['action']['kind']=='click']
    submit=[(r,p) for r,p in zip(rows,forecasts) if r['action']['kind']=='submit']
    clicked_accuracy=sum((p['checked_probabilities'][r['action']['slot']]>=.5)==bool(r['checked'][r['action']['slot']]) for r,p in clicked)/len(clicked)
    pos=[(r,p) for r,p in submit if r['success']];neg=[(r,p) for r,p in submit if not r['success']]
    balanced=(sum(p['success_probability']>=.5 for r,p in pos)/len(pos)+sum(p['success_probability']<.5 for r,p in neg)/len(neg))/2
    value_mse=sum((p['value']-r['return_target'])**2 for r,p in zip(rows,forecasts))/len(rows)
    del model;gc.collect();torch.cuda.empty_cache()
    loaded=GUIWorld.load_gui(path);roundtrip=loaded.forecast([r['context'] for r in rows])[0]
    difference=max(abs(p[k]-q[k]) for p,q in zip(forecasts,roundtrip) for k in ['utility','value','success_probability','done_probability'])
    checks={'training_mode_explicit':all(training_modes.values()),'checkpoint_roundtrip':difference<1e-5,
            'encoder_and_ema_updated':json.loads((path/'training.json').read_text())['encoder_changed'] and json.loads((path/'training.json').read_text())['ema_changed'],
            'small_training_set_clicked_accuracy':clicked_accuracy>=.85,'small_training_set_terminal_balanced_accuracy':balanced>=.9,'small_training_set_value_mse':value_mse<=.1}
    atomic(ROOT/'results/world_contracts.json',{'passed':all(checks.values()),'checks':checks,'clicked_accuracy':clicked_accuracy,'terminal_balanced_accuracy':balanced,
                                             'value_mse':value_mse,'roundtrip_max_difference':difference,'training_instances':protocol['splits']['probe'],
                                             'excluded_from_formal_data':True,'time':time.time()})
    if not all(checks.values()):raise RuntimeError('Actual-model learnability preflight failed; inspect before formal training')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['browser','world']);a=p.parse_args();browser() if a.phase=='browser' else world()
