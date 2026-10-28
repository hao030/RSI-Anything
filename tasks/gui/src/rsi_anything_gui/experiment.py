"""Transactional actual-browser episodes and training-only counterfactual collection."""
import argparse,json,random,time,math,urllib.request
from pathlib import Path
from .common import ROOT,PORT,atomic,atomic_jsonl,context,correctness,digest,label_transition,read_jsonl,legal_actions,checkpoint_identity,effective_cost
from .gui_env import GUIEnv,EpisodeTerminated,rule_action

def rank(obs,adapter=None):
    visible={k:obs[k] for k in ['instruction','elements','remaining_actions']}
    payload=json.dumps({'observation':visible,'adapter':str(adapter) if adapter else None}).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/rank',data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
    if not result.get('actions') or {'kind':'submit'} not in result['actions']:raise RuntimeError('Invalid terminal-inclusive candidate pool')
    return result

def sigmoid(x):return 1/(1+math.exp(-max(-60.,min(60.,x))))

def select(proposal,forecasts,controller):
    alpha=controller.get('alpha',0.)
    policy=[math.log(max(proposal['probabilities'][i],1e-12)) for i in proposal['candidate_ids']]
    if alpha==0:return 0,{'scores':policy,'alpha':0.,'world_used':False}
    calibration=controller.get('terminal',{'temperature':1.,'bias':0.})
    values=[]
    for action,prediction in zip(proposal['actions'],forecasts):
        if action['kind']=='submit':v=2*sigmoid(prediction['submit_success_logit']/calibration['temperature']+calibration['bias'])-1
        else:v=prediction['value']
        values.append(v)
    if alpha=='pure':scores=values
    else:
        mean=sum(values)/len(values);std=(sum((v-mean)**2 for v in values)/len(values))**.5
        scores=[p+alpha*(v-mean)/max(std,.1) for p,v in zip(policy,values)]
    return max(range(len(scores)),key=lambda i:scores[i]),{'scores':scores,'values':values,'policy_scores':policy,'alpha':alpha,'terminal':calibration,'world_used':True}

def episode(env,seed,method='policy',model=None,adapter=None,controller=None):
    wall_started=time.time()
    obs=env.reset(seed);initial_observation=obs;initial={k:obs[k] for k in ['instruction','elements']}
    started=time.monotonic();trace=[];actions=[];policy_seconds=world_seconds=browser_seconds=0.;hits=0
    termination=None
    controller=controller or {'alpha':'pure'}
    for step in range(8):
        if obs['done']:
            if not trace:
                termination={'reason':obs.get('termination_reason','environment'),'before':obs,'after':obs}
            break
        proposal=forecasts=selection=None
        if method=='rule':action=rule_action(obs)
        else:
            proposal=rank(obs,adapter);policy_seconds+=proposal['seconds'];hits+=int(proposal['cached'])
            enabled=method in ['world','hybrid'] and controller['alpha']!=0
            if enabled:
                current=env.observe()
                if current['done']:
                    termination={'reason':current.get('termination_reason','environment'),'before':obs,'after':current}
                    obs=current;break
                if model is None:raise RuntimeError('World selection without a model')
                forecasts,elapsed=model.forecast([context(obs,a) for a in proposal['actions']]);world_seconds+=elapsed
            index,selection=select(proposal,forecasts,controller if method!='policy' else {'alpha':0.})
            action=proposal['actions'][index]
        browser_started=time.monotonic()
        try:
            after,elapsed=env.step(action)
        except EpisodeTerminated as error:
            browser_seconds+=time.monotonic()-browser_started
            after=error.observation
            termination={'reason':after.get('termination_reason','environment'),'before':obs,'after':after}
            obs=after;break
        browser_seconds+=elapsed
        row=label_transition(obs,action,after,elapsed,f'{seed}:t0',step)
        row.update(proposal=proposal,forecast=forecasts,selection=selection)
        trace.append(row);actions.append(action);obs=after
    success=int(bool(obs['submitted'] and obs['done'] and correctness(obs)==1. and not obs.get('budget_exhausted')))
    for row in trace:row['episode_success']=success
    clicks=[a['slot'] for a in actions if a['kind']=='click']
    result={'episode_id':f'{seed}:t0','seed':seed,'initial_hash':digest(initial),'success':success,'steps':len(actions),
            'actions':actions,'initial':initial_observation,'final':obs,'wrong_submit':int(obs['submitted'] and not success),'repeated_clicks':len(clicks)-len(set(clicks)),
            'partial_fraction':correctness(obs),'seconds':time.monotonic()-started,'policy_nominal_seconds':policy_seconds,
            'world_seconds':world_seconds,'browser_seconds':browser_seconds,'cached_policy_requests':hits,
            'method':method,'adapter':str(adapter) if adapter else None,'controller':controller if method!='policy' else {'alpha':0.},
            'started':wall_started,'completed':time.time()}
    if termination is not None:result['termination']=termination
    return result,trace

def rollout(seeds,path,method='policy',model=None,adapter=None,controller=None):
    path=Path(path);commits=path.with_suffix('.episodes');commits.mkdir(parents=True,exist_ok=True)
    identity={'seeds':list(seeds),'method':method,'world':checkpoint_identity(getattr(model,'checkpoint_path',None)) if model else None,
              'policy':checkpoint_identity(adapter),'controller':controller,'protocol':digest(json.loads((ROOT/'protocol.json').read_text()))}
    meta=path.with_suffix('.identity.json')
    if meta.exists() and json.loads(meta.read_text())!=identity:raise RuntimeError('Rollout identity changed: '+str(path))
    atomic(meta,identity);env=GUIEnv();records=[]
    try:
        for seed in seeds:
            commit=commits/f'{seed}.json'
            if commit.exists():
                record=json.loads(commit.read_text())
                if record['identity_hash']!=digest(identity) or record['result']['seed']!=seed:raise RuntimeError('Invalid committed episode')
            else:
                result,trace=episode(env,seed,method,model,adapter,controller)
                record={'identity_hash':digest(identity),'result':result,'transitions':trace}
                atomic(commit,record)
            records.append(record)
            atomic(path.with_suffix('.progress.json'),{'pid':__import__('os').getpid(),'completed':len(records),'total':len(seeds),'updated':time.time()})
            if len(records)%10==0:print(json.dumps({'rollout':path.name,'completed':len(records),'total':len(seeds)}),flush=True)
    finally:env.close()
    outcomes=[r['result'] for r in records];trace=[t for r in records for t in r['transitions']]
    atomic_jsonl(path,outcomes);atomic_jsonl(path.with_suffix('.transitions.jsonl'),trace)
    atomic(path.with_suffix('.complete.json'),{'identity':identity,'episodes':len(outcomes),'transitions':len(trace),'transactional':True,'completed':time.time()})
    return outcomes

def summary(rows):
    n=len(rows)
    return {'n':n,'success_rate':sum(r['success'] for r in rows)/n,'mean_steps':sum(r['steps'] for r in rows)/n,
            'mean_seconds':sum(r['seconds'] for r in rows)/n,'effective_seconds':effective_cost(rows),
            'wrong_submits':sum(r['wrong_submit'] for r in rows),'repeated_clicks':sum(r['repeated_clicks'] for r in rows)}

def branch(env,seed,prefix,before,action,step,source):
    wall_started=time.time()
    env.replay(seed,prefix,before);after,seconds=env.step(action)
    row=label_transition(before,action,after,seconds,f'{seed}:branch',step)
    row.update(seed=seed,prefix=prefix,source=source,expert_continuation=True)
    continuation=[];obs=after
    while not obs['done']:
        next_action=rule_action(obs);next_obs,elapsed=env.step(next_action)
        continuation.append(label_transition(obs,next_action,next_obs,elapsed,f'{seed}:continuation',step+len(continuation)+1));obs=next_obs
    success=int(bool(obs['submitted'] and correctness(obs)==1. and not obs.get('budget_exhausted')))
    gamma=json.loads((ROOT/'protocol.json').read_text())['discount']
    value=(gamma**len(continuation))*(1. if success else -1.)
    if not after['done']:value+=.1*(correctness(after)-correctness(before))
    row.update(continuation=continuation,continuation_final=obs,continuation_success=success,return_target=value,started=wall_started,completed=time.time())
    return row

def collect_initial(seeds,path):
    path=Path(path);commits=path.with_suffix('.instances');commits.mkdir(parents=True,exist_ok=True);env=GUIEnv();all_rows=[];policy_rows=[]
    try:
        for seed in seeds:
            file=commits/f'{seed}.json'
            if file.exists():record=json.loads(file.read_text())
            else:
                start=env.reset(seed);states=[([],start)]
                action=random.Random(seed).choice(legal_actions(start)[:-1]);obs,_=env.step(action);states.append(([action],obs))
                obs=env.reset(seed);prefix=[];demonstration=[]
                while not obs['done']:
                    action=rule_action(obs);after,elapsed=env.step(action)
                    row=label_transition(obs,action,after,elapsed,f'{seed}:expert',len(prefix));row['episode_success']=1;demonstration.append(row)
                    if action['kind']=='submit':states.append((list(prefix),obs));break
                    prefix.append(action);obs=after
                correct_prefix,correct=states[-1]
                if correct['remaining_actions']>1:
                    env.replay(seed,correct_prefix,correct);wrong={'kind':'click','slot':seed%len(correct['elements'])};after,_=env.step(wrong)
                    states.append((correct_prefix+[wrong],after))
                collected=[]
                for prefix,before in states:
                    if before['done']:continue
                    for action in legal_actions(before):collected.append(branch(env,seed,prefix,before,action,len(prefix),'initial_expert_and_random_states'))
                record={'seed':seed,'branches':collected,'expert_policy_rows':demonstration};atomic(file,record)
            all_rows+=record['branches'];policy_rows+=record['expert_policy_rows']
            atomic(path.with_suffix('.progress.json'),{'pid':__import__('os').getpid(),'completed':len(list(commits.glob('*.json'))),'total':len(seeds)})
    finally:env.close()
    unique=list({r['pair_hash']:r for r in all_rows}.values())
    atomic_jsonl(path,unique);atomic_jsonl(path.with_name(path.stem+'_policy.jsonl'),policy_rows)
    atomic(path.with_suffix('.manifest.json'),{'seeds':list(seeds),'rows':len(unique),'before_dedup':len(all_rows),'data_hash':digest([r['pair_hash'] for r in unique]),'expert_assisted_training_only':True,'actual_browser_branches':True})
    return unique

def collect_online(rollout_path,output):
    output=Path(output);folder=output.with_suffix('.instances');folder.mkdir(parents=True,exist_ok=True)
    trace=read_jsonl(Path(rollout_path).with_suffix('.transitions.jsonl'));by_seed={}
    for row in trace:by_seed.setdefault(int(row['episode_id'].split(':')[0]),[]).append(row)
    env=GUIEnv();collected=[]
    try:
        for seed,trajectory in by_seed.items():
            file=folder/f'{seed}.json'
            if file.exists():record=json.loads(file.read_text())
            else:
                prefix=[];branches=[]
                for row in trajectory:
                    for action in legal_actions(row['before']):branches.append(branch(env,seed,list(prefix),row['before'],action,row['step'],'online_counterfactual'))
                    prefix.append(row['action'])
                record={'seed':seed,'branches':branches};atomic(file,record)
            collected+=record['branches']
            atomic(output.with_suffix('.progress.json'),{'pid':__import__('os').getpid(),'completed':len(list(folder.glob('*.json'))),'total':len(by_seed)})
    finally:env.close()
    unique=list({r['pair_hash']:r for r in collected}.values());atomic_jsonl(output,unique)
    atomic(output.with_suffix('.manifest.json'),{'rows':len(unique),'seeds':list(by_seed),'source_rollout':str(rollout_path),'data_hash':digest([r['pair_hash'] for r in unique]),'collected_after_entire_rollout':True,'expert_continuation_training_only':True})
    return unique

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initial']);args=p.parse_args()
    splits=json.loads((ROOT/'protocol.json').read_text())['splits']
    collect_initial(splits['initial'],ROOT/'data/initial_world.jsonl')

if __name__=='__main__':main()
