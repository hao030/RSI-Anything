"""Independent raw-data, control, lineage and saved-tensor completion audit."""
import collections,hashlib,json,math,time
from pathlib import Path
from .settings import source_hashes
from .common import ROOT,atomic,file_hash,digest,read_jsonl

errors=[];counts=collections.Counter()
def check(condition,message):
    counts['checks']+=1
    if not condition:errors.append(message)

def goals(obs):
    text=obs['instruction'][7:-18];wanted=set() if text=='nothing' else set(text.split(', '))
    return [e['label'] in wanted for e in obs['elements']]

def fraction(obs):return sum(e['checked']==g for e,g in zip(obs['elements'],goals(obs)))/len(obs['elements'])
def strict(obs):return int(bool(obs['done'] and obs['submitted'] and fraction(obs)==1. and not obs.get('budget_exhausted')))
def actions(obs):return [{'kind':'click','slot':e['slot']} for e in obs['elements']]+[{'kind':'submit'}]
def sig(x):return 1/(1+math.exp(-max(-60.,min(60.,x))))

def transition(row,where):
    before,after,action=row['before'],row['after'],row['action'];n=len(before['elements'])
    check(action in actions(before),where+': action illegal')
    check(before['instruction']==after['instruction'] and len(after['elements'])==n,where+': identity changed')
    for i,(a,b) in enumerate(zip(before['elements'],after['elements'])):
        check((a['slot'],a['ref'],a['label'])==(b['slot'],b['ref'],b['label']),where+': element changed')
        check(b['checked']==(not a['checked'] if action=={'kind':'click','slot':i} else a['checked']),where+': browser checkbox dynamics wrong')
    check(after['remaining_actions']==before['remaining_actions']-1,where+': budget wrong')
    if action['kind']=='submit':check(after['submitted'] and after['done'],where+': Submit not terminal')
    expected=json.dumps({'instruction':before['instruction'],'visible_elements':before['elements'],'remaining_actions':before['remaining_actions'],'candidate_action':action},sort_keys=True,ensure_ascii=False)
    target=json.dumps({'instruction':after['instruction'],'visible_elements':after['elements'],'terminated':after['done'],'submitted':after['submitted'],'remaining_actions':after['remaining_actions']},sort_keys=True,ensure_ascii=False)
    check(row['context']==expected and row['target']==target,where+': model input/target invalid')
    check(row['pair_hash']==digest([expected,target]),where+': transition identity wrong')
    utility=(1. if strict(after) else -1.) if after['done'] else fraction(after)-fraction(before)
    check(row['success']==strict(after) and abs(row['utility']-utility)<1e-8,where+': label wrong')
    check(row['checked']==[int(e['checked']) for e in after['elements']] and row['done']==int(after['done']),where+': supervised labels wrong')
    check(row['group_id']==digest([before['instruction'],before['elements'],before['remaining_actions']]),where+': group identity wrong')
    counts['transitions']+=1

def branch(row,where,discount):
    transition(row,where);previous=row['after']
    for r in row['continuation']:
        check(r['before']==previous,where+': expert continuation chain broken');transition(r,where+':continuation');previous=r['after']
    check(previous==row['continuation_final'] and previous['done'],where+': continuation did not actually end')
    success=strict(previous);value=discount**len(row['continuation'])*(1. if success else -1.)
    if not row['after']['done']:value+=.1*(fraction(row['after'])-fraction(row['before']))
    check(row['continuation_success']==success and abs(value-row['return_target'])<1e-8,where+': discounted actual outcome wrong')
    counts['counterfactual_branches']+=1

def episode_file(path,seeds,adapter=None,frozen_after=None):
    outcomes=read_jsonl(path);trace=read_jsonl(Path(path).with_suffix('.transitions.jsonl'));groups=collections.defaultdict(list)
    check([r['seed'] for r in outcomes]==list(seeds),str(path)+': incomplete/wrong instance cohort')
    for row in trace:
        transition(row,str(path));groups[row['episode_id']].append(row)
        proposal=row.get('proposal')
        if proposal:
            legal=actions(row['before']);order=proposal['all_action_ids'];pool=order[:4]
            if len(legal)-1 not in pool:pool=pool[:3]+[len(legal)-1]
            check(sorted(order)==list(range(len(legal))) and proposal['candidate_ids']==pool and proposal['actions']==[legal[i] for i in pool],str(path)+': candidate pool changed')
            check(abs(sum(proposal['probabilities'])-1)<1e-5,str(path)+': policy probabilities invalid')
            selection=row['selection'];alpha=selection['alpha']
            if alpha==0:
                check(row['action']==proposal['actions'][0] and row.get('forecast') is None,str(path)+': fallback bypassed base policy')
            else:
                prediction=row['forecast'];calibration=selection['terminal'];values=[]
                for action,p in zip(proposal['actions'],prediction):
                    values.append(2*sig(p['submit_success_logit']/calibration['temperature']+calibration['bias'])-1 if action['kind']=='submit' else p['value'])
                mean=sum(values)/len(values);sd=(sum((v-mean)**2 for v in values)/len(values))**.5
                scores=values if alpha=='pure' else [math.log(max(proposal['probabilities'][i],1e-12))+alpha*(v-mean)/max(sd,.1) for i,v in zip(pool,values)]
                index=max(range(len(scores)),key=lambda i:scores[i])
                check(row['action']==proposal['actions'][index] and all(abs(a-b)<1e-7 for a,b in zip(selection['scores'],scores)),str(path)+': world control does not match committed predictions')
    for outcome in outcomes:
        steps=groups[outcome['episode_id']]
        check(len(steps)==outcome['steps']<=8 and outcome['actions']==[r['action'] for r in steps],str(path)+': step/action trace mismatch')
        initial=outcome.get('initial',steps[0]['before'] if steps else None)
        check(initial is not None,str(path)+': initial observation missing')
        if initial is not None:
            check(outcome['initial_hash']==digest({k:initial[k] for k in ['instruction','elements']}),str(path)+': initial state identity wrong')
            if steps:check(initial==steps[0]['before'],str(path)+': first transition differs from reset')
        termination=outcome.get('termination')
        if termination:
            before=steps[-1]['after'] if steps else initial
            final=outcome['final'];reason=termination['reason']
            check(termination['before']==before and termination['after']==final,str(path)+': termination observation chain broken')
            check(final['done'] and not final['submitted'] and outcome['success']==0,str(path)+': invalid unapplied-action termination')
            check(reason in {'timeout','action_budget','environment'} and reason==final.get('termination_reason','environment'),str(path)+': termination reason differs')
            if before is not None:
                check(all(before[k]==final[k] for k in ['instruction','elements','remaining_actions','submitted']),str(path)+': state changed without an action')
            if reason=='timeout':check(final.get('elapsed_seconds',0)>=120,str(path)+': timeout before episode deadline')
            if reason=='action_budget':check(final['remaining_actions']==0,str(path)+': budget termination with actions remaining')
        else:
            check(bool(steps) and outcome['final']==steps[-1]['after'],str(path)+': final observation differs from last action')
        check(outcome['success']==strict(outcome['final']),str(path)+': strict completion wrong')
        check(outcome['adapter']==adapter and all(r['episode_success']==outcome['success'] for r in steps),str(path)+': policy or successful-episode lineage wrong')
        for a,b in zip(steps,steps[1:]):check(a['after']==b['before'],str(path)+': episode chain broken')
        if frozen_after is not None:check(outcome['started']>=frozen_after,str(path)+': test opened before freezing')
        if outcome['method']=='hybrid' and outcome['controller']['alpha']==0:check(outcome['world_seconds']==0,str(path)+': fallback secretly uses world')
        commit=Path(path).with_suffix('.episodes')/f"{outcome['seed']}.json"
        record=json.loads(commit.read_text());check(record['result']==outcome and record['transitions']==steps,str(path)+': transactional commit mismatch')
    counts['episodes']+=len(outcomes);counts['episode_files']+=1
    return {'n':len(outcomes),'success_rate':sum(r['success'] for r in outcomes)/len(outcomes),
            'effective_seconds':sum(r['policy_nominal_seconds']+r['world_seconds']+r['browser_seconds'] for r in outcomes)/len(outcomes)}

def compare_summary(actual,claimed,where):
    for key,value in actual.items():check(abs(value-claimed[key])<1e-8,where+': summary '+key+' differs from raw episodes')

def controller_files(folder,data,protocol,adapter=None):
    folder=Path(folder)
    compare_summary(episode_file(folder/'baseline_dev.jsonl',protocol['splits']['dev'],adapter),data['baseline'],str(folder))
    for candidate in data['candidates']:
        alpha=candidate['controller']['alpha']
        compare_summary(episode_file(folder/f'alpha{alpha}_dev.jsonl',protocol['splits']['dev'],adapter),candidate['result'],str(folder))

def online_lineage(path,rows,where):
    outcomes=read_jsonl(path);trace=read_jsonl(Path(path).with_suffix('.transitions.jsonl'));states={};last=max(r['completed'] for r in outcomes)
    prefixes=collections.defaultdict(list)
    for r in trace:
        key=(int(r['episode_id'].split(':')[0]),r['group_id'])
        states[key]=(r['before'],list(prefixes[key[0]]))
        prefixes[key[0]].append(r['action'])
    observed=collections.defaultdict(set)
    for r in rows:
        key=(r['seed'],r['group_id']);check(key in states,where+': counterfactual state was not visited by acting policy')
        if key in states:
            before,prefix=states[key];check(r['before']==before and r['prefix']==prefix,where+': counterfactual replay is not the actual acting prefix')
        check(r['started']>=last,where+': counterfactual feedback opened before all acting episodes finished')
        observed[key].add(json.dumps(r['action'],sort_keys=True))
    check(set(observed)==set(states),where+': visited states missing counterfactual feedback')
    for key,(before,prefix) in states.items():check(observed[key]=={json.dumps(a,sort_keys=True) for a in actions(before)},where+': not all legal branches were collected')

def controller(data,protocol,where):
    eligible=[]
    for candidate in data['candidates']:
        gain=candidate['result']['success_rate']-data['baseline']['success_rate'];ratio=candidate['result']['effective_seconds']/data['baseline']['effective_seconds']
        valid=gain>=protocol['controller_enable_gain'] and ratio<=protocol['controller_enable_cost_ratio']
        check(candidate['eligible']==valid,where+': controller gate wrong')
        if valid:eligible.append(candidate)
    if not eligible:check(data['fallback'] and data['controller']['alpha']==0,where+': missing policy fallback')
    else:
        best=max(eligible,key=lambda c:(c['result']['success_rate'],-c['result']['effective_seconds'],-c['controller']['alpha']))
        check(data['controller']==best['controller'] and not data['fallback'],where+': wrong controller selected')

def changed(old,new,where):
    import torch
    from safetensors.torch import load_file
    a=torch.load(Path(old)/'gui_state.pt',map_location='cpu',weights_only=True);b=torch.load(Path(new)/'gui_state.pt',map_location='cpu',weights_only=True)
    for prefix in ['online.projection','predictor','readout','value_head','state_head','feature_encoder']+(['opf'] if any(k.startswith('opf.') for k in a) else []):
        keys=[k for k in a if k.startswith(prefix+'.')]
        check(bool(keys) and any(not torch.equal(a[k],b[k]) for k in keys),where+': '+prefix+' did not change')
    for name in ['online_lora','ema_lora']:
        if (Path(old)/name).exists():
            a=load_file(str(Path(old)/name/'adapter_model.safetensors'));b=load_file(str(Path(new)/name/'adapter_model.safetensors'))
            check(any(not torch.equal(a[k],b[k]) for k in a),where+': '+name+' did not change')

def training(path,allowed):
    data=json.loads((Path(path)/'training.json').read_text())
    check(all(h in allowed for h in data['data_pair_hashes']),str(path)+': training lineage not actual permitted branches')
    check(data['full_coverage'] and data['minimum_row_exposures']>=data['epochs'],str(path)+': incomplete training')
    check(data['encoder_changed'] and (data['ema_changed'] or data['mode']=='ce'),str(path)+': encoder/EMA update missing')
    critic=data['critic_fit'];protocol=json.loads((ROOT/'protocol.json').read_text())
    check(critic['configuration']==protocol['critic_fit'] and critic['data_pair_hashes']==data['data_pair_hashes'],str(path)+': completion critic fit configuration/data changed')
    check(critic['full_coverage'] and critic['minimum_row_exposures']>=protocol['critic_fit']['epochs'],str(path)+': completion critic did not cover all training rows')
    check(all(file_hash(Path(path)/name)==sha for name,sha in data['files'].items()),str(path)+': checkpoint corrupted')
    counts['world_checkpoints']+=1

def calibration(path,protocol):
    data=json.loads(Path(path).read_text());predictions=read_jsonl(Path(path).with_suffix('.predictions.jsonl'))
    check(all(p['seed'] in protocol['splits']['terminal_calibration'] for p in predictions),str(path)+': calibration leaked another split')
    check(data['data_hash']==digest([p['pair_hash'] for p in predictions]),str(path)+': calibration sample identity changed')
    options=[]
    for temp in protocol['terminal_calibration_grid']['temperature']:
        for bias in protocol['terminal_calibration_grid']['bias']:
            mse=sum((sig(p['logit']/temp+bias)-p['success'])**2 for p in predictions)/len(predictions)
            options.append({'temperature':temp,'bias':bias,'brier':mse})
    best=min(options,key=lambda x:(x['brier'],abs(x['bias']),abs(x['temperature']-1)))
    check(all(abs(best[k]-data['selected'][k])<1e-8 for k in best),str(path)+': selected calibration not predeclared optimum')

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());splits=protocol['splits'];discount=protocol['discount']
    check(source_hashes()==protocol['source_hashes'],'Frozen source changed')
    check(len([s for v in splits.values() for s in v])==len({s for v in splits.values() for s in v}),'Instance cohorts overlap')
    for name in ['browser_contracts.json','world_contracts.json']:check(json.loads((ROOT/'results'/name).read_text())['passed'],'Required behavior preflight failed')
    initial=read_jsonl(ROOT/'data/initial_world.jsonl');allowed={r['pair_hash'] for r in initial}
    for r in initial:branch(r,'initial',discount);check(r['seed'] in splits['initial'],'Initial data wrong split')
    for r in read_jsonl(ROOT/'data/terminal_calibration.jsonl'):branch(r,'calibration',discount);check(r['seed'] in splits['terminal_calibration'],'Calibration wrong split')
    for method in ['policy','rule']:
        for split in ['dev','test']:episode_file(ROOT/f'results/baseline_{method}_{split}.jsonl',splits[split])
    gains=[]
    for mode,seed in protocol['world_runs']:
        name=f'{mode}_seed{seed}';folder=ROOT/'results'/name;completion=json.loads((folder/'complete.json').read_text())
        episode_file(folder/'update.jsonl',splits['update'])
        new=read_jsonl(folder/'update.branches.jsonl')
        online_lineage(folder/'update.jsonl',new,name)
        last_rollout=max(r['completed'] for r in read_jsonl(folder/'update.jsonl'))
        for r in new:
            branch(r,name,discount);check(r['seed'] in splits['update'],name+': update includes heldout instance')
            check(r['started']>=last_rollout,name+': branch outcomes available during acting')
        lineage=json.loads((folder/'replay.json').read_text());expected=lineage['new_pair_hashes']+lineage['replay_pair_hashes']
        check(len(lineage['new_pair_hashes'])==len(lineage['replay_pair_hashes']),name+': replay is not 1:1')
        permitted=allowed|{r['pair_hash'] for r in new}
        for version,checkpoint in [('frozen','w0'),('updated','w1')]:
            path=ROOT/'checkpoints'/name/checkpoint;training(path,allowed if checkpoint=='w0' else permitted)
            meta=json.loads((path/'training.json').read_text())
            if checkpoint=='w1':check(meta['data_pair_hashes']==expected,name+': exact replay data changed')
            frozen=json.loads((folder/f'{version}_frozen.json').read_text());controller(frozen['controller'],protocol,name+version)
            controller_files(folder/('collection_selection' if version=='frozen' else 'updated_selection'),frozen['controller'],protocol)
            compare_summary(episode_file(folder/f'{version}_pure_dev.jsonl',splits['dev']),frozen['pure_dev'],name+':pure dev')
            calibration(folder/('collection_terminal.json' if version=='frozen' else 'updated_terminal.json'),protocol)
            freeze_time=json.loads((folder/'all_versions_frozen.json').read_text())['time']
            for kind in ['pure','selected']:compare_summary(episode_file(folder/f'{version}_{kind}_test.jsonl',splits['test'],frozen_after=freeze_time),completion['outcomes'][version][kind]['test'],name+version+kind)
            diagnostic=read_jsonl(folder/f'{version}_diagnostic.jsonl');check([r['seed'] for r in diagnostic]==splits['diagnostic'],name+': diagnostics incomplete')
            for record in diagnostic:
                check(record['prediction_time']<=record['completed'],name+': forecast not committed before branches')
                committed=json.loads((folder/f'{version}_diagnostic.instances'/f"{record['seed']}.predictions.json").read_text())
                check(committed['forecasts']==record['forecasts'] and committed['time']==record['prediction_time'],name+': pre-branch forecast commit differs')
                check(len(record['branches'])==len(record['forecasts'])==len(record['proposal']['actions']),name+': diagnostic branch/prediction count differs')
                for actual,prediction in zip(record['branches'],record['forecasts']):
                    branch(actual['actual'],name+':diagnostic',discount);check(actual['prediction']==prediction,name+': diagnostic predictions changed')
                    check(actual['actual']['started']>=record['prediction_time'],name+': actual branch executed before forecast commit')
                    row=actual['actual'];bits=[bool(x) for x in row['checked']];predicted=[p>=.5 for p in prediction['checked_probabilities'][:len(bits)]]
                    check(actual['state_exact']==int(bits==predicted),name+': reported next-state accuracy incorrect')
                    if row['action']['kind']=='click':check(actual['clicked_correct']==int(predicted[row['action']['slot']]==bits[row['action']['slot']]),name+': clicked-slot accuracy incorrect')
                    check(abs(actual['value_mse']-(prediction['value']-row['return_target'])**2)<1e-8,name+': value MSE incorrect')
            counts['diagnostic_instances']+=len(diagnostic)
        changed(ROOT/'checkpoints'/name/'w0',ROOT/'checkpoints'/name/'w1',name)
        olddev=completion['outcomes']['frozen']['selected']['dev'];newdev=completion['outcomes']['updated']['selected']['dev']
        promoted=newdev['success_rate']-olddev['success_rate']>=protocol['promotion_gain'] and newdev['effective_seconds']<=olddev['effective_seconds']*protocol['promotion_cost_ratio']
        check(completion['promotion']['promoted']==promoted,name+': version promotion gate incorrect')
        gains.append(completion['outcomes']['updated']['pure']['test']['success_rate']-completion['outcomes']['frozen']['pure']['test']['success_rate']) if mode=='opf' else None
    sealed=json.loads((ROOT/'results/rsi_versions_frozen.json').read_text())
    for generation in [1,2]:
        folder=ROOT/'results'/f'rsi_g{generation}';m=json.loads((folder/'versions_frozen.json').read_text());episode_file(folder/'update.jsonl',splits[f'rsi_update{generation}'],m['old_policy'])
        new=read_jsonl(folder/'update.branches.jsonl')
        online_lineage(folder/'update.jsonl',new,f'RSI {generation}')
        for r in new:branch(r,'RSI',discount);check(r['seed'] in splits[f'rsi_update{generation}'],'RSI training split wrong')
        training(m['new_world'],allowed|{r['pair_hash'] for r in new});changed(m['old_world'],m['new_world'],f'RSI {generation}')
        calibration(folder/'terminal_calibration.json',protocol)
        controller(m['selection'],protocol,'RSI')
        controller_files(folder/'selection',m['selection'],protocol,m['new_policy'])
        for arm,p in [('old_old',m['old_policy']),('new_new',m['new_policy'])]:compare_summary(episode_file(folder/f'{arm}_selected_dev.jsonl',splits['dev'],p),m['dev'][arm],f'RSI {generation} '+arm)
        olddev,newdev=m['dev']['old_old'],m['dev']['new_new']
        check(m['promoted']==(newdev['success_rate']-olddev['success_rate']>=protocol['promotion_gain'] and newdev['effective_seconds']<=olddev['effective_seconds']*protocol['promotion_cost_ratio']),f'RSI {generation}: promotion gate incorrect')
        for arm,p in [('old_old',m['old_policy']),('old_new',m['old_policy']),('new_old',m['new_policy']),('new_new',m['new_policy'])]:
            for kind in ['pure','selected']:episode_file(folder/f'{arm}_{kind}_test.jsonl',splits['rsi_test'],p,sealed['time'])
        for arm,p in [('old_policy',m['old_policy']),('new_policy',m['new_policy'])]:episode_file(folder/f'{arm}_test.jsonl',splits['rsi_test'],p,sealed['time'])
        policy=ROOT/'checkpoints'/f'rsi_g{generation}'/'policy';metadata=policy/'training.json'
        if metadata.exists():
            data=json.loads(metadata.read_text());onpolicy=read_jsonl(folder/'update.transitions.jsonl');replay=read_jsonl(ROOT/'data/initial_world_policy.jsonl')
            sources={r['pair_hash']:r for r in onpolicy+replay}
            used=[sources[h] for h in data['example_pair_hashes']]
            check(all(r['episode_success']==1 and r['utility']>0 for r in used),'RSI policy learned from failed episode')
            check(len(data['new_pair_hashes'])==len(data['replay_pair_hashes']) and data['actual_weight_change'],'RSI policy replay or update missing')
            check(data['data_hash']==digest(data['example_pair_hashes']),'RSI policy sample identity wrong')
            import torch
            from safetensors.torch import load_file
            weights=load_file(str(policy/'adapter_model.safetensors'))
            if m['old_policy']:
                old=load_file(str(Path(m['old_policy'])/'adapter_model.safetensors'))
                check(any(not torch.equal(weights[k],old[k]) for k in weights),'RSI actual policy tensors unchanged')
            else:check(any(bool(torch.count_nonzero(v)) for k,v in weights.items() if 'lora_B' in k),'RSI actual policy LoRA stayed zero')
        else:check((policy/'not_updated.json').exists(),'RSI no-update attempt not documented')
    decision=json.loads((ROOT/'results/confirmation_decision.json').read_text())
    check(decision['run_confirmation']==(sum(gains)/3>=protocol['confirmation_threshold']),'Confirmation threshold changed')
    if decision['run_confirmation']:
        for seed in [0,1,2]:
            for version in ['frozen','updated']:episode_file(ROOT/f'results/opf_seed{seed}/{version}_confirmation.jsonl',splits['confirmation'])
    report={'passed':not errors,'errors':errors,'counts':dict(counts),'source_hashes_verified':True,'auditor_sha256':file_hash(__file__),'time':time.time()}
    atomic(ROOT/'outputs/completion_audit.json',report);print(json.dumps(report),flush=True)
    if errors:raise RuntimeError('Full independent completion audit failed')

if __name__=='__main__':main()
