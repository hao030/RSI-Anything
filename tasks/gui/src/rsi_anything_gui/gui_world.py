"""G6 encoder, EMA and vendored OPF, with GUI state and action-effect supervision."""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .common import ROOT,BASE_WORLD,atomic,digest,file_hash

from .world import DomainEncoder,ExecutionWorld,D,K,R
from jepa_anything_core import jepa_anything_objective,encoder_variance_loss

def batch_indices(rows,seed,batch_tasks=4):
    """Pair contrasting actual branches while retaining four independent instances."""
    from collections import defaultdict
    rng=random.Random(seed);tasks=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):tasks[r['episode_id'].split(':')[0]][r['group_id']].append(i)
    queues={}
    for task,groups in tasks.items():
        units=[]
        for indices in groups.values():
            indices=list(indices)
            rng.shuffle(indices);indices.sort(key=lambda i:rows[i].get('return_target',rows[i]['utility']))
            while len(indices)>1:units.append([indices.pop(0),indices.pop()])
            if indices:units.append(indices)
        rng.shuffle(units);queues[task]=units
    all_tasks=sorted(queues)
    if len(all_tasks)<batch_tasks:raise RuntimeError('Need four independent instances')
    while any(queues.values()):
        active=[t for t in all_tasks if queues[t]]
        chosen=rng.sample(active,min(batch_tasks,len(active)))
        if len(chosen)<batch_tasks:chosen+=rng.sample([t for t in all_tasks if t not in chosen],batch_tasks-len(chosen))
        batch=[]
        for task in chosen:
            if queues[task]:batch.extend(queues[task].pop())
            else:batch.append(rng.choice([i for group in tasks[task].values() for i in group]))
        yield batch

class GUIEncoder(DomainEncoder):
    """Encode public state compactly; the causal last token observes the whole state."""
    def forward(self,texts):
        rendered=[]
        for text in texts:
            obs=json.loads(text)
            parts=['Task instruction: '+obs['instruction'],'Visible checkbox states:']
            for e in obs['visible_elements']:parts.append(f"{e['slot']}: {e['label']} = {'checked' if e['checked'] else 'unchecked'}")
            parts.append('Remaining actions: '+str(obs['remaining_actions']))
            if 'candidate_action' in obs:
                a=obs['candidate_action'];parts.append('Candidate action: '+('Submit' if a['kind']=='submit' else f"click checkbox {a['slot']} ({obs['visible_elements'][a['slot']]['label']})"))
            else:parts.append(f"Submitted: {obs['submitted']}; terminated: {obs['terminated']}")
            parts.append('State representation:')
            rendered.append('\n'.join(parts))
        batch=self.tokenizer(rendered,padding=True,truncation=False,return_tensors='pt',add_special_tokens=False)
        if batch['input_ids'].shape[-1]>self.max_length:raise RuntimeError('GUI input exceeds frozen token limit')
        device=next(self.backbone.parameters()).device;batch={k:v.to(device) for k,v in batch.items()}
        hidden=self.backbone(**batch,return_dict=True,use_cache=False).last_hidden_state
        positions=batch['attention_mask'].sum(1)-1
        pooled=hidden[torch.arange(len(texts),device=device),positions].float()
        return self.projection(pooled)

class GUIWorld(ExecutionWorld):
    def __init__(self,mode='opf',training=True):
        self.gui_mode=mode
        self.use_jepa=mode!='ce'
        super().__init__(GUIEncoder(BASE_WORLD,max_length=1024),mode='standard' if mode=='standard' else 'opf',ema=.996,include_target=training and self.use_jepa)
        self.readout=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,64),nn.GELU(),nn.Linear(64,8))
        self.state_head=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,1))
        self.feature_encoder=nn.Sequential(nn.Linear(20,64),nn.GELU(),nn.LayerNorm(64))
        self.value_head=nn.Sequential(nn.Linear(2*D+70,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
        self.submit_positive_weight=1.

    def outputs(self,texts):
        c=self.encode(texts);latent,factors=self.predict_latent(c);state=self.readout(latent)
        features=[]
        for text in texts:
            obs=json.loads(text);elements=obs['visible_elements'];action=obs['candidate_action']
            bits=[float(e['checked']) for e in elements]+[0.]*(6-len(elements))
            mask=[1.]*len(elements)+[0.]*(6-len(elements))
            ids=[0.]*7;ids[6 if action['kind']=='submit' else action['slot']]=1.
            features.append(bits+mask+ids+[obs['remaining_actions']/8.])
        visible=torch.tensor(features,device=state.device)
        slot_input=torch.stack([visible[:,:6],visible[:,12:18]],dim=-1)
        state_logits=self.state_head(slot_input).squeeze(-1)+.1*state[:,:6]
        state=torch.cat([state_logits,state[:,6:]],dim=1)
        values=self.value_head(torch.cat([F.layer_norm(c,(D,)),F.layer_norm(latent,(D,)),torch.sigmoid(state[:,:6]),self.feature_encoder(visible)],dim=1))
        return c,latent,factors,state,values

    def optimizer(self,slow=True):
        groups={'lora':[],'dynamics':[],'readouts':[],'representation':[],'opf':[]}
        for name,p in self.named_parameters():
            if not p.requires_grad or name=='log_temperature':continue
            group='lora' if name.startswith('online.backbone') else 'dynamics' if name.startswith('state_head') else 'opf' if name.startswith('opf') else 'readouts' if name.startswith(('readout','value_head','feature_encoder')) else 'representation'
            groups[group].append(p)
        rates={'lora':5e-6,'dynamics':1e-2,'readouts':1e-3,'representation':1e-4,'opf':1e-5}
        return torch.optim.AdamW([{'params':p,'lr':rates[k]} for k,p in groups.items() if p],weight_decay=.01)

    def training_loss(self,rows):
        contexts,latent,factors,out,values=self.outputs([r['context'] for r in rows])
        n=len(rows)
        bits=torch.zeros(n,6,device=out.device);mask=torch.zeros_like(bits)
        for i,r in enumerate(rows):
            bits[i,:len(r['checked'])]=torch.tensor(r['checked'],device=out.device)
            mask[i,:len(r['checked'])]=1
        done=torch.tensor([r['done'] for r in rows],device=out.device,dtype=torch.float)
        success=torch.tensor([r['success'] for r in rows],device=out.device,dtype=torch.float)
        utility=torch.tensor([r['utility'] for r in rows],device=out.device,dtype=torch.float)
        returns=torch.tensor([r['return_target'] for r in rows],device=out.device,dtype=torch.float)
        seconds=torch.tensor([min(r['seconds']/2.,2.) for r in rows],device=out.device)
        state_loss=(F.binary_cross_entropy_with_logits(out[:,:6],bits,reduction='none')*mask).sum()/mask.sum()
        done_loss=F.binary_cross_entropy_with_logits(out[:,6],done)
        submitted=torch.tensor([r['action']['kind']=='submit' for r in rows],device=out.device,dtype=torch.float)
        success_loss=(F.binary_cross_entropy_with_logits(values[:,0],success,reduction='none',pos_weight=out.new_tensor(self.submit_positive_weight))*submitted).sum()/submitted.sum().clamp_min(1)
        weights=1+submitted
        utility_loss=(F.smooth_l1_loss(values[:,1],utility,reduction='none')*weights).sum()/weights.sum()
        value_loss=(F.smooth_l1_loss(values[:,2],returns,reduction='none')*weights).sum()/weights.sum()
        resource=F.smooth_l1_loss(F.softplus(out[:,7]),seconds)
        ranking=out.sum()*0;pairs=0
        for i,a in enumerate(rows):
            for j,b in enumerate(rows[:i]):
                if a['group_id']==b['group_id'] and abs(a['return_target']-b['return_target'])>1e-6:
                    good,bad=(i,j) if a['return_target']>b['return_target'] else (j,i)
                    ranking+=F.relu(.1-values[good,2]+values[bad,2]);pairs+=1
        ranking=ranking/max(pairs,1)
        representation=out.sum()*0
        parts={'state':state_loss,'done':done_loss,'success':success_loss,'utility':utility_loss,'value':value_loss,'resource':resource,'ranking':ranking}
        if self.target is not None:
            with torch.no_grad():z=self.encode([r['target'] for r in rows],target=True)
            if self.opf is not None:
                target_factors=self.opf(z.detach())
                objective=jepa_anything_objective(factors,target_factors,self.opf.analysis_basis(),contexts)
                representation=objective.total
                parts.update(prediction=objective.prediction,orthogonality=objective.orthogonality,factor_activity=objective.factor_activity,encoder_variance=objective.encoder_variance)
            else:
                representation=F.mse_loss(latent,z.detach())+encoder_variance_loss(contexts)
                parts['prediction']=F.mse_loss(latent,z.detach())
        return representation+state_loss+done_loss+success_loss+utility_loss+value_loss+.1*resource+.2*ranking,parts,contexts,factors

    @torch.no_grad()
    def forecast(self,contexts):
        self.eval()
        start=time.monotonic()
        _,_,_,out,values=self.outputs(contexts);out=out.float();values=values.float()
        rows=[]
        for x,v in zip(out,values):
            rows.append({'checked_probabilities':torch.sigmoid(x[:6]).tolist(),'done_probability':float(torch.sigmoid(x[6])),
                         'submit_success_logit':float(v[0]),'success_probability':float(torch.sigmoid(v[0])),
                         'utility':float(v[1]),'value':float(v[2]),'seconds_prediction':float(F.softplus(x[7])*2)})
        return rows,time.monotonic()-start

    def save_gui(self,path,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.online.backbone.save_pretrained(path/'online_lora')
        if self.target is not None:self.target.backbone.save_pretrained(path/'ema_lora')
        state={k:v.cpu() for k,v in self.state_dict().items() if not k.startswith(('online.backbone','target.backbone'))}
        torch.save(state,path/'gui_state.pt')
        metadata.update(mode=self.gui_mode,dimensions={'d':D,'k':K,'r':R},schema='gui-jepa-transition-v1',ema=.996,base=str(BASE_WORLD),
                        trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),schema_version='gui-jepa-v2')
        metadata['files']={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file() and p.name!='training.json'}
        atomic(path/'training.json',metadata)

    @classmethod
    def load_gui(cls,path,training=False):
        path=Path(path);metadata=json.loads((path/'training.json').read_text())
        for name,sha in metadata['files'].items():
            if hashlib.sha256((path/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Checkpoint hash mismatch: '+name)
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        model=cls(metadata['mode'],training=training)
        for module,folder in [(model.online,'online_lora'),(model.target,'ema_lora')]:
            if module is not None:set_peft_model_state_dict(module.backbone,load_peft_weights(str(path/folder),device='cpu'),adapter_name='default')
        state=torch.load(path/'gui_state.pt',map_location='cpu',weights_only=True)
        if model.target is None:state={k:v for k,v in state.items() if not k.startswith('target.')}
        incompatible=model.load_state_dict(state,strict=False)
        if incompatible.unexpected_keys or any(not x.startswith(('online.backbone','target.backbone')) for x in incompatible.missing_keys):raise RuntimeError(str(incompatible))
        return model.cuda().train(training)

def parameter_hash(module):
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()

def checkpoint_reusable(path,rows,mode,seed,parent=None,replay_rows=None,epochs=3):
    path=Path(path);metadata=path/'training.json'
    if not metadata.exists():return False
    data=json.loads(metadata.read_text());expected={'mode':mode,'seed':seed,'parent':str(parent) if parent else None,
        'epochs':epochs,'data_pair_hashes':[r['pair_hash'] for r in rows],'replay_rows':len(replay_rows or [])}
    if any(data.get(k)!=v for k,v in expected.items()) or any(file_hash(path/name)!=sha for name,sha in data['files'].items()):
        raise RuntimeError('Existing checkpoint does not match the exact frozen training request: '+str(path))
    return True

def fit_critic(model,rows,seed):
    """Fit completion readouts on cached model predictions from training rows only.

    Holding the representation fixed prevents rare Submit feedback from chasing
    a changing encoder. No actual next bits enter the cached critic inputs.
    """
    protocol=json.loads((ROOT/'protocol.json').read_text())
    cfg=protocol['critic_fit'];model.eval();started=time.monotonic();inputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),32):
            chunk=rows[start:start+32]
            c,latent,_,state,_=model.outputs([r['context'] for r in chunk])
            inputs.append(torch.cat([F.layer_norm(c,(D,)),F.layer_norm(latent,(D,)),torch.sigmoid(state[:,:6])],dim=1).detach())
    encoded=torch.cat(inputs);features=[]
    for row in rows:
        obs=json.loads(row['context']);elements=obs['visible_elements'];action=obs['candidate_action'];ids=[0.]*7
        ids[6 if action['kind']=='submit' else action['slot']]=1.
        features.append([float(e['checked']) for e in elements]+[0.]*(6-len(elements))+[1.]*len(elements)+[0.]*(6-len(elements))+ids+[obs['remaining_actions']/8.])
    features=encoded.new_tensor(features)
    success=encoded.new_tensor([r['success'] for r in rows]);utility=encoded.new_tensor([r['utility'] for r in rows]);value=encoded.new_tensor([r['return_target'] for r in rows])
    positives=[i for i,r in enumerate(rows) if r['action']['kind']=='submit' and r['success']]
    negatives=[i for i,r in enumerate(rows) if r['action']['kind']=='submit' and not r['success']]
    if not positives or not negatives:raise RuntimeError('Completion critic needs actual positive and negative Submit training examples')
    submitted=torch.tensor([r['action']['kind']=='submit' for r in rows],device=encoded.device)
    groups=encoded.new_tensor([int(r['group_id'][:12],16) for r in rows],dtype=torch.long)
    optimizer=torch.optim.AdamW(list(model.value_head.parameters())+list(model.feature_encoder.parameters()),lr=cfg['lr'],weight_decay=.01)
    rng=random.Random(seed+10000);exposures=[0]*len(rows);last_loss=None;steps=0
    for epoch in range(cfg['epochs']):
        order=list(range(len(rows)));rng.shuffle(order)
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]
            if not any(i in positives for i in ids):ids.append(rng.choice(positives))
            if not any(i in negatives for i in ids):ids.append(rng.choice(negatives))
            for i in ids:exposures[i]+=1
            index=torch.tensor(ids,device=encoded.device);out=model.value_head(torch.cat([encoded[index],model.feature_encoder(features[index])],dim=1))
            pos=submitted[index]&(success[index]==1);neg=submitted[index]&(success[index]==0)
            terminal=.5*F.binary_cross_entropy_with_logits(out[pos,0],success[index][pos])+.5*F.binary_cross_entropy_with_logits(out[neg,0],success[index][neg])
            regression=F.mse_loss(out[:,1],utility[index])+F.mse_loss(out[:,2],value[index])
            same=groups[index,None]==groups[index][None,:];difference=value[index,None]-value[index][None,:];valid=same&(difference>1e-6)
            ranking=F.relu(.1-out[:,2,None]+out[:,2][None,:])[valid].mean() if valid.any() else out.sum()*0
            loss=terminal+regression+.2*ranking
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite completion critic loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.value_head.parameters())+list(model.feature_encoder.parameters()),1.,error_if_nonfinite=True);optimizer.step()
            last_loss=float(loss.detach());steps+=1
        if (epoch+1)%10==0:print(json.dumps({'critic_epoch':epoch+1,'critic_epochs':cfg['epochs'],'loss':last_loss,'seconds':time.monotonic()-started}),flush=True)
    model.train()
    return {'configuration':cfg,'rows':len(rows),'steps':steps,'seconds':time.monotonic()-started,'last_loss':last_loss,
            'minimum_row_exposures':min(exposures),'full_coverage':min(exposures)>=cfg['epochs'],
            'positive_submit_rows':len(positives),'negative_submit_rows':len(negatives),'input_source':'cached predicted next-state probabilities and context/latent from training rows; actual next bits excluded',
            'data_pair_hashes':[r['pair_hash'] for r in rows]}

def train_world(rows,path,mode,seed,parent=None,replay_rows=None,epochs=3):
    torch.set_num_threads(4);torch.manual_seed(seed);random.seed(seed)
    if len({r['episode_id'].split(':')[0] for r in rows})<4:raise RuntimeError('Training needs multiple independent instances')
    hashes=[r['pair_hash'] for r in rows]
    model=GUIWorld.load_gui(parent,training=True) if parent else GUIWorld(mode,training=True).cuda()
    model.train();torch.cuda.reset_peak_memory_stats()
    submits=[r for r in rows if r['action']['kind']=='submit']
    positive=sum(r['success'] for r in submits)
    model.submit_positive_weight=min(20.,max(1.,(len(submits)-positive)/max(positive,1)))
    optimizer=model.optimizer(slow=True)
    before=parameter_hash(model.online)
    ema_before=parameter_hash(model.target) if model.target else None
    # EMA is frozen: inspect its actual weights, rather than its requires_grad parameters.
    def ema_hash():
        if model.target is None:return None
        h=hashlib.sha256()
        for name,p in model.target.named_parameters():
            if 'lora_' in name or name.startswith('projection'):h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()
    ema_before=ema_hash()
    batches=[]
    started=time.monotonic();seen=set();last=None
    planned=[(epoch,ids) for epoch in range(epochs) for ids in batch_indices(rows,seed+epoch,batch_tasks=4)]
    exposures=[0]*len(rows);ranking_batches=0
    for step,(epoch,ids) in enumerate(planned,1):
        for index in ids:exposures[index]+=1
        seen.update(ids);batch=[rows[i] for i in ids]
        if len({r['episode_id'].split(':')[0] for r in batch})<4:raise RuntimeError('G6 variance batch lacks four independent instances')
        optimizer.zero_grad(set_to_none=True)
        loss,parts,c,f=model.training_loss(batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if step==1:
            grad={name:sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None) for name,module in [('online',model.online),('predictor',model.predictor),('readout',model.readout),('value_head',model.value_head),('state_head',model.state_head),('feature_encoder',model.feature_encoder)]}
            if any(v==0 for v in grad.values()):raise RuntimeError('Missing required training gradient')
            if model.target and any(p.grad is not None for p in model.target.parameters()):raise RuntimeError('EMA target received gradients')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();model.update_ema();last=c
        pairs=sum(a['group_id']==b['group_id'] and abs(a['return_target']-b['return_target'])>1e-6 for i,a in enumerate(batch) for b in batch[:i])
        ranking_batches+=bool(pairs)
        row={'step':step,'epoch':epoch,'ranking_pairs':pairs,'loss':float(loss.detach()),'parts':{k:float(v.detach()) for k,v in parts.items()},'seconds':time.monotonic()-started,'distinct_instances':len({r['episode_id'].split(':')[0] for r in batch})}
        batches.append(row)
        atomic(Path(path)/'progress.json',{'pid':__import__('os').getpid(),'last_batch':row,'rows_seen':len(seen),'rows_total':len(rows)})
        if step%10==0:print(json.dumps(row),flush=True)
    critic=fit_critic(model,rows,seed)
    after=parameter_hash(model.online);ema_after=ema_hash()
    if before==after or model.target and ema_before==ema_after:raise RuntimeError('Required encoder/EMA update did not happen')
    summary={'seed':seed,'parent':str(parent) if parent else None,'data_hash':digest(sorted(hashes)),'rows':len(rows),'unique_pairs':len(set(hashes)),
             'epochs':epochs,'full_coverage':min(exposures)>=epochs,'minimum_row_exposures':min(exposures),'encoder_changed':before!=after,'ema_changed':ema_before!=ema_after,'batches':batches,
             'seconds':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'geometry':model.geometry(last),'pilot':True,
             'replay_rows':len(replay_rows or []),'module_gradient_first_step':grad,'submit_positive_weight':model.submit_positive_weight,
             'ranking_batches':ranking_batches,'ranking_coverage':ranking_batches/len(batches),'target_includes_public_goal':True,'value_head_conditioned_on_context_and_predicted_state':True,
             'data_pair_hashes':hashes,'seeded_group_sampler':'four-instances-contrast-pairs-v2','critic_fit':critic}
    model.save_gui(path,summary)
    del optimizer
    return model
