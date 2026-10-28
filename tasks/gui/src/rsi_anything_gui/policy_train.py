import gc
import random
import time
from pathlib import Path
import torch
from .common import ROOT,BASE_POLICY,atomic,digest,file_hash
from .policy_service import prompt

def positive_rows(rows):
    return list({r['pair_hash']:r for r in rows if r.get('episode_success')==1 and r['utility']>0 and (r['action']['kind']=='click' or r['success'])}.values())

def stratified(rows,count,rng):
    terminal=[r for r in rows if r['action']['kind']=='submit'];clicks=[r for r in rows if r['action']['kind']=='click']
    rng.shuffle(terminal);rng.shuffle(clicks)
    selected=terminal[:min(64,count,len(terminal))]
    rest=clicks+terminal[len(selected):];rng.shuffle(rest)
    selected+=rest[:count-len(selected)]
    if len(selected)<count and selected:selected+=rng.choices(selected,k=count-len(selected))
    return selected

def train_policy(new,replay,path,seed,parent=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model
    input_new_hash=digest([r['pair_hash'] for r in new]);input_replay_hash=digest([r['pair_hash'] for r in replay])
    new=positive_rows(new);replay=positive_rows(replay)
    if not new:
        atomic(Path(path)/'not_updated.json',{'reason':'No positive actions from strictly successful actual episodes','parent':str(parent) if parent else None,
                                            'input_new_hash':input_new_hash,'input_replay_hash':input_replay_hash})
        return parent
    rng=random.Random(seed);new=stratified(new,min(256,len(new)),rng)
    old=stratified(replay,len(new),rng) if replay else []
    if len(old)!=len(new):raise RuntimeError('Successful replay pool cannot satisfy 1:1 policy update')
    rows=new+old;rng.shuffle(rows)
    torch.set_num_threads(4);torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(BASE_POLICY,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(BASE_POLICY,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
    if parent:model=PeftModel.from_pretrained(base,parent,is_trainable=True)
    else:model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj']))
    model.cuda().train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    def adapter_hash():
        import hashlib
        h=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()
    before_hash=adapter_hash();torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    examples=[]
    for r in rows:
        text,actions=prompt(tokenizer,r['before'])
        action_id=actions.index(r['action'])
        ids=tokenizer.encode(text,add_special_tokens=False)+tokenizer.encode(str(action_id),add_special_tokens=False)
        if ids!=tokenizer.encode(text+str(action_id),add_special_tokens=False):raise RuntimeError('Training/inference token boundary mismatch')
        if len(ids)>1024:raise RuntimeError('Policy example exceeds explicit token limit')
        examples.append(ids)
    started=time.monotonic();batches=[]
    for step,i in enumerate(range(0,len(examples),4),1):
        batch=examples[i:i+4];length=max(map(len,batch));b=len(batch)
        inputs=torch.full((b,length),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
        attention=torch.zeros_like(inputs);labels=torch.full_like(inputs,-100)
        for j,ids in enumerate(batch):
            inputs[j,:len(ids)]=torch.tensor(ids,device='cuda');attention[j,:len(ids)]=1;labels[j,len(ids)-1]=ids[-1]
        optimizer.zero_grad(set_to_none=True)
        loss=model(input_ids=inputs,attention_mask=attention,labels=labels,use_cache=False).loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite policy SFT loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step()
        row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started};batches.append(row)
        atomic(Path(path)/'progress.json',{'pid':__import__('os').getpid(),'last_batch':row,'examples':len(examples)})
    after_hash=adapter_hash()
    if before_hash==after_hash:raise RuntimeError('Policy weights did not update')
    model.save_pretrained(path)
    atomic(Path(path)/'training.json',{'seed':seed,'parent':str(parent) if parent else None,'new_rows':len(new),'replay_rows':len(old),'examples':len(examples),
                                    'data_hash':digest([r['pair_hash'] for r in rows]),'only_actual_positive_feedback':True,'epochs':1,'batches':batches,
                                    'seconds':time.monotonic()-started,'protocol':'successful-episode-terminal-stratified-v2',
                                    'new_pair_hashes':[r['pair_hash'] for r in new],'replay_pair_hashes':[r['pair_hash'] for r in old],
                                    'example_pair_hashes':[r['pair_hash'] for r in rows],'actual_weight_change':before_hash!=after_hash,
                                    'new_submit_rows':sum(r['action']['kind']=='submit' for r in new),'replay_submit_rows':sum(r['action']['kind']=='submit' for r in old),
                                    'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'strict_successful_episodes_only':True})
    metadata=__import__('json').loads((Path(path)/'training.json').read_text())
    metadata.update(input_new_hash=input_new_hash,input_replay_hash=input_replay_hash,files={p.name:file_hash(p) for p in Path(path).glob('*') if p.name in ['adapter_model.safetensors','adapter_config.json']})
    atomic(Path(path)/'training.json',metadata)
    del optimizer,model,base;gc.collect();torch.cuda.empty_cache()
    return Path(path)

if __name__=='__main__':
    import argparse
    from .common import read_jsonl
    p=argparse.ArgumentParser();p.add_argument('--new',required=True);p.add_argument('--replay',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--parent');a=p.parse_args()
    result=train_policy(read_jsonl(a.new),read_jsonl(a.replay),a.output,a.seed,a.parent)
    atomic(Path(a.output)/'result.json',{'adapter':str(result) if result else None})
