# Public release: removed the server import path and unused non-GUI methods.
"""True execution JEPA: trainable online LoRA encoder, matching EMA encoder, operational OPF synthesis."""
import copy,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from jepa_anything_core import OrthogonalFactorProjection,UnconstrainedMultiHeadPredictor,MLPPredictor,jepa_anything_objective,encoder_variance_loss,factor_activity_loss,audit_opf_geometry
D,K,R=256,4,64
class Overlength(ValueError):pass
class DomainEncoder(nn.Module):
 def __init__(self,base_path,max_length=16384):
  super().__init__()
  from transformers import AutoModel,AutoTokenizer
  from peft import LoraConfig,get_peft_model,TaskType
  self.tokenizer=AutoTokenizer.from_pretrained(base_path,local_files_only=True)
  self.tokenizer.pad_token=self.tokenizer.eos_token
  self.tokenizer.padding_side='right';self.max_length=max_length
  base=AutoModel.from_pretrained(base_path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
  config=LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'])
  self.backbone=get_peft_model(base,config)
  self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.backbone.enable_input_require_grads()
  self.backbone.config.use_cache=False
  self.projection=nn.Sequential(nn.Linear(base.config.hidden_size,D),nn.LayerNorm(D))
 def forward(self,texts):
  batch=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
  if batch['input_ids'].shape[-1]>self.max_length:raise Overlength('encoder input exceeds explicit limit')
  device=next(self.backbone.parameters()).device;batch={k:v.to(device) for k,v in batch.items()}
  outputs=self.backbone(**batch,return_dict=True,use_cache=False).last_hidden_state
  mask=batch['attention_mask'].unsqueeze(-1)
  pooled=(outputs.float()*mask).sum(1)/mask.sum(1).clamp_min(1)
  return self.projection(pooled)
class ExecutionWorld(nn.Module):
 def __init__(self,encoder,mode='opf',ema=.996,include_target=True):
  super().__init__();self.mode=mode;self.ema_decay=ema;self.online=encoder
  self.target=copy.deepcopy(encoder).requires_grad_(False).eval() if mode!='ce' and include_target else None
  if mode=='opf':
   self.opf=OrthogonalFactorProjection(D,K,R,learnable=True,orthogonality_mode='soft_gram')
   self.predictor=UnconstrainedMultiHeadPredictor(D,D,K,R,512)
  elif mode=='multi':
   self.opf=None;self.predictor=UnconstrainedMultiHeadPredictor(D,D,K,R,640)
  elif mode in ('standard','ce'):
   self.opf=None;self.predictor=MLPPredictor(D,D,640)
  else:raise ValueError(mode)
  self.log_temperature=nn.Parameter(torch.zeros(()))
  self.readout=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,64),nn.GELU(),nn.Linear(64,6))
 def train(self,mode=True):
  super().train(mode)
  if self.target is not None:self.target.eval()
  return self
 def predict_latent(self,context):
  p=self.predictor(context)
  if self.mode=='opf':return self.opf.compose(p),p
  if self.mode=='multi':return p.flatten(-2),p
  return p,p.reshape(-1,K,R)
 def encode(self,texts,target=False,microbatch=1):
  encoder=self.target if target else self.online
  if encoder is None:raise RuntimeError('CE has no target')
  # Concatenate representation graphs before variance computation: actual diverse batch, not accumulation.
  rows=[]
  for i in range(0,len(texts),microbatch):rows.append(encoder(texts[i:i+microbatch]))
  return torch.cat(rows,dim=0)
 @torch.no_grad()
 def update_ema(self):
  if self.target is None:return
  targets=dict(self.target.named_parameters())
  for name,p in self.online.named_parameters():
   if p.requires_grad:targets[name].mul_(self.ema_decay).add_(p,alpha=1-self.ema_decay)
 def geometry(self,context):
  if self.opf is None:return {}
  z=context.detach().float();f=self.opf(z)
  report=audit_opf_geometry(self.opf,z)
  return report.to_dict()

