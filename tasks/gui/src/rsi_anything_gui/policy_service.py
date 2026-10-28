"""Shared GPU policy with constrained legal single-token output and adapter-aware cache."""
import argparse
import contextlib
import http.server
import json
import sqlite3
import threading
import time
from pathlib import Path
import torch
from .common import ROOT,BASE_POLICY,PORT,atomic,digest,legal_actions,file_hash

SYSTEM='You control a browser checkbox form. Follow the instruction using the current visible checked states. Click needed boxes, avoid undoing correct choices, and submit only when the form is correct. Output only the numeric ID of the next action.'

def prompt(tokenizer,obs):
    actions=legal_actions(obs)
    body=json.dumps({'instruction':obs['instruction'],'visible_elements':obs['elements'],'remaining_actions':obs['remaining_actions'],'legal_actions':[{'id':i,**a} for i,a in enumerate(actions)]},ensure_ascii=False)
    text=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True)
    return text+'Action ID: ',actions

class PolicyEngine:
    def __init__(self):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(BASE_POLICY,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(BASE_POLICY,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').cuda().eval()
        self.peft=False
        self.adapters={}
        self.base_identity=digest(json.loads((ROOT/'base_manifest.json').read_text()))
        self.adapter_identities={}
        self.lock=threading.Lock()
        self.db=sqlite3.connect(ROOT/'data/policy_cache.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS ranks (key TEXT PRIMARY KEY, result TEXT)')
        self.ids=[self.tokenizer.encode(str(i),add_special_tokens=False) for i in range(7)]
        if any(len(ids)!=1 for ids in self.ids):raise RuntimeError('Legal action IDs must be single tokens')

    def rank(self,obs,adapter=None):
        identity='base'
        if adapter:
            path=Path(adapter).resolve()
            if ROOT/'checkpoints' not in path.parents:raise ValueError('Adapter path outside GUI experiment')
            if str(path) not in self.adapter_identities:
                self.adapter_identities[str(path)]=digest([file_hash(path/'training.json'),file_hash(path/'adapter_model.safetensors'),file_hash(path/'adapter_config.json')])
            identity=self.adapter_identities[str(path)]
        key=digest([self.base_identity,identity,obs['instruction'],obs['elements'],obs['remaining_actions'],SYSTEM,'include-submit-v2'])
        with self.lock:
            row=self.db.execute('SELECT result FROM ranks WHERE key=?',(key,)).fetchone()
            if row:
                result=json.loads(row[0]);result['cached']=True;return result
            scope=contextlib.nullcontext()
            if adapter:
                path=Path(adapter).resolve()
                if ROOT/'checkpoints' not in path.parents:raise ValueError('Adapter path outside GUI experiment')
                name='a'+identity[:20]
                if name not in self.adapters:
                    if not self.peft:
                        from peft import PeftModel
                        self.model=PeftModel.from_pretrained(self.model,path,adapter_name=name,is_trainable=False).eval();self.peft=True
                    else:self.model.load_adapter(path,adapter_name=name,is_trainable=False)
                    self.adapters[name]=str(path)
                self.model.set_adapter(name)
            elif self.peft:scope=self.model.disable_adapter()
            text,actions=prompt(self.tokenizer,obs)
            tokens=self.tokenizer(text,return_tensors='pt',truncation=False,add_special_tokens=False).to('cuda')
            start=time.monotonic()
            with scope,torch.inference_mode():
                logits=self.model(**tokens,use_cache=False).logits[0,-1,[x[0] for x in self.ids[:len(actions)]]].float()
                probs=torch.softmax(logits,dim=0)
                order=torch.argsort(logits,descending=True).tolist()
            candidates=order[:4]
            submit=len(actions)-1
            if submit not in candidates:candidates=candidates[:3]+[submit]
            result={'actions':[actions[i] for i in candidates],'candidate_ids':candidates,'all_action_ids':order,'probabilities':probs.tolist(),'seconds':time.monotonic()-start,'input_tokens':tokens.input_ids.numel(),'cached':False,'policy_identity':identity,'base_identity':self.base_identity,'candidate_protocol':'include-submit-v2'}
            self.db.execute('INSERT INTO ranks VALUES (?,?)',(key,json.dumps(result)));self.db.commit()
            return result

def serve(port):
    engine=PolicyEngine()
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
        def do_POST(self):
            try:
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result=engine.rank(data['observation'],data.get('adapter'))
                body=json.dumps(result).encode();self.send_response(200)
            except Exception as e:
                body=json.dumps({'error':repr(e)}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    atomic(ROOT/'logs/policy_ready.json',{'pid':__import__('os').getpid(),'port':port,'started':time.time(),'model':str(BASE_POLICY),'protocol':'constrained-legal-action-likelihood-v1'})
    http.server.ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=PORT);serve(p.parse_args().port)
