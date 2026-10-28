"""Finite owned-job scheduler with resource and immutable-source checks."""
import fcntl,json,os,signal,socket,subprocess,sys,time,urllib.request
from .settings import source_hashes
from .common import ROOT,PORT,atomic,file_hash

children={};events=[]

def cleanup_owned():
    """Stop only process groups created by this scheduler, including nested trainers."""
    for process in children.values():
        # A failed group leader can leave a nested trainer alive; still signal its group.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+10
    for process in children.values():
        try:process.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:pass
    for process in children.values():
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def interrupted(signum,frame):
    raise KeyboardInterrupt('Scheduler interrupted by signal '+str(signum))
def status(stage,detail=None):
    event={'stage':stage,'detail':detail,'time':time.time()};events.append(event)
    atomic(ROOT/'logs/pipeline_status.json',{'pid':os.getpid(),'stage':stage,'events':events[-100:],'children':{n:{'pid':p.pid,'returncode':p.poll()} for n,p in children.items()},'updated':time.time()})
    print(json.dumps(event),flush=True)

def gpu_free(gpu):
    memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if memory>500:raise RuntimeError(f'GPU {gpu} has another workload')

def launch(name,args,gpu=None):
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',RSI_GUI_ROOT=str(ROOT),RSI_GUI_POLICY_PORT=str(PORT))
    if gpu is not None:gpu_free(gpu);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (ROOT/'logs'/f'{name}.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-m','rsi_anything_gui.'+args[0].removesuffix('.py'),*args[1:]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children[name]=p;status('started',{'name':name,'pid':p.pid,'gpu':gpu});return p

def wait(name,deadline=14400):
    p=children[name];started=time.monotonic()
    while p.poll() is None:
        if time.monotonic()-started>deadline:raise RuntimeError(name+' exceeded deadline; inspect owned live job')
        status('running',{'name':name,'pid':p.pid});time.sleep(20)
    if p.returncode:raise RuntimeError(name+f' exited {p.returncode}; no automatic retry')
    status('completed',{'name':name})

def pool(jobs):
    pending=list(jobs);running={};gpus=json.loads((ROOT/'protocol.json').read_text())['resource_plan']['workers']
    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                name,args=pending.pop(0);running[gpu]=(name,launch(name,args,gpu),time.monotonic())
        for gpu,(name,p,start) in list(running.items()):
            code=p.poll()
            if code is not None:
                if code:raise RuntimeError(f'{name} exited {code}; inspect log')
                del running[gpu];status('completed',{'name':name,'gpu':gpu})
            elif time.monotonic()-start>21600:raise RuntimeError(name+' deadline exceeded')
        status('pool_running',{'active':{gpu:name for gpu,(name,p,t) in running.items()},'pending':len(pending)})
        if running:time.sleep(20)

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    if not source_hashes()==protocol['source_hashes']:raise RuntimeError('Source differs from frozen protocol')
    with socket.socket() as port_check:
        try:port_check.bind(('127.0.0.1',PORT))
        except OSError as error:raise RuntimeError('Policy service port is already in use') from error
    for name in ['browser_contracts.json','world_contracts.json']:
        if not json.loads((ROOT/'results'/name).read_text())['passed']:raise RuntimeError('Preflight not passed: '+name)
    launch('policy_service',['policy_service.py'],protocol['resource_plan']['policy_gpu']);start=time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/ready',timeout=2) as r:
                if json.load(r)['ready']:break
        except Exception:pass
        if children['policy_service'].poll() is not None or time.monotonic()-start>600:raise RuntimeError('Policy service not ready')
        time.sleep(5)
    launch('initial',['experiment.py','initial']);wait('initial',3600)
    # Calibration is collected once, separately from training and development.
    launch('calibration_data',['prepare.py']);wait('calibration_data',3600)
    launch('baseline',['worker.py','baseline'])
    pool([(f'{m}_seed{s}',['worker.py','run','--mode',m,'--seed',str(s)]) for m,s in protocol['world_runs']])
    wait('baseline');launch('rsi',['rsi.py'],protocol['resource_plan']['rsi_world_gpu']);wait('rsi',21600)
    gains=[]
    for seed in [0,1,2]:
        r=json.loads((ROOT/'results'/f'opf_seed{seed}'/'complete.json').read_text())
        gains.append(r['outcomes']['updated']['pure']['test']['success_rate']-r['outcomes']['frozen']['pure']['test']['success_rate'])
    gain=sum(gains)/3;confirm=gain>=protocol['confirmation_threshold']
    atomic(ROOT/'results/confirmation_decision.json',{'run_confirmation':confirm,'mean_pilot_gain':gain,'threshold':protocol['confirmation_threshold'],'pure_world_comparison':True})
    if confirm:pool([(f'confirm_seed{s}',['worker.py','confirm','--mode','opf','--seed',str(s)]) for s in [0,1,2]])
    launch('audit',['audit.py']);wait('audit',1800)
    launch('aggregate',['aggregate.py']);wait('aggregate',600)
    service=children['policy_service'];service.terminate();service.wait(timeout=60)
    status('all_stages_complete',{'confirmation_run':confirm,'owned_policy_service_stopped':True})

def entrypoint():
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupted)
    with (ROOT/'logs/controller.lease').open('w') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);lease.write(str(os.getpid()));lease.flush()
        try:main()
        except Exception as error:
            import traceback
            status('failed',{'error':repr(error),'traceback':traceback.format_exc()});raise
        finally:cleanup_owned()

if __name__=='__main__':entrypoint()
