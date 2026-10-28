import json,time,hashlib,argparse
from pathlib import Path
from .common import ROOT,atomic,digest,file_hash
from .settings import SETTINGS, MINIWoB_COMMIT, source_hashes, runtime_identity

def freeze():
    cohorts={'probe':list(range(190000,190004)),'initial':list(range(200000,200100)),
             'update':list(range(210000,210100)),'terminal_calibration':list(range(220000,220050)),
             'dev':list(range(230000,230050)),'diagnostic':list(range(240000,240050)),
             'test':list(range(250000,250100)),'confirmation':list(range(260000,260630)),
             'rsi_test':list(range(270000,270100)),'rsi_update1':list(range(280000,280100)),
             'rsi_update2':list(range(290000,290100))}
    flat=[s for values in cohorts.values() for s in values]
    assert len(flat)==len(set(flat))
    p={'version':'rsi-anything-public-1.0','original_protocol':'gui-complete-repair-v2','created':time.time(),'splits':cohorts,'task':'official-click-checkboxes',
       'miniwob_commit':MINIWoB_COMMIT,'observation':'visible-DOM-text','timeout_seconds':120,'max_steps':8,
       'candidate_protocol':'top4-with-submit-included-same-for-all-arms','world_runs':[['opf',0],['opf',1],['opf',2],['ce',0],['standard',0]],
       'initial_epochs':3,'update_epochs':3,'probe_epochs':20,'replay_ratio':1.,'discount':.9,
       'initial_training':'100 instances, reset/random/correct/near-correct training states, all legal real branches, expert actual continuations',
       'online_training':'100 on-policy instances followed by offline all-legal-action browser branches and expert continuations; no branch feedback while acting',
       'world_target':'actual next visible state, public task instruction, submitted, terminated, remaining budget',
       'value_head':'current goal/state/action context plus predicted next-state latent and visible non-oracle features',
       'encoder':'compact public state text, causal last-token pooling; no target-bit matching features',
       'state_decoder':'learned shared per-slot MLP of observed current bit and candidate click indicator plus predicted-latent residual; no coded state flip',
       'optimizer_rates':{'lora':5e-6,'representation':1e-4,'opf':1e-5,'readouts':1e-3,'learned_slot_dynamics':1e-2},
       'critic_fit':{'epochs':100,'batch_size':128,'lr':1e-3},
       'critic_fit_protocol':'freeze representations after joint training; completion heads trained on cached model predicted states from the same training rows; balanced positive/negative Submit BCE, utility/return MSE and same-state ranking',
       'value_target':'discounted actual expert-continuation terminal outcome + 0.1 immediate progress; terminal +1/-1',
       'terminal_calibration_grid':{'temperature':[.5,1.,2.,4.],'bias':[-2.,-1.,0.,1.,2.]},
       'terminal_calibration_metric':'Brier on separate 50-instance real candidate branches; not development or test',
       'hybrid_alpha_grid':[.25,1.],'controller_enable_gain':.04,'controller_enable_cost_ratio':4.,
       'promotion_gain':.04,'promotion_cost_ratio':1.2,'cost_gate':'policy uncached-equivalent seconds + actual world + browser seconds; observed wall reported separately',
       'world_selection':'submit uses separately calibrated submit-success probability; non-submit uses predicted discounted return; hybrid retains policy prior',
       'pure_world_always_reported':True,'fallback_to_policy_if_no_controller_passes':True,
       'main_comparison':'pure updated OPF minus pure frozen OPF on 100 paired instances, 3 fixed trained models',
       'confirmation_threshold':.05,'confirmation_n':630,'confirmation_target':'pure W1 vs pure W0, all3 OPF seeds',
       'rsi_generations':2,'rsi_training_seeds':1,'rsi_crossed_tests':['old_old','old_new','new_old','new_new'],
       'rsi_policy_only_tests':True,'freeze_both_rsi_generations_before_rsi_test':True,
       'policy_update':'deduplicated positive actions only from strictly successful actual episodes; up to256 new+256 replay, terminal stratification; no positives => documented no update',
       'source_hashes':source_hashes(),'base_manifest_hash':digest(json.loads((ROOT/'base_manifest.json').read_text())),
       'previous_test_never_reused':True,'expert_assistance_train_only':True,'not_official_default_timeout_score':True,
       'resource_plan':SETTINGS['resource_plan'],'runtime_identity':runtime_identity(),
       'limits':['one DOM task template','controls one seed','RSI two proposals from incumbent, one training seed','conditional confirmation pilot-sized, not power guarantee']}
    path=ROOT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if {k:v for k,v in existing.items() if k!='created'}!={k:v for k,v in p.items() if k!='created'}:
            raise RuntimeError('Frozen source, inputs or configuration changed; use a new run directory')
        return existing
    atomic(path,p);return p

def bases():
    from .settings import BASE_WORLD, BASE_POLICY
    manifest={}
    for folder in [BASE_WORLD, BASE_POLICY]:
        if not (folder/'config.json').is_file():
            raise FileNotFoundError('Missing local base model: '+str(folder))
        manifest[folder.name]={str(p.relative_to(folder)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
                               for p in sorted(folder.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(folder).parts}
    path=ROOT/'base_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Base inputs changed; use a new run directory')
    atomic(path,manifest)
    print(json.dumps({'base_models':list(manifest),'files':sum(map(len,manifest.values()))}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['bases','freeze']);a=p.parse_args()
    bases() if a.phase=='bases' else freeze()
