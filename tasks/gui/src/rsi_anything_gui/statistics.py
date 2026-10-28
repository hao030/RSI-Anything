"""Task-paired statistics from the validated original aggregator."""
import numpy as np
from scipy.stats import binomtest
from .common import read_jsonl

def paired(paths_a,paths_b):
    a=[{r['seed']:r for r in read_jsonl(p)} for p in paths_a];b=[{r['seed']:r for r in read_jsonl(p)} for p in paths_b]
    seeds=sorted(a[0])
    if any(set(x)!=set(seeds) for x in a+b):raise RuntimeError('Paired instances incomplete')
    if any(len({x[s]['initial_hash'] for x in a+b})!=1 for s in seeds):raise RuntimeError('Initial browser states not paired')
    diff=np.array([[x[s]['success']-y[s]['success'] for s in seeds] for x,y in zip(a,b)],float)
    mean=diff.mean(0);rng=np.random.default_rng(20261006);ids=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    tests=[]
    for x,y in zip(a,b):
        wins=sum(x[s]['success']>y[s]['success'] for s in seeds);losses=sum(x[s]['success']<y[s]['success'] for s in seeds)
        tests.append({'wins':wins,'losses':losses,'exact_mcnemar_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.})
    return {'instances':len(seeds),'fixed_training_runs':len(a),'effect':float(mean.mean()),'task_cluster_bootstrap_95_ci':np.quantile(mean[ids].mean(1),[.025,.975]).tolist(),
            'seed_effects':diff.mean(1).tolist(),'per_run_tests':tests,'all_zero_difference_bootstrap_is_not_equivalence_proof':bool(np.all(diff==0))}

