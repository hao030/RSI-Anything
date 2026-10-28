import csv,json,math,time
import numpy as np
from scipy.stats import binomtest
from .common import ROOT,atomic,read_jsonl
from .experiment import summary

from .statistics import paired

def main():
    output=ROOT/'outputs';protocol=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((output/'completion_audit.json').read_text())
    if not audit['passed']:raise RuntimeError('Audit did not pass')
    arms=[];costs=[];runs=[]
    def add(name,path):
        rows=read_jsonl(path);s=summary(rows);s['arm']=name;arms.append(s)
        costs.append({'arm':name,'observed_wall_seconds':s['mean_seconds'],'effective_seconds':s['effective_seconds'],
                      'policy_uncached_equivalent_seconds':sum(r['policy_nominal_seconds'] for r in rows)/len(rows),
                      'world_seconds':sum(r['world_seconds'] for r in rows)/len(rows),'browser_seconds':sum(r['browser_seconds'] for r in rows)/len(rows),
                      'cached_requests':sum(r['cached_policy_requests'] for r in rows)})
    for mode,seed in protocol['world_runs']:
        name=f'{mode}_seed{seed}';folder=ROOT/'results'/name;run=json.loads((folder/'complete.json').read_text());runs.append(run)
        for version in ['frozen','updated']:
            for kind in ['pure','selected']:add(name+'_'+version+'_'+kind,folder/f'{version}_{kind}_test.jsonl')
    for method in ['policy','rule']:add('baseline_'+method,ROOT/f'results/baseline_{method}_test.jsonl')
    primary=paired([ROOT/f'results/opf_seed{s}/updated_pure_test.jsonl' for s in [0,1,2]],[ROOT/f'results/opf_seed{s}/frozen_pure_test.jsonl' for s in [0,1,2]])
    selected=paired([ROOT/f'results/opf_seed{s}/updated_selected_test.jsonl' for s in [0,1,2]],[ROOT/f'results/opf_seed{s}/frozen_selected_test.jsonl' for s in [0,1,2]])
    secondary={}
    for name,a,b in [('pure_updated_vs_policy',ROOT/'results/opf_seed0/updated_pure_test.jsonl',ROOT/'results/baseline_policy_test.jsonl'),
                     ('selected_updated_vs_policy',ROOT/'results/opf_seed0/updated_selected_test.jsonl',ROOT/'results/baseline_policy_test.jsonl'),
                     ('pure_opf_vs_ce',ROOT/'results/opf_seed0/updated_pure_test.jsonl',ROOT/'results/ce_seed0/updated_pure_test.jsonl'),
                     ('pure_opf_vs_standard',ROOT/'results/opf_seed0/updated_pure_test.jsonl',ROOT/'results/standard_seed0/updated_pure_test.jsonl'),
                     ('initial_opf_vs_ce_shared_data',ROOT/'results/opf_seed0/frozen_pure_test.jsonl',ROOT/'results/ce_seed0/frozen_pure_test.jsonl'),
                     ('initial_opf_vs_standard_shared_data',ROOT/'results/opf_seed0/frozen_pure_test.jsonl',ROOT/'results/standard_seed0/frozen_pure_test.jsonl')]:secondary[name]=paired([a],[b])
    previous=0.
    for i,(p,name) in enumerate(sorted((x['per_run_tests'][0]['exact_mcnemar_p'],n) for n,x in secondary.items())):
        previous=max(previous,min(1.,p*(len(secondary)-i)));secondary[name]['holm_adjusted_p']=previous
    rsi=[]
    for generation in [1,2]:
        folder=ROOT/'results'/f'rsi_g{generation}';data=json.loads((folder/'complete.json').read_text())
        for arm in ['old_old','old_new','new_old','new_new']:
            for kind in ['pure','selected']:add(f'rsi_g{generation}_{arm}_{kind}',folder/f'{arm}_{kind}_test.jsonl')
        for arm in ['old_policy','new_policy']:add(f'rsi_g{generation}_{arm}',folder/f'{arm}_test.jsonl')
        data['paired_policy_only']=paired([folder/'new_policy_test.jsonl'],[folder/'old_policy_test.jsonl'])
        data['paired_joint_selected']=paired([folder/'new_new_selected_test.jsonl'],[folder/'old_old_selected_test.jsonl'])
        rsi.append(data)
    decision=json.loads((ROOT/'results/confirmation_decision.json').read_text());confirmation=None
    if decision['run_confirmation']:confirmation=paired([ROOT/f'results/opf_seed{s}/updated_confirmation.jsonl' for s in [0,1,2]],[ROOT/f'results/opf_seed{s}/frozen_confirmation.jsonl' for s in [0,1,2]])
    results={'protocol':protocol,'primary_pure':primary,'selected_update_effect':selected,'secondary':secondary,'arms':arms,'runs':runs,'rsi':rsi,'confirmation_decision':decision,'confirmation':confirmation,'audit':audit,
             'interpretation':'Expert-assisted counterfactual training and architecture/control repairs changed several factors; old-vs-v2 is not a single-factor causal comparison. Pure-world endpoints expose whether fallback masks a weak world.'}
    atomic(output/'results.json',results);atomic(output/'costs.json',{'arms':costs,'shared_policy_cache':True,'gate_uses_uncached_equivalent_cost':True,'cost_equivalent_not_independent_cold_retest':True})
    with (output/'arm_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,list(arms[0]));writer.writeheader();writer.writerows(arms)
    training=[]
    for p in (ROOT/'checkpoints').glob('*/**/training.json'):
        if any(part.startswith('probe_excluded') for part in p.parts):continue
        d=json.loads(p.read_text());critic=d.get('critic_fit',{});training.append({'checkpoint':str(p.parent.relative_to(ROOT)),'rows':d.get('rows',d.get('examples')),'representation_or_policy_epochs':d['epochs'],'critic_epochs':critic.get('configuration',{}).get('epochs',0),'total_training_seconds':d['seconds'],'critic_seconds':critic.get('seconds',0.),'peak_gpu_gib':d['peak_gpu_bytes']/1024**3,'ranking_coverage':d.get('ranking_coverage')})
    with (output/'training_costs.csv').open('w') as f:
        writer=csv.DictWriter(f,list(training[0]));writer.writeheader();writer.writerows(training)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5));names=[f'{m} seed{s}' for m,s in protocol['world_runs']];x=np.arange(len(names));lookup={r['arm']:r['success_rate'] for r in arms}
    for i,(version,kind,color) in enumerate([('frozen','pure','#4477AA'),('updated','pure','#228833'),('frozen','selected','#CCBB44'),('updated','selected','#AA3377')]):
        ax.bar(x+(i-1.5)*.18,[lookup[f'{m}_seed{s}_{version}_{kind}'] for m,s in protocol['world_runs']],width=.18,label=version+' '+kind,color=color)
    ax.axhline(lookup['baseline_policy'],ls='--',color='black',label='policy baseline');ax.set_xticks(x,names);ax.set_ylim(0,1.05);ax.set_ylabel('Strict success');ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig(output/'success_rates.png',dpi=180);plt.close(fig)
    lines=['# GUI 完整修复与重跑结果','',f"独立审计：{'通过' if audit['passed'] else '失败'}。",'',
           '新实例、新冻结协议。真实 Chromium、DOM 文本、120 秒、最多8步。训练使用实际执行的候选分支和仅训练阶段的规则演示/后续完成反馈；所有世界模型对照共享这些数据。',
           '修复了公开目标缺失、价值头上下文、候选排序监督、提交校准、基础策略回退、失败回合正反馈训练和事务恢复。纯世界模型与带回退的控制器分别评价。','',
           '世界模型先联合训练3轮，再固定训练集的模型预测表示，拟合完成读出头100轮。状态预测使用可学习逐槽位动力学头；该头能读取当前勾选状态与候选点击位置。因此状态准确率提升本身不能证明JEPA贡献。读出头缓存不含真实下一状态标签，所有训练设置对各世界模型一致。','',
           f"纯 OPF 更新效应：{primary['effect']*100:.1f} 个百分点，配对区间 {primary['task_cluster_bootstrap_95_ci']}；训练种子作为三个固定模型。",'',
           '| 方法 | n | 成功率 | 平均步数 | 重复点击 |','|---|---:|---:|---:|---:|']
    for a in arms:lines.append(f"| {a['arm']} | {a['n']} | {a['success_rate']:.1%} | {a['mean_steps']:.2f} | {a['repeated_clicks']} |")
    lines+=['','## 控制器回退与模型诊断','']
    for run in runs:
        for version in ['frozen','updated']:
            out=run['outcomes'][version];lines.append(f"- {run['mode']} seed{run['seed']} {version}：alpha={out['controller']['controller']['alpha']}，回退={out['controller']['fallback']}；诊断={json.dumps(out['diagnostic'],ensure_ascii=False)}")
    lines+=['','## RSI 与解释范围','']
    for g in rsi:lines.append(f"- 第{g['versions']['generation']}代晋升={g['versions']['promoted']}；策略单独变化={g['paired_policy_only']['effect']*100:.1f}个百分点；联合选定控制器变化={g['paired_joint_selected']['effect']*100:.1f}个百分点。")
    lines+=['',f"630实例确认：{'已执行' if decision['run_confirmation'] else '未达到预设条件，未执行'}。",'',
            '旧版与新版同时改变了训练数据覆盖、监督、结构和控制方式，其差异不能归因于某一项修复。JEPA贡献应看同数据的OPF、标准JEPA和监督对照。',
            '采用基础策略回退时成功率不能直接记为世界模型收益。无差异样本得到退化bootstrap区间不能证明总体等效。RSI只有一个种子、两个候选代次；仍为一个DOM任务的实验。']
    (output/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
