"""Rebuild the 14 archived LF operating points and their training runs."""
from pathlib import Path
import argparse, json, os
R=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(R/'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir',type=Path,default=R/'figures')
args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
N=json.loads((R.parents[1]/'analysis/numbers.json').read_text(encoding='utf8'))
plt.rcParams.update({'font.family':['Times New Roman','DejaVu Serif'],'font.size':8.6,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.48,2.8))
arms=[('full_ppo','Full-weight PPO','#22699a','o','1e-5'),
      ('full_grpo','Full-weight GRPO','#6098bb','s','1e-5'),
      ('res_ppo','Residual PPO','#b64b22','^','3e-3'),
      ('res_grpo','Residual GRPO','#d89125','D','1e-3')]
assert sum(len(v) for v in N['lrcurve']['LF'].values())==14
for key,label,color,marker,selected in arms:
    rows=N['lrcurve']['LF'][key]
    ax.plot([100*r['nom_loss'] for r in rows],[100*r['fault'] for r in rows],color=color,marker=marker,ms=4,lw=.85,label=label)
    for r in rows:
        ax.scatter([100*(N['frozen']['nom']-v) for v in r['nom_runs']],[100*v for v in r['fault_runs']],color=color,marker=marker,s=9,alpha=.28,edgecolors='none')
        if r['lr']==selected:
            ax.scatter(100*r['nom_loss'],100*r['fault'],s=84,marker='*',color=color,edgecolor='black',lw=.6,zorder=6)
ax.axhline(100*N['frozen']['LF'],color='.5',ls=':',lw=.8)
ax.text(1,100*N['frozen']['LF']+1,'Frozen',color='.45',fontsize=7.6)
ax.set_xlabel('Nominal loss relative to frozen planner (pp)')
ax.set_ylabel('Little-finger-lock success (%)')
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower right',fontsize=7.4,framealpha=.95)
fig.tight_layout(pad=.6)
for ext in ('png','pdf'):fig.savefig(args.out_dir/('fig1_tradeoff.'+ext),dpi=600)
print('Rebuilt 14 candidate means and all associated independent-run points.')
