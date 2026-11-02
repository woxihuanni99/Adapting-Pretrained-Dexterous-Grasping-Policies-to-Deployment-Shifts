from pathlib import Path
import json,os
R=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N=json.loads((R.parents[2]/'analysis/numbers.json').read_text(encoding='utf8'))
S=json.loads((R/'numbers.json').read_text(encoding='utf8'))
F=R.parents[2]/'manuscript/cscwd2027_v5/figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.48,3.05))
arms=[('full_ppo','Full PPO','#22699a','o','1e-5'),('full_grpo','Full GRPO','#6098bb','s','1e-5'),
      ('res_ppo','Residual PPO','#b64b22','^','3e-3'),('res_grpo','Residual GRPO','#d89125','D','1e-3')]
ax.axvspan(S['interpolation']['estimate']['lower'],S['interpolation']['estimate']['upper'],color='.9',zorder=0)
inset=ax.inset_axes([.63,.18,.34,.32])
for key,label,color,marker,rate in arms:
    rows=N['lrcurve']['LF'][key]
    x=[r['nom_loss']*100 for r in rows];y=[r['fault']*100 for r in rows]
    ax.plot(x,y,color=color,marker=marker,ms=3.9,lw=.9,label=label)
    inset.plot(x,y,color=color,marker=marker,ms=2,lw=.65)
    for r in rows:
        ax.scatter([100*(N['frozen']['nom']-v) for v in r['nom_runs']],[100*v for v in r['fault_runs']],color=color,marker=marker,s=9,alpha=.3,edgecolors='none')
        if r['lr']==rate:ax.scatter(r['nom_loss']*100,r['fault']*100,s=78,marker='*',color=color,edgecolor='black',lw=.65,zorder=5)
ax.axhline(N['frozen']['LF']*100,color='.5',ls=':',lw=.7)
ax.text(.2,18.3,'Frozen',fontsize=7.5,color='.4')
ax.set_xlim(0,12);ax.set_ylim(14,80)
ax.set_xlabel('Nominal loss relative to frozen planner (pp)',fontsize=8.8)
ax.set_ylabel('Reference-LF success (%)',fontsize=8.8)
ax.spines[['right','top']].set_visible(False)
ax.legend(loc='upper left',fontsize=7.6,ncol=2,frameon=False,handlelength=1.2,columnspacing=.7)
inset.set_xlim(0,45);inset.set_ylim(14,80);inset.set_xticks([0,20,40]);inset.set_yticks([20,60]);inset.tick_params(labelsize=6.8,pad=1,length=2)
inset.set_title('Full range',fontsize=7.2,pad=2)
fig.subplots_adjust(left=.16,right=.98,bottom=.17,top=.98)
for ext in ('png','pdf'):fig.savefig(F/('fig1_tradeoff.'+ext),dpi=600)
print('Figure rebuilt from the 14 original candidate points.')
