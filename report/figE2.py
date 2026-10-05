# рисунок к сводке о дифференциальной модели: числа из отчётов wm_stein7 и wm_stein8
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
d=np.array([0,1,3.0]); fig,ax=plt.subplots(1,3,figsize=(7.5,2.9))
P=[('a   near axis / near diagonal\n     baseline sessions',[1.15,1.37,1.69],[0.94,1.25,1.51],[1.39,1.50,1.88],[1.23,1.32,1.44],[1.40,1.60,1.78],[1.97,2.45,2.81],[1.93,2.48,2.90]),
   ('b   middle / near diagonal\n     baseline sessions',[1.00,1.19,1.26],[0.83,1.11,1.13],[1.20,1.28,1.40],[1.00,1.00,1.02],[1.10,1.14,1.16],[1.01,1.04,1.05],[1.14,1.24,1.33]),
   ('c   near axis / near diagonal\n     post sessions',[0.91,1.22,1.25],[0.74,1.08,1.02],[1.09,1.37,1.54],[1.16,1.35,1.43],[1.34,1.59,1.81],[1.70,2.52,2.81],[1.71,2.44,2.67])]
for a,(ttl,y,lo,hi,ds,dm,os_,om) in zip(ax,P):
    y=np.array(y); a.errorbar(d,y,yerr=[y-np.array(lo),np.array(hi)-y],fmt='-o',color='k',capsize=2.5,lw=1.3,ms=4,label='data',zorder=5)
    a.plot(d,ds,'--s',color='#c0392b',mfc='white',ms=4,lw=1,label='D, sawtooth'); a.plot(d,dm,'-s',color='#c0392b',ms=4,lw=1,label='D, smooth')
    a.plot(d,os_,'--^',color='#2471a3',mfc='white',ms=4,lw=1,label='R, sawtooth'); a.plot(d,om,'-^',color='#2471a3',ms=4,lw=1,label='R, smooth')
    a.axhline(1,color='0.7',lw=0.6); a.set_xlabel('delay, s'); a.set_title(ttl,loc='left',fontsize=7.6); a.set_xticks([0,1,3])
ax[0].set_ylim(0.8,3.2); ax[2].set_ylim(0.6,3.1)
h,l=ax[0].get_legend_handles_labels(); o=[4,0,1,2,3]; fig.legend([h[i] for i in o],[l[i] for i in o],frameon=False,fontsize=7,loc='lower center',ncol=5)
fig.tight_layout(w_pad=1.2,rect=(0,0.08,1,1)); fig.savefig('figE2.pdf'); print('ok')
