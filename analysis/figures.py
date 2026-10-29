"""Render the three result figures."""


from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import re
import fitz
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve


SMALL_SWAY='#006D77'


LARGE_SWAY='#C65D4E'


HIGHLIGHT='#EAF3F3'


CONDITIONS=[('trunk_sway_small','Small sway',SMALL_SWAY),('trunk_sway_large','Large sway',LARGE_SWAY)]


MODELS=['gaitprint', 'multirocket_hydra', 'hydra', 'multirocket', 'minirocket', 'inceptiontime', 'multi_model_lstm']


NAMES={'gaitprint': 'GaitPrint-IMU', 'multirocket_hydra': 'MultiROCKET–HYDRA [21, 34]', 'hydra': 'HYDRA [21]', 'multirocket': 'MultiROCKET [20]', 'minirocket': 'MiniROCKET [19]', 'inceptiontime': 'InceptionTime [18]', 'multi_model_lstm': 'Multi-Model LSTM [15]'}


def apply_style():
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": .5, "axes.edgecolor": '#20252B', "text.color": '#20252B',
        "axes.labelcolor": '#20252B', "xtick.color": '#20252B', "ytick.color": '#20252B',
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.major.width": .5, "ytick.major.width": .5,
        "legend.frameon": True, "legend.facecolor": "white",
        "legend.edgecolor": "#BFC6CC", "legend.framealpha": 1,
        "legend.fancybox": False, "legend.borderpad": .35,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False})


def style():
    apply_style()
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
        'ytick.labelsize':7.5,'legend.fontsize':7.5,'svg.fonttype':'none'})


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,metadata={'Creator':'GaitPrint-IMU reproducibility release'})
    plt.close(fig)


def cell(frame,**keys):
    for key,value in keys.items(): frame=frame.loc[frame[key].eq(value)]
    if len(frame)!=1: raise ValueError(f'Expected one cell: {keys}, found {len(frame)}')
    return frame.iloc[0]


def duration_panel(participants):
    fig,a=plt.subplots(figsize=(180/25.4,142/25.4))
    a.text(-.20,1.12,'a',transform=a.transAxes,weight='bold',fontsize=10)
    for j,(condition,label,color) in enumerate(CONDITIONS):
        values=[participants.loc[participants.group.eq(condition)&participants.variant.eq('complete')&
            participants.window_samples.eq(w),'window_accuracy'].to_numpy()*100 for w in [100,200,300,400,500]]
        a.boxplot(values,positions=np.arange(1,6)+(j-.5)*.30,widths=.24,patch_artist=True,
            showfliers=False,manage_ticks=False,boxprops=dict(facecolor='white',edgecolor=color,linewidth=.7),
            medianprops=dict(color=color,linewidth=1),whiskerprops=dict(color=color,linewidth=.7),
            capprops=dict(color=color,linewidth=.7))
        a.plot(np.arange(1,6)+(j-.5)*.30,[v.mean() for v in values],'.',color=color,ms=4,label=label)
        for x,value in zip(range(1,6),values):
            a.text(x,22-j*11,f'{value.mean():.1f}',ha='center',va='center',color=color,fontsize=7)
    a.axvspan(1.6,2.4,color=HIGHLIGHT,zorder=0)
    a.set(xticks=np.arange(1,6),xlabel='Window duration (s)',ylabel='Participant accuracy (%)',ylim=(0,103))
    a.legend(loc='lower left',bbox_to_anchor=(0,1.08),ncol=2)
    return fig,a


LABELS=['GaitPrint-IMU', 'MultiROCKET–HYDRA [21, 34]', 'HYDRA [21]', 'MultiROCKET [20]', 'MiniROCKET [19]', 'InceptionTime [18]', 'Multi-Model LSTM [15]']


def identification_top(summary,output):
    names=[name.replace('MultiROCKET–HYDRA','MultiROCKET–\nHYDRA').replace('Multi-Model LSTM','Multi-Model\nLSTM') for name in LABELS]
    style()
    fig=plt.figure(figsize=(180/25.4,86/25.4))
    left=.18; seam=.535; right=.94; bottom=.17; height=.73
    a=fig.add_axes([left,bottom,seam-left,height])
    b=fig.add_axes([seam,bottom,right-seam,height],sharey=a)
    y=np.arange(len(MODELS))
    a.axhspan(-.5,.5,color=HIGHLIGHT,zorder=0)
    annotations=[]
    for j,(condition,label,color,marker) in enumerate([
        ('trunk_sway_small','Small sway',SMALL_SWAY,'o'),
        ('trunk_sway_large','Large sway',LARGE_SWAY,'s')]):
        rows=[cell(summary,model=m,group=condition,metric='window_accuracy') for m in MODELS]
        mean=np.array([r.estimate for r in rows])*100
        lo=np.array([r.ci_low for r in rows])*100
        hi=np.array([r.ci_high for r in rows])*100
        positions=y+(j-.5)*.40
        a.axvline(mean[0],color=color,linestyle=(0,(3,2)),linewidth=.7,alpha=.65,zorder=1)
        a.errorbar(mean,positions,xerr=[mean-lo,hi-mean],fmt=marker,color=color,
            ms=3,elinewidth=.75,capsize=1.5,label=label,zorder=3)
        for i,value in enumerate(mean):
            # Reserve a shared numerical column beyond both reference lines.
            a.text(113,positions[i],f'{value:.1f}',ha='right',va='center',color=color,fontsize=7)
            annotations.append(dict(model=MODELS[i],condition=condition,estimate_percent=float(value)))
    a.set(yticks=y,yticklabels=names,ylim=(len(y)-.5,-.5),
        xlim=(0,120),xticks=[0,25,50,75,100],xlabel='Window-level accuracy (%)')
    a.tick_params(axis='y',length=0,pad=4)
    a.spines['right'].set_visible(True)
    a.spines['right'].set_color('#BFC6CC')
    a.legend(loc='lower left',bbox_to_anchor=(left,.93),bbox_transform=fig.transFigure,
        ncol=2,columnspacing=1.2,fontsize=7)
    conditions=['fpa_small','fpa_large','step_width_small','step_width_large','walk_speed_small','walk_speed_large','dual_task']
    values=np.array([[cell(summary,model=m,group=c,metric='window_accuracy').estimate*100 for c in conditions] for m in MODELS])
    if not (0<=values.min()<=values.max()<=100):
        raise ValueError('Invalid accuracy outside 0–100%')
    lower=50 if values.min()>=50 else 0
    knots=np.arange(lower,101,10 if lower==50 else 20)
    norm=Normalize(vmin=lower,vmax=100)
    cmap=plt.get_cmap('viridis')
    heat=b.pcolormesh(np.arange(8)-.5,np.arange(len(y)+1)-.5,values,
        cmap=cmap,norm=norm,shading='flat',rasterized=False)
    b.set(xticks=np.arange(7),xticklabels=['FPA\n−10°','FPA\n+10°','Width\n−5.4 cm','Width\n+7 cm','Slow','Fast','Dual\ntask'])
    b.tick_params(axis='y',left=False,labelleft=False)
    b.tick_params(axis='x',length=0,labelsize=7)
    b.spines['left'].set_visible(False)
    for r in range(len(y)):
        for c in range(7):
            rgb=np.array(cmap(norm(values[r,c]))[:3])
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            luminance=linear@np.array([.2126,.7152,.0722])
            b.text(c,r,f'{values[r,c]:.1f}',ha='center',va='center',fontsize=7,
                color='black' if luminance>.179 else 'white')
    b.axhline(.5,color=SMALL_SWAY,lw=.8)
    color_axis=fig.add_axes([.955,bottom,.01,height])
    colorbar=fig.colorbar(heat,cax=color_axis,orientation='vertical',ticks=knots)
    colorbar.ax.tick_params(labelsize=7,pad=2)
    fig.text(.995,.962,'Accuracy\n(%)',ha='right',va='center',fontsize=7)
    colorbar.solids.set_rasterized(False)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    reference_x=[a.transData.transform((cell(summary,model='gaitprint',group=condition,
        metric='window_accuracy').estimate*100,0))[0] for condition in ['trunk_sway_small','trunk_sway_large']]
    minimum_gap=fig.dpi/25.4
    number_labels=[text for text in a.texts if re.fullmatch(r'\d+\.\d',text.get_text())]
    assert len(number_labels)==14
    for label in number_labels:
        if label.get_window_extent(renderer).x0-max(reference_x)<minimum_gap:
            raise ValueError('Sway number too close to a vertical reference line')
    save(fig,output/'identification_top.pdf')


def component_figure(design,output):
    style()
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    fig=plt.figure(figsize=(180/25.4,55/25.4))
    axes=[fig.add_axes([x/180,12/55,76/180,39/55]) for x in [13,103]]
    variants=['temporal','temporal_multiscale','temporal_spectral','complete']
    labels=['Temporal','+ Multiscale','+ Spectral','GaitPrint-IMU']
    for ax in axes:
        ax.axvspan(2.6,3.4,color=HIGHLIGHT,zorder=0)
        ax.set_xticks(np.arange(4),labels)
        ax.tick_params(axis='x',pad=7)
        ax.minorticks_off()
    for j,(condition,_,color) in enumerate(CONDITIONS):
        rows=[cell(design,group=condition,variant=v,window_samples=200) for v in variants]
        mean=np.array([r.participant_mean_accuracy for r in rows])*100
        lo=np.array([r.ci_low for r in rows])*100
        hi=np.array([r.ci_high for r in rows])*100
        pos=np.arange(4)+(j-.5)*.25
        axes[0].errorbar(pos,mean,yerr=[mean-lo,hi-mean],fmt='o' if j==0 else 's',color=color,ms=3,capsize=2,lw=.8)
        for x,value in enumerate(mean):
            axes[0].text(x,68-j*6,f'{value:.1f}',ha='center',va='center',color=color,fontsize=7)
        errors=[(1-r.pooled_window_accuracy)*100 for r in rows]
        axes[1].bar(pos,errors,width=.23,color=color)
        for x,value in zip(pos,errors):
            axes[1].annotate(f'{value:.1f}',(x,value),xytext=(0,5),textcoords='offset points',ha='center',color=color,fontsize=7)
    axes[0].set(ylabel='Participant-mean accuracy (%)',ylim=(60,102))
    axes[1].set(ylabel='Pooled window error (%)',ylim=(0,25))
    for x,letter in [(3,'b'),(93,'c')]:
        fig.text(x/180,53/55,letter,fontsize=10,weight='bold',va='top')
    save(fig,output/'component_panels.pdf')


def render_all(source,computed):
    output=computed/'figures'
    output.mkdir(parents=True,exist_ok=True)
    identification_top(pd.read_csv(computed/'statistics/identification_summary.csv'),output)
    component_figure(pd.read_csv(computed/'design/design_summary.csv'),output)
    mm=72/25.4
    document=fitz.open()
    page=document.new_page(width=180*mm,height=140*mm)
    with fitz.open(output/'identification_top.pdf') as top:
        page.show_pdf_page(fitz.Rect(0,0,180*mm,83*mm),top,0)
    with fitz.open(output/'component_panels.pdf') as bottom:
        page.show_pdf_page(fitz.Rect(0,85*mm,180*mm,140*mm),bottom,0)
    page.insert_text((3*mm,4*mm),'a',fontname='hebo',fontsize=10)
    document.save(output/'figure_3.pdf',garbage=4,deflate=True)
    document.close()
    duration_figure(source,computed,output)
    diagnostic_figure(computed/'figure_sources',output)


def duration_figure(ROOT,computed,OUT):
    names=[name.replace('MultiROCKET–HYDRA','MultiROCKET–\nHYDRA').replace('Multi-Model LSTM','Multi-Model\nLSTM') for name in LABELS]
    style()
    timing={m:{phase:json.loads((ROOT/'timing_final'/m/phase/file).read_text())
        for phase,file in [('registration','registration.json'),('inference','timing.json')]} for m in MODELS}
    people=pd.read_csv(ROOT/'design/participant_by_seed.csv')
    keys=['window_samples','variant','group','subject']
    participants=people.groupby(keys,as_index=False).mean(numeric_only=True)
    fig,a=duration_panel(participants)
    fig.set_size_inches(180/25.4,142/25.4)
    a.set_position([.073,.55,.37,.34])
    a.texts[0].set_transform(fig.transFigure)
    a.texts[0].set_position((.025,.93))
    c=fig.add_axes([.64,.55,.35,.34])
    rank=pd.read_csv(ROOT/'analysis/identification.csv')
    rank=rank.loc[rank.fold.eq(0)]
    values=np.array([[rank.loc[rank.model.eq(m)&rank.group.eq(condition),f'rank{k}'].mean()*100
        for condition in ['trunk_sway_small','trunk_sway_large'] for k in [1,3,5]] for m in MODELS])
    assert np.all(np.diff(values[:,:3],axis=1)>=0) and np.all(np.diff(values[:,3:],axis=1)>=0)
    cmap=plt.get_cmap('Purples')
    c.pcolormesh(np.arange(7)-.5,np.arange(8)-.5,values,cmap=cmap,vmin=0,vmax=100,shading='flat')
    c.set(ylim=(6.5,-.5),xticks=range(6),xticklabels=['1','3','5','1','3','5'],
        yticks=range(7),yticklabels=names,xlabel='Rank')
    c.tick_params(axis='both',length=0,labelsize=7.5,pad=3)
    c.axvline(2.5,color='white',lw=1.5)
    for r in range(7):
        for col in range(6):
            c.text(col,r,f'{values[r,col]:.1f}',ha='center',va='center',fontsize=7,
                color='white' if values[r,col]>65 else '#20252B')
    for x,label in [(1,'Small'),(4,'Large')]:c.text(x,-.90,label,ha='center',va='bottom',fontsize=7,clip_on=False)
    d=fig.add_axes([.09,.10,.34,.34])
    inset=fig.add_axes([.285,.27,.13,.135])
    claims=pd.read_csv(ROOT/'analysis/trial_claims.csv')
    grid=np.linspace(0,1,1001)
    for protocol,linestyle,label in [('enrolled','-','Enrolled impostors'),('subject_held_out','--','Unenrolled impostors')]:
        subset=claims.loc[claims.model.eq('gaitprint')&claims.protocol.eq(protocol)&claims.condition.str.startswith('trunk_sway_')]
        curves=[]
        for _,rows in subset.groupby(['seed','fold']):
            far,tar,_=roc_curve(rows.genuine,rows.score)
            points=pd.DataFrame({'far':far,'frr':1-tar}).groupby('far').frr.min()
            curves.append(np.interp(grid,points.index,points.values))
        curves=np.stack(curves)
        d.plot(grid*100,curves.mean(0)*100,linestyle,color=SMALL_SWAY,lw=1,label=label)
        inset.plot(grid*100,curves.mean(0)*100,linestyle,color=SMALL_SWAY,lw=.8)
        if protocol=='subject_held_out':
            d.fill_between(grid*100,np.maximum(0,curves.mean(0)-curves.std(0))*100,
                np.minimum(1,curves.mean(0)+curves.std(0))*100,color=SMALL_SWAY,alpha=.12,lw=0)
            inset.fill_between(grid*100,np.maximum(0,curves.mean(0)-curves.std(0))*100,
                np.minimum(1,curves.mean(0)+curves.std(0))*100,color=SMALL_SWAY,alpha=.12,lw=0)
    d.plot([0,20],[0,20],':',color='#BFC6CC',lw=.7)
    d.set(xlim=(0,20),ylim=(0,20),xlabel='False acceptance rate (%)',ylabel='False rejection rate (%)')
    d.legend(loc='lower left',bbox_to_anchor=(0,1.02),fontsize=7,ncol=2,columnspacing=.8,handletextpad=.4)
    d.add_patch(Rectangle((0,0),5,5,fill=False,edgecolor='#687782',lw=.65,linestyle=(0,(3,2))))
    inset.set(xlim=(0,5),ylim=(0,5),xticks=[2.5,5],yticks=[0,2.5,5])
    inset.tick_params(labelsize=6,pad=1,length=2)
    inset.text(.5,1.10,'Low-error region',transform=inset.transAxes,ha='center',fontsize=6.5)
    inset.plot([0,5],[0,5],':',color='#BFC6CC',lw=.6)
    for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#687782');spine.set_linewidth(.6)
    registration=fig.add_axes([.62,.10,.17,.34])
    processing=fig.add_axes([.82,.10,.17,.34],sharey=registration)
    for axis in [registration,processing]:
        axis.axhspan(-.5,.5,color=HIGHLIGHT,zorder=0)
        axis.set(ylim=(6.5,-.5),xlim=(0,145),xticks=[0,50,100],yticks=range(7))
        axis.tick_params(axis='x',labelsize=7)
    registration.set_yticklabels(names,fontsize=7.5)
    processing.tick_params(axis='y',left=False,labelleft=False)
    registration.set_xlabel('Baseline refit (s)',fontsize=7.5)
    processing.set_xlabel('Inference\n(ms/window)',fontsize=7.5)
    for i,m in enumerate(MODELS):
        seconds=timing[m]['registration']['compute_seconds_excluding_checkpoint_writes']
        registration.scatter(seconds,i,s=15,color='#687782',marker='s')
        registration.annotate(f'{seconds:.2f}',(seconds,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=6.5,color='#46535B')
        row=timing[m]['inference']['summary']['total_ms']
        processing.errorbar(row['median'],i,xerr=[[row['median']-row['q25']],[row['q75']-row['median']]],
            fmt='o',ms=3,color=SMALL_SWAY,capsize=1.5,lw=.7)
        processing.annotate(f'{row["median"]:.2f}',(row['q75'],i),xytext=(4,0),textcoords='offset points',va='center',fontsize=6.5,color=SMALL_SWAY)
    for x,y,label in [(.46,.93,'b'),(.025,.485,'c'),(.445,.485,'d')]:fig.text(x,y,label,weight='bold',fontsize=10)
    save(fig,OUT/'figure_4.pdf')
    (OUT/'rank_cells.json').write_text(json.dumps(dict(models=MODELS,columns=['Small rank 1','Small rank 3','Small rank 5','Large rank 1','Large rank 3','Large rank 5'],percent=values.tolist()),indent=2))


def diagnostic_figure(DATA,OUT):
    style()
    plt.rcParams.update({"axes.labelsize":9.5,"font.size":8,"legend.fontsize":8})
    fig=plt.figure(figsize=(180/25.4,98/25.4))
    rects=[(15,22,43,60),(76,22,43,60),(138,22,30,60)]
    a,b,c=[fig.add_axes((r[0]/180,r[1]/98,r[2]/180,r[3]/98)) for r in rects]
    teal,plum='#2A9D8F','#8E6C8A'
    conds=['trunk_sway_small','trunk_sway_large']
    effects=pd.read_csv(DATA/'participant_feature_budget_gains.csv')
    for i,condition in enumerate(conds):
     d=effects.loc[effects.condition.eq(condition),'accuracy_gain_pp'].sort_values()
     x=np.r_[d.iloc[0]-.5,d.to_numpy()];y=np.r_[0,np.arange(1,len(d)+1)/len(d)*100]
     a.step(x,y,where='post',color=[teal,plum][i],linestyle=['-','--'][i],label=['Small sway','Large sway'][i])
    a.axvline(0,color='#7F7F7F',linewidth=.7,linestyle=':')
    a.set(xlabel='Accuracy gain (pp)',ylabel='Cumulative participants (%)',ylim=(0,102),xlim=(-16,50))
    a.set_xticks([-10,0,20,40]);a.set_yticks([0,25,50,75,100]);l1=a.legend(loc='upper left',bbox_to_anchor=(-.03,1.2),handlelength=1.8)
    margins=pd.read_csv(DATA/'verification_margin_source.csv')
    genuine=margins[margins.probe_type.eq('Genuine claim')].copy()
    unknown=margins[margins.probe_type.eq('Unknown: best claim')].copy()
    bins=np.linspace(-.7,1.45,38)
    margin_exports=[]
    for d,label,color,ls in [(genuine,'Genuine claim',teal,'-'),(unknown,'Impostor search',plum,'--')]:
     # Repeated fits retain their own margin/threshold; equal total weight per physical trial.
     d=d.copy();d['weight']=1/d.groupby(['subject','condition','recording']).margin.transform('size')
     d['weight']/=d.weight.sum()
     b.hist(d.margin,bins=bins,weights=d.weight,density=True,histtype='step',color=color,linestyle=ls,linewidth=1,label=label)
     d['probe_type']=label;margin_exports.append(d)
    b.axvline(0,color='#7F7F7F',linewidth=.7,linestyle=':')
    b.set(xlabel='Score − decision threshold',ylabel='Probability density',xlim=(-.7,1.45),ylim=(0,None))
    b.set_xticks([-.5,0,.5,1]);l2=b.legend(loc='upper left',bbox_to_anchor=(-.03,1.2),handlelength=1.5,handletextpad=.4)
    rates=pd.read_csv(DATA/'wearer_rejection_heatmap.csv',index_col=0).drop(columns='rank')
    # Use identical row edges in every column. Matching face-colored edges prevent
    # PDF viewers from introducing white seams between adjacent vector cells.
    cmap=plt.get_cmap('cividis').copy()
    cmap.set_bad('#E6E6E6')
    im=c.pcolormesh(np.arange(5)-.5,np.arange(len(rates)+1)+.5,np.ma.masked_invalid(rates.to_numpy()*100),cmap=cmap,vmin=0,vmax=100,rasterized=False,edgecolors='face',linewidth=.1,antialiased=False)
    c.set_xlim(-.5,3.5)
    c.set_ylim(len(rates)+.5,.5)
    missing_rows,missing_cols=np.where(rates.isna().to_numpy())
    c.scatter(missing_cols,missing_rows+1,marker='x',s=5,linewidths=.4,color='#7F7F7F',zorder=3)
    c.set(xticks=[0,1,2,3],xticklabels=['Small','Large','Small','Large'],ylabel='Participant order',yticks=[1,15,30,45,61])
    c.tick_params(axis='x',labelsize=7,pad=2)
    c.axvline(1.5,color='white',linewidth=1)
    # Group model labels are categorical axis labels, not panel titles.
    c.text(.25,-.12,'GaitPrint',ha='center',va='top',transform=c.transAxes,fontsize=8)
    c.text(.75,-.12,'MR–HYDRA',ha='center',va='top',transform=c.transAxes,fontsize=8)
    c.xaxis.set_minor_locator(plt.NullLocator());c.yaxis.set_minor_locator(plt.NullLocator())
    cbax=fig.add_axes((171/180,22/98,2.1/180,60/98))
    cb=fig.colorbar(im,cax=cbax,ticks=[0,50,100]);cb.solids.set_rasterized(False)
    cb.ax.tick_params(labelsize=7,pad=1)
    cb.ax.set_title('FRR (%)',fontsize=8,pad=5)
    for i,ax in enumerate([a,b,c]):
        ax.text(-.27 if i==2 else -.25,1.03,'('+chr(97+i)+')',transform=ax.transAxes,weight='bold',fontsize=9.5,va='bottom')
    save(fig,OUT/'figure_5.pdf')
