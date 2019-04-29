"""
train_viz:
utils for plotting feature/KPI distribution.
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
from sklearn.metrics import confusion_matrix
import geomadi.train_reshape as t_r

def plotHist(y,nBin=7,threshold=2.5,lab="",isLog=False):
    """plot histogram and its binning"""
    y = np.array(y)
    y = y[~np.isnan(y)]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, 10))
    mHist ,xHist = np.histogram(y,bins=(nBin+1))
    y1, psum = t_r.binOutlier(y,nBin=nBin,threshold=threshold)
    mHist1 = np.bincount(y1)
    plt.bar(psum[1:],mHist1,(max(psum)-min(psum))/float(len(psum)),label="bin_"+lab,fill=False,alpha=0.5,color=colors[0])
    plt.plot(psum[1:],mHist,label=lab)
    plt.legend()
    if isLog:
        plt.xscale('log',basex=10)
        plt.yscale('log',basey=10)
    plt.show()

def featureImportance(X,y):
    """display feature importance"""
    import xgboost as xgb
    import operator
    X_train = X
    y_train = y
    dtrain = xgb.DMatrix(x_train, label=y_train)
    gbdt = xgb.train(xgb_params, dtrain, num_rounds)
    importance = gbdt.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().show()

def plotPairGrid(X):
    """plot a pair grid"""
    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),xy=(.1, .6),xycoords=ax.transAxes,size = 24)

    cmap = sns.cubehelix_palette(light=1, dark = 0.1, hue = 0.5, as_cmap=True)
    sns.set_context(font_scale=2)
    gp = sns.PairGrid(X)
    gp.map_upper(plt.scatter, s=10, color = 'red')
    gp.map_diag(sns.distplot, kde=False, color = 'red')
    gp.map_lower(sns.kdeplot, cmap = cmap)
    #gp.map_lower(corrfunc);
    plt.show()

def plotRelevanceTree(X,y,isPlot=False):
    """feature importance by extra tree classifier"""
    clf = ExtraTreesClassifier()
    clf.fit(X,y)
    featL = clf.feature_importances_
    sortL = sorted(featL)
    sortI = [sortL.index(x) for x in featL]
    if isPlot:
        plt.plot(sorted(featL))
        plt.show()
    return featL, sortL
    
def plotBoxFeat(c_M,y):
    """plot boxplot of features"""
    c_M.loc[:,"y"] = y
    fig, ax = plt.subplots(2,2)
    cL = c_M.columns
    # c_M.boxplot(column="y_dif",by="t_pop_dens",ax=ax[0,0])
    # c_M.boxplot(column="y_dif",by="t_bast",ax=ax[0,1])
    c_M.boxplot(column="y_dif",by=cL[0],ax=ax[0,0])
    c_M.boxplot(column="y_dif",by=cL[1],ax=ax[0,1])
    c_M.boxplot(column="y_dif",by=cL[2],ax=ax[1,0])
    c_M.boxplot(column="y_dif",by=cL[3],ax=ax[1,1])
    ax[0,0].set_ylim(0,3.)
    ax[0,1].set_ylim(0,6.)
    ax[1,0].set_ylim(0,3.)
    ax[1,1].set_ylim(0,6.)
    plt.show()

def plotFeatCorr(t_M):
    """plot correlation in feature matrix"""
    corMat = t_M.corr()
    corrs = corMat.sum(axis=0)
    corr_order = corrs.argsort()[::-1]
    corMat = corMat.loc[corr_order.index,corr_order.index]
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.title('Correlation matrix between the features')
    plt.xticks(rotation=15)
    plt.yticks(rotation=45)
    plt.show()

def plotFeatCorrScatter(t_M):
    """plot correlation + scatter matrix"""
    pd.plotting.scatter_matrix(t_M, diagonal="kde")
    plt.tight_layout()
    plt.show()
    
def plotMatrixCorr(X1,X2):
    """plot cross correlation"""
    X1 = np.array(X1)
    X2 = np.array(X2)
    xcorr = np.corrcoef(X1,X2)
    from matplotlib import gridspec
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[:,0])
    ax1.imshow(X1)
    ax2 = fig.add_subplot(gs[:,1])
    ax2.imshow(X2)
    ax3 = fig.add_subplot(gs[0,2])
    ax3.imshow(xcorr)
    ax4 = fig.add_subplot(gs[1,2])
    ax4.hist(xcorr.ravel(), bins=20, range=(0.0, 1.0), fc='k', ec='k')
    gs.update(wspace=0.5, hspace=0.5)
    ax1.set_title("own data")
    ax2.set_title("customer data")
    ax3.set_title("correlation matrix")
    ax4.set_title("correlation overall")
    plt.show()
    #xcorr = np.correlate(X1,X2,mode='full')
    #xcorr = sp.signal.correlate2d(X1,X2)
    #xcorr = sp.signal.fftconvolve(X1, X2,mode="same")

def plotPerfHeat(scorL):
    """ plot performance heatmap"""
    scorP = scorL.copy()
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    tP = [x for x in scorP.columns if bool(re.search("v_",x))]
    scorL.sort_values(tL[-1],inplace=True)
    scorP.sort_values(tP[-1],inplace=True)
    sns.set(font_scale=1.2)
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    scorV = scorP[tP].apply(clampF)
    labP = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    scorP.index = scorP[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]
    yP = scorP[scorP[tP[-1]]>0.6].index[0]
    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ax[0].set_title("train")
    ax[0] = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax[0])
    ax[0].hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax[0].set_xticklabels(labL)
    ax[1].set_title("validation")
    ax[1] = sns.heatmap(scorP[tP],cmap=cmap,linewidths=.0,cbar=cbar_ax,ax=ax[1])
    ax[1].hlines(y=yP,xmin=tP[0],xmax=tP[-1],color="r",linestyle="dashed")
    ax[1].set_xticklabels(labP)
    for i in range(2):
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
    plt.show()

def plotPerfHeatDouble(scorP,scorL):
    """plot double sided performance heatmap"""
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    tP = [x for x in scorP.columns if bool(re.search("r_",x))]    
    scorL.sort_values(tL[-1],inplace=True)
    scorP.sort_values(tP[-1],inplace=True)
    sns.set(font_scale=1.2)
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    scorV = scorP[tP].apply(clampF)
    labP = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    scorP.index = scorP[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]
    yP = scorP[scorP[tP[-1]]>0.6].index[0]
    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ax[0].set_title("train")
    ax[0] = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax[0])
    ax[0].hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax[0].set_xticklabels(labL)
    ax[1].set_title("validation")
    ax[1] = sns.heatmap(scorP[tP],cmap=cmap,linewidths=.0,cbar=cbar_ax,ax=ax[1])
    ax[1].hlines(y=yP,xmin=tP[0],xmax=tP[-1],color="r",linestyle="dashed")
    ax[1].set_xticklabels(labP)
    for i in range(2):
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
    plt.show()

def plotPCA(X):
    """calculates and plot a PCA"""
    pca = PCA().fit(X)
    y = np.std(pca.transform(X), axis=0)**2
    x = np.arange(len(y)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    ax.set_yscale('log')
    plt.show()
    foo = pca.transform(X)
    bar = pd.DataFrame({"PC1":foo[0,:],"PC2":foo[1,:],"Class":y})
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    plt.show()

def plotImg(X,nCol=3):
    """plot images on multiple columns"""
    f, ax = plt.subplots(ncols=nCol)
    N = int(X.shape[0]/(nCol))
    for i,a in enumerate(ax):
        a.imshow(X[i*N:(i+1)*N,:])
    plt.show()

def kpiDis(scor,tLab="",saveF=None,col_cor="y_cor",col_dif="y_dif",col_sum="sum",isRel=True,ax=None):
    """plot cumulative histogram of KPI: standard deviation and correlation"""
    nbin = 20
    nRef = sum(~np.isnan(scor[col_sum]))
    nShare = sum(~np.isnan(scor[col_cor]))
    locShare = [(x/nbin) for x in range(nbin+1)]
    difShare = [np.sum(np.abs(scor[col_dif]) < x)/nRef for x in locShare]
    corShare = [np.sum(scor[col_cor] > x)/nRef for x in locShare]
    cor = scor[col_cor]
    cor = cor[~np.isnan(cor)]
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    #plt.hist(cor,bins=20,normed=1,histtype='step',cumulative=-1,label='Reversed emp.')
    if isRel:
        ax.plot(locShare,difShare,label="relative error",color="b")
        ax.fill_between(locShare[:9],0,difShare[:9],label="err < 0.4",color="b",alpha=.5)
        ax.fill_between(locShare[:11],0,difShare[:11],label="err < 0.5",color="b",alpha=.3)
    else :
        ax.plot(locShare,difShare,label="relative difference",color="b")
        ax.fill_between(locShare[:5],0,difShare[:5],label="err < 0.2",color="b",alpha=.5)
        ax.fill_between(locShare[:7],0,difShare[:7],label="err < 0.3",color="b",alpha=.3)
    ax.plot(locShare,corShare,label="correlation",color="g")
    ax.fill_between(locShare[12:],0,corShare[12:],label="corr > 0.6",color="g",alpha=.5)
    ax.fill_between(locShare[10:],0,corShare[10:],label="corr > 0.5",color="g",alpha=.3)
    ax.set_ylabel("covered locations")
    ax.set_xlabel("relative value")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    #fig.patch.set_alpha(0.7)
    ax.patch.set_alpha(0.0)
    ax.grid(b=True, which='major', color='k', linestyle='--',linewidth=0.25,axis="both")
    #ax.minorticks_on()
#    ax.tick_params(color="k")
    ax.set_title(tLab+" - locations: " + str(nShare) + "/" + str(nRef))
    ax.legend()
    if saveF:
        plt.savefig(saveF)
    #    plt.grid()
    else:
        if not ax:
            plt.show()

def plotHistogram(y,nbin=20,label="metric",ax=None):
    """plot histogram and cumulative distribution"""
    y = np.nan_to_num(y)
    plt.rcParams['axes.facecolor'] = 'white'
    values, base = np.histogram(y, bins=40)
    values = values/sum(values)
    cumulative = np.cumsum(values)
    values = values/max(values)
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    # n, bins, patches = ax.hist(y,nbin,normed=1,histtype='step',cumulative=True,label='Empirical')
    # ax.hist(y,bins=bins,normed=1,histtype='step',cumulative=-1,label='Reversed emp.')
    ax.step(base[:-1], cumulative, c='blue',label="empirical",linestyle="-.")
    ax.step(base[:-1], 1.-cumulative, c='green',label="reversed emp",linestyle="-.")
    ax.step(base[:-1], values, c='red',label="histogram",linewidth=2,linestyle="-")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(label)
    ax.set_ylabel('Likelihood of occurrence')
    ax.grid()
    if not ax:
        plt.show()

def plotTimeSeries(g,ax=None,hL=[None]):
    """plot all time series in a data frame on different rows"""
    groups = list(range(g.shape[1]))
    i = 1
    cL = g.columns
    X1 = g.values
    if not ax:
        fig, ax = plt.subplots(1,1)
    if any(hL):
        t = t_r.day2time(hL)
    else:
        t = range(g.shape[0])
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(t,X1[:, group])
        plt.title(cL[group], y=0.5, loc='right')
        i += 1
    plt.xticks(rotation=15)

def plotConfidenceInterval(y,ax=None,label="value",nInt=5,color="blue"):
    """plot a confidence interval and scatter points"""
    x, yd, xf, yf = s_l.serRunAvDev(y,nInt=nInt)
    y_up = np.array(yf) + .5*np.array(yd)
    y_dw = np.array(yf) - .5*np.array(yd)
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.scatter(x,y,color=color,marker="+",label=label,alpha=.25)
    ax.plot(xf,yf,color=color,linestyle='-.',lw=2,label=label,alpha=.5)
    xf[0] = 0.
    xf[nInt-1] = 1.
    ax.fill_between(xf,y_up,y_dw,alpha=.25,label='std interval',color=color)
    ax.legend()

def plotOccurrence(y,ax=None):
    """plot histogram of occurrences of values"""
    t, n = np.unique(y,return_counts=True)
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.bar(t,n)
    for tick in ax.get_xticklabels():
            tick.set_rotation(15)
    
def plotCorr(X,labV=None,ax=None):
    """plot a correlation heatmap"""
    if labV:
        corMat = pd.DataFrame(X,columns=labV).corr()
    else:
        corMat = np.corrcoef(X.T)
    sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn',ax=ax)

def plotConfMat(y,y_pred):
    """plot confusion matrix"""
    cm = confusion_matrix(y,y_pred)
    cm = np.array(cm)
    plt.xlabel("prediction")
    plt.ylabel("score")
    plt.imshow(cm)
    plt.grid(b=None)
    plt.show()
    return cm

def plotHyperPerf(scorV):
    """plot a double boxplot showing the performances per hyperparameter"""
    scorM = t_r.factorize(scorV.copy())
    setL = ((scorM.var(axis=0)/scorM.mean(axis=0)**2).abs() > 1e-4) | (scorV.columns.isin(['cor','err']))
    scorV = scorV[scorV.columns[setL]]
    scorM = t_r.factorize(scorV.copy())
    tL = [x for x in scorM.columns if not any([x == y for y in ["cor","err","time"]])]
    for i in tL:
        fig, ax = plt.subplots(1,2)
        scorV.boxplot(by=i,column="cor",ax=ax[0])
        scorV.boxplot(by=i,column="err",ax=ax[1])    
        plt.show()
    return scorV
