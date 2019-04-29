"""
train_reshape:
reshape dataframes and time series in preparation for training.
"""

import random, csv, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra

def day2time(hL):
    """transform day format into datetime (for plotting)"""
    return [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in hL]

def hour2time(hL):
    """transform time format into datetime (for plotting)"""
    return [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hL]

def timeCol(g):
    """returns all the columns with time format"""
    return sorted([x for x in g.columns if bool(re.search("T",x))])

def overlap(hL,hL1):
    """return only the overlapping values"""
    return sorted(list(set(hL) & set(hL1)))

def binOutlier(y,nBin=6,threshold=3.5):
    """bin with special treatment for the outliers"""
    n = nBin
    ybin = [threshold] + [x*100./float(n-1) for x in range(1,n-1)] + [100.-threshold]
    pbin = np.unique(np.nanpercentile(y,ybin))
    n = min(n,pbin.shape[0])
    delta = (pbin[n-1]-pbin[0])/float(n-1)
    pbin = [np.nanmin(y).min()] + [x*delta + pbin[0] for x in range(n)] + [np.nanmax(y).max()]
    if False:
        plt.hist(y,fill=False,color="red")
        plt.hist(y,fill=False,bins=pbin,color="blue")
        plt.show()
        sigma = np.std(y) - np.mean(y)
    t = np.array(pd.cut(y,bins=np.unique(pbin),labels=range(len(np.unique(pbin))-1),right=True,include_lowest=True))
    t[np.isnan(t)] = -1
    t = np.asarray(t,dtype=int)
    return t, pbin

def binMatrix(X,nBin=6,threshold=2.5):
    """bin a continuum parametric matrix"""
    c_M = pd.DataFrame()
    psum = pd.DataFrame(index=range(nBin+2))
    for i in range(X.shape[1]):
        xcol = X[:,i]
        c_M.loc[:,i], binN = binOutlier(xcol,threshold=2.5)
        psum.loc[range(len(binN)),i] = binN
    return c_M, psum
    
def binVector(y,nBin=6,threshold=2.5):
    """bin a continuum parametric vector"""
    y, psum = t_f.binOutlier(y,nBin=nBin,threshold=threshold)
    y = np.array(y).astype(int)
    return y, psum

def binMask(typeV):
    """turn a comma separated list into a mask integer"""
    typeV = typeV.astype(str)
    typeV.loc[typeV=="nan"] = ""
    typeV = [re.sub("\]","",x) for x in typeV]
    typeV = [re.sub("\[","",x) for x in typeV]
    stypeL = [str(x).split(", ") for x in list(set(typeV))]
    stypeL = np.unique(list(chain(*stypeL)))
    maskV = []
    for i,p in enumerate(typeV):
        binMask = 0
        for j,t in enumerate(stypeL):
            binMask += 2**(j*bool(re.search(t,str(p))))
        maskV.append(binMask)
    return maskV

def id2dim(tist,sL,idField="id_poi"):
    """partition a dataset in learning matrices and reference data"""
    XL = []
    yL = []
    for i,g in tist.groupby(idField):
        X, y = g[sL].values, g['ref'].values
        XL.append(X)
        yL.append(y)
    XL = np.array(XL)
    yL = np.array(yL)
    return XL, yL

def id2group(sact,mist,hL,idField="id_poi"):
    """from two datasets create a learning matrix and reference data"""
    XL = []
    yL = []
    idL = []
    for i,g in sact.groupby(idField):
        setL = mist[idField] == i
        if sum(setL) == 0:
            continue
        y = mist.loc[setL,hL].values
        X = g[hL].values
        if y.sum() <= 0.:
            continue
        if X.sum() <= 0.:
            continue
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        XL.append(X.T)
        yL.append(y[0])
        idL.append(i)
    #XL = np.array(XL)
    #yL = np.array(yL)
    return XL, yL, idL

def factorize(X):
    """factorize all string fields"""
    if type(X).__module__ == np.__name__:
        return X
    for i in X.select_dtypes(object):
        X.loc[:,i], _ = pd.factorize(X[i])
    X = X.replace(float("Nan"),0)
    return X

def dayInWeek(df,idField="id_poi"):
    """morph a time series into an image representing a week (7xN) for convolutional neural network training"""
    hL = [x for x in df.columns if bool(re.search("T",x))]
    wd = day2time(hL)
    X, den, idL, normL = [], [], [], []
    for i,g in df.groupby(idField):
        dw = pd.DataFrame({idField:i,"year":[x[0] for x in wd],"week":[x[1] for x in wd],"day":[x[2] for x in wd],"count":g.values[0][1:]})        
        dwP = dw.pivot_table(index=[idField,"year","week"],columns="day",values="count",aggfunc=np.sum)
        setL = ~dwP.isnull().any(axis=1)
        dwP = dwP.loc[setL,:]
        norm = dwP.values.max()
        X.append(dwP.values/norm)
        normL.append(norm)
        den.append(dwP.reset_index())
        idL.append(i)
    den = pd.concat(den)
    return X, idL, den, norm

def splitInWeek(df,idField="id",isEven=True):
    """morph a time series into an image representing a week (7x24) for convolutional neural network training"""
    isocal = df['day'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").isocalendar())
    df.loc[:,"wday"] = [x[2] for x in isocal]
    df.loc[:,"week"] = [x[1] for x in isocal]
    df.loc[:,"month"] = df['day'].apply(lambda x:x[2:7])
    df.loc[:,"month"] = df[['month','week']].apply(lambda x: "%s-%02d" % (x[0],x[1]),axis=1)
    hL = [x for x in df.columns.values if bool(re.match("^[-+]?[0-9]+$",str(x)))]
    X = df[hL].values#[(10*7):(11*7)]
    imgL = []
    idL = []
    den = pd.DataFrame()
    for i,g in df.groupby(idField):
        for j,gm in g.groupby("month"):
            norm = gm[hL].max().max()
            if norm in [float('nan'),float('inf'),-float('inf')]:
                continue
            gl = gm[hL].values#/norm
            if gl.shape != (7,24):
                continue
            imgL.append(gl)
            idL.append({"id":i,"week":j})
            den = pd.concat([den,gm],axis=0)

    idL = pd.DataFrame(idL)
    N = len(imgL)
    X = np.concatenate(imgL).reshape((N,7,24))
    X[np.isnan(X)] = 0
    norm = X.max().max().max()
    X = X/norm
    return X, idL, den, norm

def loadMnist():
    """download mnist digit dataset for testing"""
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    X = np.concatenate((x_train,x_test))
    X = np.reshape(X,(X.shape[0],28,28))
    return X

def splitLearningSet(X,y,f_train=0.80,f_valid=0):
    """split a learning set"""
    seed = 128
    rng = np.random.RandomState(seed)
    f_test = 1 - f_valid
    N = X.shape[0]
    shuffleL = random.sample(range(N),N)
    partS = [0,int(N*f_train),int(N*(f_test)),N]
    y_train = np.asarray(y[shuffleL[partS[0]:partS[1]]])
    y_test  = np.asarray(y[shuffleL[partS[1]:partS[2]]])
    y_valid = np.asarray(y[shuffleL[partS[2]:partS[3]]],dtype=np.int32)
    x_train = np.asarray(X[shuffleL[partS[0]:partS[1]]])
    x_test  = np.asarray(X[shuffleL[partS[1]:partS[2]]])
    x_valid = np.asarray(X[shuffleL[partS[2]:partS[3]]],dtype=np.int32)
    return y_train, y_test, y_valid, x_train, x_test, x_valid

def day2isocal(pist,idField="id_poi",isDay=True):
    """convert a series of time over multiple years into a isocalendar time"""
    hL = pist.columns[[bool(re.search('-??T',x)) for x in pist.columns]]
    nhL = pist.columns[[not bool(re.search('-??T',x)) for x in pist.columns]]
    if isDay:
        cal = day2time(hL)
        ical = ["%02d-%02dT" % (x.isocalendar()[1],x.weekday()) for x in cal]
    else:
        cal = hour2time(hL)
        ical = ["%02d-%02dT%02d:00:00" % (x.isocalendar()[1],x.weekday(),x.hour) for x in cal]

    pist = pist[list(nhL) + list(hL)]
    pist.columns = list(nhL) + list(ical)
    mist = pist.groupby(pist.columns,axis=1).agg(np.mean)
    sist = pist.groupby(pist.columns,axis=1).agg(np.std)
    ucal = sist.columns[[bool(re.search('-??T',x)) for x in sist.columns]]
    sist.loc[:,ucal] = sist.loc[:,ucal]/mist.loc[:,ucal]
    mist.loc[:,nhL] = pist.loc[:,nhL]
    sist.loc[:,nhL] = pist.loc[:,nhL]
    return mist, sist

def hour2day(g):
    """remove the time and add up to days"""
    hL = g.columns[[bool(re.search('-??T',x)) for x in g.columns]]
    nhL = g.columns[[not bool(re.search('-??T',x)) for x in g.columns]]
    dL = [x[:11] for x in hL]
    g = g[list(nhL) + list(hL)]
    g.columns = list(nhL) + list(dL)
    c = g.groupby(g.columns,axis=1).agg(sum)
    c.loc[:,nhL] = c.loc[:,nhL]
    return c
    
def isocal2day(g,dateL,idField):
    """reform a dataframe into isocalendar"""
    hL = [x for x in g.columns.values if bool(re.match(".*-.*",str(x)))]
    dL = pd.DataFrame({"isocal":hL})
    dL = pd.merge(dL,dateL[['day','isocal']],how="left",on="isocal")
    dL = dL[dL['day'] == dL['day']]
    g = g[[idField] + list(dL['isocal'])]
    cL = [{x['isocal']:x['day']} for i,x in dL.iterrows()]
    g.columns = [idField] + list(dL['day'])
    return g

def ical2date(hL,year=None):
    """return a look up list with date and isocalendar"""
    if year == None:
        year = datetime.datetime.today().year
    orig = datetime.datetime.strptime(str(year) + "-01-01","%Y-%m-%d")
    lL = [orig + datetime.timedelta(days=x) for x in range(365)]
    ical = ["%02d-%02dT" % (x.isocalendar()[1],x.weekday()) for x in lL]
    lookUp = pd.DataFrame({"date":day2time(lL)
                           ,"ical":ical
                           })
    lookUp = lookUp.groupby("date").first().reset_index()
    lookUp = lookUp.groupby("ical").first().reset_index()
    lookUp = lookUp[lookUp['ical'].isin(ical)]
    return lookUp

