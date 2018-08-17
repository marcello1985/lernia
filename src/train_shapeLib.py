#%pylab inline
##https://stackoverflow.com/questions/41860817/hyperparameter-optimization-for-deep-learning-structures-using-bayesian-optimiza
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
from scipy import signal
from scipy.cluster import hierarchy as hc
#from scipy.optimize import leastsq as least_squares
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, wminkowski, squareform
from scipy.stats import multivariate_normal as mvnorm
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
#from statsmodels.formula.api import ols
#import statsmodels
#import statsmodels.api as sm
from tzlocal import get_localzone
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate as interpolate
import scipy.signal
import seaborn as sns
import sklearn as sk
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import geomadi.train_filter as t_f
import importlib
from itertools import chain
import geomadi.series_lib as s_l

tz = get_localzone()
nBin = 5

def plog(text):
    print(text)


class shapeLib:
    """feature caracterisation for time signals"""
    def __init__(self,X):
        self.X = np.array(X)

    def periodic(self,period=24):
        """feature selection of signals over period"""
        n_per = int(self.X.shape[1]/period)
        X_d = np.zeros(shape=(self.X.shape[0],period))
        for i in range(n_per):
            X_d = X_d + self.X[:,i*period:(i+1)*period]

        def ser_fun(x,t,param):
            #return x[0]*np.exp(-x[1]*(t-x[2])**2) + x[3]*np.exp(-x[4]*(t-x[5])**2)
            return x[0] + x[1] * t + x[2] * t * t
        def ser_fun_min(x,t,y,param):
            return ser_fun(x,t,param) - y
        param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
        rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
        X_d = X_d / rsum[:,np.newaxis]
        X_d[np.isnan(X_d)] = 0
        t = np.array(range(X_d.shape[1])).astype('float')
        x0 = [2.29344311e-02,4.91473902e-03,-1.56686399e+01]#,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
        param = [2.*np.pi/(18.),2.*np.pi/(7.*18.)]
        i = 606
        convL = []
        for i in range(X_d.shape[0]):
            res = least_squares(ser_fun_min,x0,args=(t,X_d[i],param))
            convL.append(res.x)
        if False:
            X = np.array(sact[hL])
            X = np.array(vist[hL])
            X_d = np.zeros(shape=(X.shape[0],period))
            n_per = int(X.shape[1]/period)
            for i in range(n_per):
                X_d = X_d + X[:,i*period:(i+1)*period]
            X_d = X_d.mean(axis=0)
            X_d = X_d/sum(X_d)
            t = np.array(range(X_d.shape[0])).astype('float') + 11
            res = least_squares(ser_fun_min,x0,args=(t,X_d,param))
            plt.plot(t,X_d,label="original")
            plt.plot(t,ser_fun(res.x,t,param),label="parab fit")
            plt.legend()
            plt.xlabel("hour")
            plt.ylabel("count")
            plt.show()
        return pd.DataFrame(convL,columns=["t_interc","t_slope","t_conv"])

    def daily(self,period=24):
        """feature selection of signals over period"""
        n_per = int(self.X.shape[1]/period)
        X1 = self.X[:,:period*n_per]
        X_d = X1.reshape(-1,n_per,period).sum(axis=2)
        x0 = [2.29344311e-02,4.91473902e-03,1.56686399e+01,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
        param = [2.*np.pi/(7),2.*np.pi/(7.*4)]
        def ser_fun(x,t,param):
            #return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
            return x[0] + x[1] * np.sin(param[0]*t + x[2]) + x[3]*t + x[4]*t*t
        def ser_fun_min(x,t,y,param):
            return ser_fun(x,t,param) - y

        rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
        X_d = X_d / rsum[:,np.newaxis]
        X_d[np.isnan(X_d)] = 0
        t = np.array(range(X_d.shape[1])).astype('float')
        convL = []
        for i in range(X_d.shape[0]):
            res = least_squares(ser_fun_min,x0,args=(t,X_d[i],param))
            convL.append(res.x)
        if False:
            X = np.array(sact[hL])
            X = np.array(vist[hL])
            n_per = int(X.shape[1]/period)
            X1 = X[:,:period*n_per]
            X_d = X1.reshape(-1,n_per,period).sum(axis=2)
            X_d = X_d.mean(axis=0)
            # rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
            # X_d = X_d / rsum[:,np.newaxis]
            t = np.array(range(X_d.shape[0])).astype('float')
            res = least_squares(ser_fun_min,x0,args=(t,X_d,param))
            plt.plot(t,X_d,label="original")
            plt.plot(t,ser_fun(x1,t,param),label="complete fit")
            plt.plot(t,ser_fun([x1[0],x1[1],x1[2],0.,0.],t,param),label="sinus part")
            plt.plot(t,ser_fun([x1[0],0.,0.,x1[3],x1[4]],t,param),label="parab part")
            plt.legend()
            plt.xlabel("day")
            plt.ylabel("count")
            plt.show()
        return pd.DataFrame(convL,columns=["t_interc","t_slope","t_conv"])

    
    def seasonal(self,period=18):
        """feature selection of signals over season"""
        param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
        x1 = [-6.03324843e+02,1.25552574e+02,-3.98806217e+00,1.67340000e-02,1.98480000e-02,9.72000000e-04]
        def ser_sin(x,t,param):
            #return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
            return x[0] + x[1] * np.sin(param[0]*t + x[2]) + x[3]*t + x[4]*t*t
        
        def ser_sin_min(x,t,y,param):
            return ser_sin(x,t,param) - y
        t = np.array(range(self.X.shape[1])).astype('float')
        convL = []
        for i in range(self.X.shape[0]):
            res = least_squares(ser_sin_min,x1,args=(t,self.X[i],param))
            convL.append([res.x[1],res.x[3],res.x[4]])
        if False:
            X = np.array(sact[hL])
            X = np.array(vist[hL])
            X_d = X.mean(axis=0)
            t = np.array(range(X_d.shape[0])).astype('float')
            res = least_squares(ser_sin_min,x1,args=(t,X_d,param))
            plt.plot(t,X_d,label="original")
            plt.plot(t,ser_sin(res.x,t,param),label="sinus + parab fit")
            plt.legend()
            plt.xlabel("hour")
            plt.ylabel("count")
            plt.show()
        return pd.DataFrame(convL,columns=["t_ampl","t_trend1","t_trend2"])

    def monthly(self,period=24):
        """feature selection of signals over season"""
        param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
        def ser_sin(x,t,param):
            return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
        def ser_sin_min(x,t,y,param):
            return ser_sin(x,t,param) - y
        t = np.array(range(self.X.shape[1])).astype('float')
        convL = []
        x1 = [-6.03324843e+02,1.25552574e+02,-3.98806217e+00,1.67340000e-02,1.98480000e-02,9.72000000e-04]
        for i in range(self.X.shape[0]):
            res = least_squares(ser_sin_min,x1,args=(t,self.X[i],param))
            convL.append([res.x[0],res.x[1],res.x[3]])
        if False:
            i = 0
            i = 1188
            t = np.array(range(X.shape[1])).astype('float')
            res = least_squares(ser_sin_min,x1,args=(t,X[i],param))
            plt.plot(t,X[i],label="original")
            plt.plot(t,ser_sin(res.x,t,param),label="sinus + parab fit")
            plt.plot(t,ser_sin([x1[0],x1[1],x1[2],0.,0.],t,param),label="sinus part")
            plt.plot(t,ser_sin([x1[0],0.,0.,x1[3],x1[4]],t,param),label="parab part")
            plt.plot(t,ser_sin([x1[0],x1[1],x1[2],x1[3],x1[4]],t,param),label="complete fit")
            plt.legend()
            plt.xlabel("hour")
            plt.ylabel("count")
            plt.show()

        return pd.DataFrame(convL,columns=["t_m_inter","t_m_trend1","t_m_trend2"])

    def statistical(self):
        t_M = pd.DataFrame()
        t_M.loc[:,'t_max'] = np.max(self.X,axis=1)
        t_M.loc[:,'t_std'] = np.std(self.X,axis=1)
        t_M.loc[:,'t_sum'] = np.nansum(self.X,axis=1)
        t_M.loc[:,'t_median'] = self.X.argmax(axis=1)
        return t_M

    def plotPCA(self):
        """calculates and plot a PCA"""
        pca = PCA().fit(self.X)
        y = np.std(pca.transform(self.X), axis=0)**2
        x = np.arange(len(y)) + 1
        fig, ax = plt.subplots(1)
        ax.plot(x, y, "o-")
        plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
        plt.ylabel("Variance")
        ax.set_yscale('log')
        plt.show()
        foo = pca.transform(self.X)
        bar = pd.DataFrame({"PC1":foo[0,:],"PC2":foo[1,:],"Class":y})
        sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
        plt.show()

    def calcPCA(self):
       pca = PCA().fit(self.X)
       return pca.transform(self.X)
       
    def plotImg(self):
        plt.imshow(self.X)
        plt.show(self.X)
        colN = 3
        f, ax = plt.subplots(ncols=colN)
        N = int(X.shape[0]/(colN))
        for i,a in enumerate(ax):
            a.imshow(self.X[i*N:(i+1)*N,:])
        plt.show()

class reduceFeature:
    """reduce signal complexity into stable features"""
    def __init__(self,X):
        self.X = np.array(X)

    def fillMissing(self,y):
        """fill missing values"""
        nans, ym = s_l.interpMissing(y)
        y[nans] = ym
        return y
        
    def interpMissing(self):
        """interpolate missing values"""
        self.X = np.apply_along_axis(lambda y: self.fillMissing(y),axis=1,arr=self.X)

    def smooth(self,width=3,steps=5):
        """smooth signal"""
        self.X = np.apply_along_axis(lambda y: s_l.serSmooth(y,width=width,steps=steps),axis=1,arr=self.X)
    def runAv(self,steps=5):
        """running average"""
        self.X = np.apply_along_axis(lambda y: s_l.serRunAv(y,steps=steps),axis=1,arr=self.X)

    def getMatrix(self):
        """return matrix"""
        return self.X

    def calibFact(self,indexL):
        """use a subset for calibration"""
        return self.X[indexL].sum(axis=1)
        
    def interpDoubleGauss(self,t,y,x0=[None],p0=[None]):
        """interpolate via double Gaussian"""
        if not any(x0):
            x0 = [3.,3.,min(y)]
            x0[0] = 1./(2.*np.power(x0[0],2.))
            x0[1] = 1./(2.*np.power(x0[1],2.))
        if not any(p0):
            p0 = [8.5,13.5,max(y)*.75,max(y)*.9]
            # p0[2] = p0[2]/(np.sqrt(2.*np.pi)*x0[0])
            # p0[3] = p0[3]/(np.sqrt(2.*np.pi)*x0[1])
        def ser_dExp(x,t,p):
            exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
            exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
            return x[2] + p[2]*exp1 + p[3]*exp2
        def ser_residual(x,t,y,p):
            return (y-ser_dExp(x,t,p))
        res = least_squares(ser_residual,x0,args=(t,y,p0),method="trf",loss="soft_l1")#,bounds=(0,200.))
        #res = least_squares(ser_residual,x0,args=(t,y,p0))
        x0 = res.x
        def ser_dExp(p,t,x):
            exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
            exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
            return x[2] + p[2]*exp1 + p[3]*exp2
        def ser_residual(p,t,y,x):
            return (y-ser_dExp(p,t,x))
        res = least_squares(ser_residual,p0,args=(t,y,x0),method="trf",loss="soft_l1")#,bounds=(0,200.))
        #res = least_squares(ser_residual,p0,args=(t,y,x0))
        p0 = res.x
        t1 = t#np.linspace(t[0],t[len(t)-1],50)
        y1 = ser_dExp(p0,t1,x0)
        return t1, y1, p0, x0

    def interpPoly(self,t,y,x0=[None],p0=[None]):
        """interpolate via polynomial"""
        if not any(x0):
            x0 = [1.50208343e+01,-5.06158347e+00,3.59972679e-01,2.50569233e-01,-2.91416175e-02,1.09982221e-03,-1.37937148e-05]
            p0 = [0]
        def ser_dExp(x,t,p):
            res1 = x[0] + t*x[1] + t*t*x[2] + t*t*t*x[3] + t**4*x[4] + t**5*x[5] + t**6*x[6]
            return res1
        def ser_residual(x,t,y,p):
            return (y-ser_dExp(x,t,p))
        res = least_squares(ser_residual,x0,args=(t,y,p0))
        x0 = res.x
        t1 = t#np.linspace(t[0],t[len(t)-1],50)
        y1 = ser_dExp(x0,t1,00)
        return t1, y1, p0, x0

    def interpFun(self,t,y):
        """interpolate via fitting function"""
        # t,y1,p1,x1 = self.interpPoly(t,y)
        # return x1
        t,y1,p1,x1 = self.interpDoubleGauss(t,y)
##        t,y1,p1,x1 = self.interpDoubleGauss(t,y,x0=x1,p0=p1)
        if False:
            def ser_dExp(x,t,p):
                exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
                exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
                return x[2] + p[2]*exp1 + p[3]*exp2
            pt = p1.copy()
            pt[2] = 0
            y3 = ser_dExp(x1,t,pt)
            pt = p1.copy()
            pt[3] = 0
            y4 = ser_dExp(x1,t,pt)
            print(p1)
            print(x2)
            plt.plot(t,y,label="vist")
            plt.plot(t,y1,label="ref_gauss")
            plt.plot(t,y3,label="gauss_1")
            plt.plot(t,y4,label="gauss_2")
            plt.xlabel("hour")
            plt.ylabel("count")
            plt.legend()
            plt.show()
        return y1 #np.concatenate([p1,x1])

    def fit(self):
        t = np.array([int(x) for x in range(self.X.shape[1])])
        self.X = np.apply_along_axis(lambda y: self.interpFun(t,y),axis=1,arr=self.X)
    
    def getFeatures(self):
        t = np.array([int(x) for x in range(self.X.shape[1])])
        X1 = np.apply_along_axis(lambda y: self.interpFun(t,y),axis=1,arr=self.X)
        return X1

        
class featureSel:
    """reduce dimensionality of training matrices"""
    def __init__(self):
        return 
        
    def variance(self,X):
        """reduce dimensionality"""
        p_M = sp.stats.describe(X)
        varL = np.sqrt(p_M.variance)/p_M.mean
        cSel = [x for x in range(X.shape[1])]
        cNeg = [i for i,x in enumerate(varL) if np.isnan(x) or np.abs(x) < 1.]
        T = np.delete(X,cNeg,axis=1)
        return T, cNeg

    def std(self,X,n_tail=2):
        p_M = X.describe()
        varL = p_M.iloc[2,:]/p_M.iloc[1,:]
        varL = np.abs(varL)
        varL = varL.sort_values(ascending=True)
        return varL.head(varL.shape[0]-n_tail).index, varL
    
    def chi2(self,X,y):
        X_new = SelectKBest(chi2,k=2).fit_transform(X,y)
    
    def kmeans(self,X,n_clust=5):
        pca = PCA().fit(X)
        A_q = pca.components_.T
        if False:
            plt.imshow(A_q)
            plt.show()

        kmeans = KMeans(n_clusters=n_clust).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_
        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances(A_q[i, :], cluster_centers[c, :])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_] 

    def treeClas(self,X,y):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        clf.feature_importances_  
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        X_new.shape               
        
class scoreLib:
    """build a scoring based on correlation and regression"""
    def __init__(self):
        return

    def score(self,X1,X2,hL=[],id_name="cluster"):
        """return a scoring based on correlation and regression"""
        # clf = linear_model.LinearRegression()
        # clf = linear_model.Ridge(alpha = .5)
        # clf = linear_model.MultiTaskElasticNet()
        # clf = linear_model.LogisticRegression()
        # clf = linear_model.HuberRegressor()
        clf = linear_model.BayesianRidge()
        # clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        lr = LinearRegression()
        scoreM = pd.DataFrame(index=X1.index)
        missingL = []
        for i,x2 in X2.iterrows():
            selI = X1[id_name] == x2[id_name]            
            if any(selI)==False:
                missingL.append(x2[id_name])
                continue
            X = np.array(X1.loc[selI,hL])#.transpose()
            pline = np.array(x2.loc[hL])
            if sum(pline) <= 0.:
                missingL.append(x2[id_name])
                continue
            sline = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X)
            cline = np.apply_along_axis(lambda x: sp.stats.pearsonr(x,pline),axis=1,arr=X)[:,0]
            iline = np.apply_along_axis(lambda x: sklearn.metrics.mutual_info_score(x,pline),axis=1,arr=X)
            lline = lr.fit(X.T,pline).coef_
            #chis  = np.apply_along_axis(lambda x: sp.stats.chisquare(x,pline),axis=1,arr=X)[:,0]
            rline = clf.fit(X.transpose(),pline).coef_ ##fit = sm.OLS(pline, msact).fit() ##fit = regressor.fit(msact,pline)
            rsum = np.nansum(X,axis=1).astype('float')
            if False:
                plt.plot(X.transpose())
                plt.plot(range(pline.shape[0]),pline,linestyle="--")
                plt.show()
            if False:
                X = X / rsum[:,np.newaxis]
                pline = pline / np.nansum(pline)
            scoreM.loc[selI,"y_cor"] = cline
            delta = (np.nanmax(rline) - np.nanmin(rline))
            if not np.isnan(delta):
                scoreM.loc[selI,"y_reg"]  = (rline - np.nanmin(rline))/delta
            delta = pline.sum()#+ sline.sum()
            if not np.isnan(delta):            
                scoreM.loc[selI,"y_dif"] = abs(sline - pline.sum())/delta
            scoreM.loc[selI,"y_lin"] = lline
            scoreM.loc[selI,"y_ent"] = iline
            scoreM.loc[selI,"sum"] = sline
            scoreM.loc[selI,id_name] = x2[id_name]
            #scoreM.loc[selI,"y_chi"] = chis
            # logit = sm.Logit(y,X).fit()
            # sact.loc[selI,"logit"] = logit.params
            # lda = LinearDiscriminantAnalysis().fit(X, y)
            # scaling = lda.scalings_[:, 0]
        print("missing clusters:" + str(len(missingL)))
        print(missingL)
        print("not defined: %.3f" % (scoreM[scoreM[id_name] != scoreM[id_name]].shape[0]/scoreM.shape[0]) )
        return scoreM#.replace(np.nan,0)

    def regression(self,X,y):
        clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        fit = clf.fit(X,y)
        return fit.coef_

    def regQuadratic(self,y):
        def ser_fun(x,t):
            return x[0] + x[1]*t + x[2]*t*t + x[3]*t*t*t + x[4]*t*t*t*t
        def ser_fun_min(x,t,y):
            return ser_fun(x,t) - y
        x0 = [2.29344311e-02,4.91473902e-03,-1.56686399e+01,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
        t = np.array(range(y.shape[0]))
        x1,n = least_squares(ser_fun_min,x0,args=(t,y))
        return ser_fun(x1,t)

class clustLib():
    def __init__(self):
        return

    def group2Nei(self,X1,X2,max_d=0.1,max_nei=None):
        cellL = pd.DataFrame()
        for i in range(X1.shape[0]):
            x_c, y_c = X1[['x','y']].iloc[i]
            disk = ((X2['X']-x_c)**2 + (X2['Y']-y_c)**2)
            disk = disk.loc[disk <= max_d**2]
            if max_nei:
                if disk.shape[0] > max_nei:
                    disk = disk.sort_values()
                    disk = disk.head(max_nei)
            tmp = cells.loc[disk.index]
            tmp.loc[:,"cluster"] = cvist['cluster'].iloc[i]
            cellL = pd.concat([cellL,tmp],axis=0)
        cellL = cellL.groupby('cilac').head(1)

    def groupNei(self,X1,max_d=0.1,max_nei=None):
        Z = linkage(X1, 'ward')
        return fcluster(Z,max_d,criterion='distance')#k, criterion='maxclust')

        
    # tree = cKDTree(np.c_[tvist['x'],tvist['y']])
    # point_neighbors_list = []
    # for point in points:
    #     distances, indices = tree.query(point, len(points), p=2, distance_upper_bound=max_distance)
    #     point_neighbors = []
    #     for index, distance in zip(indices, distances):
    #         if distance == inf:
    #             break
    #         point_neighbors.append(points[index])
    #     point_neighbors_list.append(point_neighbors)
    
        
    def cluster(self,mact):
        """spectral clustering and area sorting"""
        A_mA = mact - mact.mean(1)[:,None]
        ssA = (A_mA**2).sum(1);
        rsum = mact.sum(axis=1).astype('float')
        psum = np.percentile(rsum,[x*100./4. for x in range(5)])
        rquant = pd.qcut(rsum,5,range(5)) > 1
        rquant = rsum > psum[2]
        act = act[rquant]
        mact = mact[rquant]
        rsum = mact.sum(axis=1).astype('float')
        mact = mact / rsum[:,np.newaxis]
        cpact1 = np.corrcoef(mact)
        #pact4 = pdist(mact,'correlation')
        # af = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,damping=0.5, max_iter=200, preference=-100, verbose=False).fit(cpact1)
        af = SpectralClustering(n_clusters=6, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0, assign_labels="kmeans", degree=3, coef0=1, kernel_params=None).fit(cpact1)
        plog(set(af.labels_))
        print(af.labels_)
        d = pd.DataFrame(cpact2)
        link = hc.linkage(d.values,method='centroid')
        o1 = hc.leaves_list(link)
        mat = d.iloc[o1,:]
        mat = mat.iloc[:, o1[::-1]]
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(cpact1)
        axarr[0,1].imshow(mat)
        plt.show()


def kpiDis(scor,tLab="",saveF=None,col_cor="y_cor",col_dif="y_dif",col_sum="sum"):
    nRef = sum(~np.isnan(scor[col_sum]))
    nShare = sum(~np.isnan(scor[col_cor]))
    locShare = [(x/10) for x in range(11)]
    difShare = [np.sum(np.abs(scor[col_dif]) < x)/nRef for x in locShare]
    corShare = [np.sum(scor[col_cor] > x)/nRef for x in locShare]
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(locShare,difShare,label="difference",color="b")
    plt.fill_between(locShare[:3],0,difShare[:3],label="diff < 0.2",color="b",alpha=.5)
    plt.fill_between(locShare[:4],0,difShare[:4],label="diff < 0.3",color="b",alpha=.3)
    plt.plot(locShare,corShare,label="correlation",color="g")
    plt.fill_between(locShare[6:],0,corShare[6:],label="corr > 0.6",color="g",alpha=.5)
    plt.fill_between(locShare[5:],0,corShare[5:],label="corr > 0.5",color="g",alpha=.3)
    plt.ylabel("covered locations")
    plt.xlabel("relative value")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    #fig.patch.set_alpha(0.7)
    ax.patch.set_alpha(0.0)
    ax.grid(b=True, which='major', color='k', linestyle='--',linewidth=0.25,axis="both")
    #ax.minorticks_on()
#    ax.tick_params(color="k")
    plt.title(tLab+" - locations: " + str(nShare) + "/" + str(nRef))
    plt.legend()
    if saveF:
        plt.savefig(saveF)
    #    plt.grid()
    else:
        plt.show()

def binMask(typeV):
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

def featureImportance(X,y,tL,method=0):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from xgboost import XGBClassifier

    if method == 0:
        mod = ExtraTreesClassifier(n_estimators=250,random_state=0)
        mod.fit(X, y)
        importances = mod.feature_importances_
        std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)

    if method == 1:
        mod = SelectKBest(score_func=chi2, k=X.shape[1])
        fit = mod.fit(X, y)
        importances = fit.scores_
        std = np.apply_along_axis(np.std,0,fit.transform(X))

    if method == 2: #recursive feature elimination
        mod = LogisticRegression()
        rfe = RFE(mod, 3)
        fit = rfe.fit(X, y)
        importances = fit.ranking_
        std = .1

    if method == 3:
        mod = XGBClassifier()
        fit = mod.fit(X, y)
        importances = mod.feature_importances_
        std = .01#np.apply_along_axis(np.std,0,fit.transform(X))
        
    indices = np.argsort(importances)[::-1]
    impD = pd.DataFrame({"importance":importances,"std":std,"idx":np.argsort(importances)[::-1],"label":np.array(tL)[indices]})
    impD.sort_values("importance",inplace=True,ascending=False)
    impD = impD.reset_index()
    return impD


def mutualInformation(x,y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi
