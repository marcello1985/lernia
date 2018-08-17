#%pylab inline
##http://scikit-learn.org/stable/modules/ensemble.html
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from collections import defaultdict
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import interpolate
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import cophenet
from scipy.misc import imread
from scipy.optimize import leastsq as least_squares
from scipy.spatial.distance import pdist
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeRegressor
import geomadi.train_filter as t_f
from sklearn.ensemble import ExtraTreesClassifier
#import h5py
import keras
import pylab
import random
import scipy, scipy.stats
import sklearn as sk
import sklearn.discriminant_analysis
import sklearn.dummy
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.metrics as skm
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import statsmodels.formula.api as sm
import tensorflow as tf

def plog(text):
    print(text)

def binMatrix(X,nBin=6,threshold=2.5):
    """bin a continuum parametric matrix"""
    X = np.array(X).astype(float)
    t_M, psum_x = t_f.binMatrix(X,nBin)
    return t_M, psum_x

def binVector(y,nBin=6,threshold=2.5):
    """bin a continuum parametric vector"""
    y, psum = t_f.binOutlier(y,nBin=nBin,threshold=threshold)
    y = np.array(y).astype(int)
    return y, psum

def plotHist(y,nBin=7,threshold=2.5,lab=""):
    """plot histogram and its binning"""
    y = y[~np.isnan(y)]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, 10))
    mHist ,xHist = np.histogram(y,bins=(nBin+1))
    y1, psum = t_f.binOutlier(y,nBin=nBin,threshold=threshold)
    mHist1 = np.bincount(y1)
    plt.bar(psum[1:],mHist1,(max(psum)-min(psum))/float(len(psum)),label="bin_"+lab,fill=False,alpha=0.5,color=colors[0])
    plt.plot(psum[1:],mHist,label=lab)
    plt.legend()
    plt.show()

class trainMod:
    """routine to find the best performing model"""
    def __init__(self,X,y):
        """load matrices and split test and train sets"""
        if X.shape[0] != y.shape[0]:
            raise ValueError("wrong dimensions: X:%d,y:%d" % (X.shape[0],y.shape[0]) )
        X = self.factorize(X)
        self.X = np.array(X).astype(float)
        self.y = np.array(y).astype(int)
        #X = StandardScaler().fit_transform(X) #X = preprocessing.scale(X)

    def factorize(self,X):
        for i in X.select_dtypes(object):
            X.loc[:,i], _ = pd.factorize(X[i])
        X = X.replace(float("Nan"),0)
        return X
            
    def setMatrix(self,X):
        """set a new matrix"""
        X = self.factorize(X)
        self.X = np.array(X).astype(float)
        
    def setScore(self,y):
        """set a new score"""
        self.y = np.array(y).astype(int)

    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def perfCla(self,clf):
        """perform a single classification"""
        t_start = time.clock()
        if clf['type'] == "class" :
            y_train1 = label_binarize(self.y_train, classes=np.unique(self.y_train))
            y_test1  = label_binarize(self.y_test, classes=np.unique(self.y_test))
        elif clf['type'] == "logit" :
            y_train1 = self.y_train.ravel()
            y_test1  = self.y_test.ravel()
        mod = clf['mod'].fit(self.X_train,y_train1)
        y_score = mod.predict_proba(self.X_test)
        if clf['score'] == "ravel" :
            y_score = y_score.ravel()
            y_score1 = label_binarize(self.y_test, classes=np.unique(self.y_test))
            y_score1 = y_score1.ravel()
        else:
            y_score = np.hstack([y_score[x][:,1] for x in range(len(np.unique(self.y_test)))])
            y_score1 = y_test1.ravel()
        x_pr, y_pr, _ = skm.roc_curve(y_score1,y_score)
        print(clf)
        print(self.X_train.shape)
        print(self.X_test.shape)
        print(y_train1.shape)
        print(y_test1.shape)
        train_score = mod.score(self.X_train,y_train1)
        test_score  = mod.score(self.X_test ,y_test1 )
        #cv = cross_validate(mod,self.X,self.y,scoring=['precision_macro','recall_macro'],cv=5,return_train_score=True)
        cv = "off"
        fsc = skm.f1_score(y_test1,mod.predict(self.X_test),average="weighted")
        acc = skm.accuracy_score(y_test1,mod.predict(self.X_test))
        y_predict = mod.predict(self.X_test) == y_test1
        auc = skm.auc(x_pr,y_pr)## = np.trapz(fpr,tpr)
        t_end = time.clock()
        t_diff = t_end - t_start
        return mod, train_score, test_score, t_diff, x_pr, y_pr, auc, fsc, acc, cv
    
    def loopMod(self,paramF="train.json",test_size=0.4):
        """loop over all avaiable models"""
        self.X_train,self.X_test,self.y_train,self.y_test = sk.model_selection.train_test_split(self.X, self.y,test_size=test_size,random_state=0)
        trainR = []
        model = []
        rocC = []
        tml = modelList(paramF)
        tml.set_params()
        for index in range(tml.nCat()):
            clf = tml.retCat(index)
            if not clf['active']:
                continue
            mod, trainS, testS, t_diff, x_pr, y_pr, auc, fsc, acc, cv = self.perfCla(clf)
            trainR.append([clf['name'],trainS,testS,t_diff,auc,fsc,acc,clf["type"]])
            model.append(mod)
            rocC.append([x_pr,y_pr])
            #print("{m} trained {c} in {f:.2f} s".format(m=modN,c=index, f=t_diff))
        trainR = pd.DataFrame(trainR)
        trainR.columns = ["model","train_score","test_score","time","auc","fsc","acc","type"]
        trainR.loc[:,'perf'] = trainR['acc']*trainR['auc']
        trainR = trainR.sort_values(['perf'],ascending=False)
        mod = model[trainR.index.values[0]]
        self.rocC = rocC
        self.trainR = trainR
        y_pred = mod.predict(self.X)
        try:
            y_class = y_pred.dot(range(y_pred.shape[1]))
        except IndexError:
            y_class = y_pred
        self.y_pred = y_pred
        return mod, trainR#, self.y, y_class
 
    def plotRoc(self):
        """plot roc curve"""
        if not hasattr(self,"rocC"):
            print("first train the models .loopMod()")
            return 
        plt.clf()
        plt.plot([0, 1],[0, 1],'k--',label="model | auc  f1   acc")
        for idx, mod in self.trainR.iterrows():
            plt.plot(self.rocC[idx][0],self.rocC[idx][1],label='%s | %0.2f %0.2f %0.2f ' %
                     (mod['model'],mod['auc'],mod['fsc'],mod['acc']))
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right",prop={'size':12})#,'family':'monospace'})
        #plt.savefig(baseDir+'/fig/modelPerformaces.jpg')
        plt.show()

    def plotConfMat(self):
        """plot confusion matrix"""
        cm = confusion_matrix(self.y,self.y_pred)
        print(cm)
        plt.xlabel("prediction")
        plt.ylabel("score")
        plt.imshow(cm)
        plt.show()

    def save(self,mod,fName):
       joblib.dump(mod,fName) 

    def load(self,fName):
        clf = joblib.load(fName)
        return clf

    def tune(self,paramF="train.json",tuneF="train_tune.json"):
        """tune all avaiable models"""
        tml = modelList(paramF)
        params = tml.get_params()
        with open(tuneF) as f:
            pgrid = json.load(f)
        for idx in range(len(pgrid)):
            if not pgrid[idx]['active']:
                continue
            print("tuning: " + pgrid[idx]['name'])
            clf = tml.retCat(idx)['mod']
            CV_rfc = GridSearchCV(estimator=clf,param_grid=pgrid[idx]['param_grid'],cv=5,return_train_score=False)
            CV_rfc.fit(self.X, self.y)
            for k,v in CV_rfc.best_params_.items():
                params[idx][k] = v

        with open(paramF,'w') as f:
            f.write(json.dumps(params))


    def tuneKeras():
        seed = 7
        numpy.random.seed(seed)
        model = KerasClassifier(build_fn=create_model, verbose=0)
        param_grid ={"batch_size":[10, 20, 40, 60, 80, 100],"epochs":[10, 50, 100]}
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(X, Y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

def modKeras(x_train,y_train):
    Nneu = x_train.shape[1]
    Nent = y_train.shape[0]
    try:
        Ncat = y_train.shape[1]
    except:
        Ncat = 1
    model = Sequential()
    model.add(Dense(input_dim=Nneu,output_dim=Nneu,activation='relu'))#,init="uniform"))
    keras.layers.core.Dropout(rate=0.15)
    model.add(Dense(input_dim=Nneu,output_dim=Nneu,activation='relu'))#,init="uniform"))
    keras.layers.core.Dropout(rate=0.15)
    model.add(Dense(input_dim=Nneu,output_dim=Ncat,activation='sigmoid'))#,init="uniform"))
    #model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    sgd = keras.optimizers.SGD(lr=0.001,decay=1e-7,momentum=.9)
    adam = keras.optimizers.adam(lr=0.01,decay=1e-5)
    #model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    model.get_params = model.get_config()
    return model

class modelList():
    """list of sklearn models to iterate"""
    catCla = [
        {"active":True,"name":"random_forest"  ,"type":"class","score":"stack","mod":sk.ensemble.RandomForestClassifier()}
        ,{"active":True,"name":"decision_tree" ,"type":"class","score":"stack","mod":sk.tree.DecisionTreeClassifier()}
        ,{"active":True,"name":"extra_tree"    ,"type":"class","score":"stack","mod":sk.ensemble.ExtraTreesClassifier()}
        ,{"active":True,"name":"perceptron"    ,"type":"class","score":"ravel","mod":sk.neural_network.MLPClassifier()}
        ,{"active":True,"name":"k_neighbors"   ,"type":"class","score":"stack","mod":sk.neighbors.KNeighborsClassifier()}
        ,{"active":True,"name":"grad_boost"    ,"type":"logit","score":"ravel","mod":sk.ensemble.GradientBoostingClassifier()}
        ,{"active":False,"name":"support_vector","type":"logit","score":"ravel","mod":sk.svm.SVC()}
        ,{"active":True,"name":"discriminant"  ,"type":"logit","score":"ravel","mod":sk.discriminant_analysis.LinearDiscriminantAnalysis()}
        ,{"active":True,"name":"logit_reg"     ,"type":"logit","score":"ravel","mod":sk.linear_model.LogisticRegression()}
        #,sk.dummy.DummyClassifier(strategy='stratified',random_state=10)
#        ,{"active":True,"name":"keras"         ,"type":"logit","score":"ravel","mod":modKeras()}
    ]

    othCla = [
        sk.svm.SVR()
        ,sk.gaussian_process.GaussianProcessClassifier()
        ,sk.naive_bayes.GaussianNB()
        ,sk.discriminant_analysis.QuadraticDiscriminantAnalysis()
        ,sk.linear_model.LinearRegression()
        ,sk.ensemble.AdaBoostClassifier()
    ]

    def __init__(self,paramF="train.json"):
        self.paramF = paramF
        return

    def nCat(self):
        return len(modelList.catCla)

    def retCat(self,n):
        return modelList.catCla[n]

    def get_params(self):
        params = []
        for mod in modelList.catCla:
            params.append(mod['mod'].get_params())
        with open(self.paramF,'w') as f:
            f.write(json.dumps(params))
        return params

    def set_params(self):
        with open(self.paramF) as f:
            params = json.load(f)
        for i,clf in enumerate(modelList.catCla):
            mod = clf['mod']
            mod.set_params(**params[i])
        return params

    def get_model_name(self):
        return [x['name'] for x in modelList.catCla]
    


def plotFeatCorr(t_M):
    corMat = t_M.corr()
    corrs = corMat.sum(axis=0)
    corr_order = corrs.argsort()[::-1]
    corMat = corMat.loc[corr_order.index,corr_order.index]
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.title('Correlation matrix between the features')
    plt.show()

def plotFeatCorrScatter(t_M):
    pd.plotting.scatter_matrix(t_M, diagonal="kde")
    plt.tight_layout()
    plt.show()
    

def plotMatrixCorr(X1,X2):
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

def plotBoxFeat(c_M,y):
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

def regressor(X,vf,vg):
    from sklearn.ensemble import BaggingRegressor
    # clf = linear_model.Lasso(alpha=0.1)
    # clf = linear_model.LinearRegression()
    # clf = DecisionTreeRegressor()
    clf = BaggingRegressor(DecisionTreeRegressor())
    d_quot = vg/vf
    d_quot[d_quot!=d_quot] = 1.
    d_quot[d_quot==float('Inf')] = 1.
    fit_q = clf.fit(X,d_quot.values)
    r_quot = fit_q.predict(X)
    if False:
        d_taylor = (vf - vg)/vf
        fit_t = clf.fit(X,d_taylor.values)
        r_taylor = clf.fit(X,d_taylor.values).predict(X)
        r_tayl = (1.-r_taylor)
        return r_tayl, fit_t
    return r_quot, fit_q

def saveModel(fit,fName):
    joblib.dump(fit,fName)

def crossVal(clf,X,y,cv=5):
    # X = ["a", "b", "c", "d"]
    # kf = KFold(n_splits=2)
    # for i in range(cv):
    #     X_train,X_test,y_train,y_test = sk.model_selection.train_test_split(X,y,test_size=test_size,random_state=0)
    scores = cross_val_score(clf, X, y, cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores

def featureImportance(X,y):
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
    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),xy=(.1, .6), xycoords=ax.transAxes,size = 24)

    cmap = sns.cubehelix_palette(light=1, dark = 0.1, hue = 0.5, as_cmap=True)
    sns.set_context(font_scale=2)
    gp = sns.PairGrid(X)
    gp.map_upper(plt.scatter, s=10, color = 'red')
    gp.map_diag(sns.distplot, kde=False, color = 'red')
    gp.map_lower(sns.kdeplot, cmap = cmap)
    #gp.map_lower(corrfunc);
    plt.show()

def featureRelevanceTree(X,y,isPlot=False):
    clf = ExtraTreesClassifier()
    clf.fit(X,y)
    featL = clf.feature_importances_
    sortL = sorted(featL)
    sortI = [sortL.index(x) for x in featL]
    if isPlot:
        plt.plot(sorted(featL))
        plt.show()
    return featL, sortL
    
def splitLearningSet(X,y,f_train=0.80,f_valid=0):
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
