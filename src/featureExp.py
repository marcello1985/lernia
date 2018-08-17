#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as t_l
import geomadi.series_lib as s_l
import geomadi.train_lib as tlib
import geopandas as gpd
import etl.etl_mapping as e_m
import seaborn as sns
import geomadi.train_filter as t_f

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats, optimize
from sklearn.datasets import load_diabetes
from sklearn.cross_validation import train_test_split
from theano import shared
np.random.seed(9)


dateL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
dateL = dateL.drop(columns=["wday","use","visibility","pressureError","lightDur","precipProbability"])
X = dateL[['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax']].values
y, _= pd.factorize(dateL['icon'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.25, random_state=0)
X.shape, y_.shape, X_tr.shape, X_te.shape
shA_X = shared(X_tr)
linear_model = pm.Model()
with linear_model: 
    alpha = pm.Normal("alpha", mu=y_tr.mean(),sd=10)
    betas = pm.Normal("betas", mu=0, sd=1000, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", sd=100) 
    mu = alpha + np.array([betas[j]*shA_X[:,j] for j in range(X.shape[1])]).sum()
#    mu = alpha + pm.dot(betas, X_tr.T)
    likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_tr)
#    map_estimate = pm.find_MAP(model=linear_model, fmin=optimize.fmin_powell)
    step = pm.NUTS()
    trace = pm.sample(1000, step)

chain = trace[100:]
pm.traceplot(chain);
plt.show()

sns.kdeplot(y_tr, alpha=0.5, lw=4, c='b')
for i in range(100):
    sns.kdeplot(ppc['likelihood'][i], alpha=0.1, c='g')
plt.show()


    
correlation_matrix = dateL.corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True,annot=True,cmap='RdYlGn')
plt.title('Correlation matrix between the features')
plt.show()

sns.pairplot(dateL,hue='icon',diag_kind='kde',plot_kws={'alpha':0.6,'s':80,'edgecolor':'k'},size=4,vars=['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax'])
plt.show()

sns.pairplot(dateL,hue='icon',diag_kind='kde',kind="kdeplot",vars=['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax'],markers="+",plot_kws=dict(s=50,edgecolor="b",linewidth=1),diag_kws=dict(shade=True))
plt.show()


g = sns.PairGrid(dateL,hue='icon',vars=['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax'])
g.map_upper(sns.regplot) 
g.map_lower(sns.residplot) 
g.map_diag(plt.hist) 
plt.show()


g = sns.PairGrid(dateL,vars=['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax'])
g.map_upper(sns.regplot) 
g.map_lower(sns.kdeplot) 
g.map_diag(plt.hist) 
plt.show()



def corr(x, y, **kwargs):
    coef = np.corrcoef(x, y)[0][1]
    label = r'$\rho$ = ' + str(round(coef, 2))
    label = str(round(coef, 2))
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.8), size = 20, xycoords = ax.transAxes)

grid = sns.PairGrid(data=dateL,vars=['cloudCover','precipIntensity','pressure','windSpeed','Tmin','Tmax'],size=4)#,hue="icon")
grid = grid.map_upper(plt.scatter,color='darkred')
grid = grid.map_upper(corr)
grid = grid.map_lower(sns.kdeplot,cmap='Reds')
grid = grid.map_diag(plt.hist,bins=10,edgecolor='k',color='darkred');
plt.show()


grid = grid.map_upper(plt.scatter,color='darkred')
grid = grid.map_diag(plt.hist,bins=10,color='darkred',edgecolor = 'k')
grid = grid.map_lower(sns.kdeplot,cmap='Reds')
plt.show()

    
grid = sns.PairGrid(data= df[df['year'] == 2007], vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], size = 4)

