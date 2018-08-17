import sklearn as sk
import sklearn.ensemble
import sklearn.tree
import sklearn.neural_network
import sklearn.svm
import sklearn.gaussian_process
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.discriminant_analysis
import sklearn.dummy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


tuneL = {'loss': 'deviance', 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
tuneL = {'loss': 'deviance', 'max_features': 'sqrt', 'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 50}
tuneL = {'loss': 'deviance', 'max_features': 'auto', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200}
tuneL = {'loss': 'deviance', 'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.1}
catCla = [
    ##random forest
    sk.ensemble.RandomForestClassifier(n_estimators=tuneL['n_estimators'],criterion='entropy',max_features=tuneL['max_features'],max_depth=5,bootstrap=True,oob_score=True,n_jobs=12,random_state=33)
    ##random forest 2
    ,sk.ensemble.RandomForestClassifier(n_estimators=tuneL['n_estimators'],criterion='gini',n_jobs=12,max_depth=15,max_features=tuneL['max_features'],min_samples_split=2,random_state=None)
    ##decision tree
    ,sk.tree.DecisionTreeClassifier(criterion="gini",random_state=tuneL['n_estimators'],max_depth=10,min_samples_leaf=5)
    ##extra tree
    ,sk.ensemble.ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=1e-07, min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=250,n_jobs=1,oob_score=False,random_state=0,verbose=0,warm_start=False)
    ##neural network
    ,sk.neural_network.MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,beta_2=0.999, early_stopping=False, epsilon=1e-08,hidden_layer_sizes=(tuneL['n_estimators'],), learning_rate='constant',learning_rate_init=0.001, max_iter=200, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=None,shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,verbose=False, warm_start=False)
    ##k-neighbors
    ,sk.neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=3, p=2,weights='uniform')
]

binCla = [
    ##dummy variables
    #sk.dummy.DummyClassifier(strategy='stratified',random_state=10)
    ##gradient boost
    sk.ensemble.GradientBoostingClassifier(criterion='friedman_mse',init=None,learning_rate=tuneL['learning_rate'], loss=tuneL['loss'], max_depth=tuneL['max_depth'],max_features=tuneL['max_features'], max_leaf_nodes=None,min_impurity_decrease=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=tuneL['n_estimators'], presort='auto', random_state=10,subsample=1.0, verbose=0, warm_start=False)
    ##support vector
    ,sk.svm.SVC(C=1.0,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape=None,degree=3,gamma='auto',kernel='rbf',max_iter=-1,probability=True,random_state=0,shrinking=True,tol=0.001,verbose=False)
    ,sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
    ##logistic regression
    ,sk.linear_model.LogisticRegression(C=100.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,max_iter=tuneL['n_estimators'], multi_class='ovr',n_jobs=12,penalty='l2',random_state=None,solver='liblinear',tol=0.0001,verbose=0,warm_start=False)
    ]

othCla = [
    ##support vector
    sk.svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    ##gaussian process
    ,sk.gaussian_process.GaussianProcessClassifier(copy_X_train=True,kernel=1**2 * RBF(length_scale=1), max_iter_predict=tuneL['n_estimators'],multi_class='one_vs_rest', n_jobs=1, n_restarts_optimizer=0,optimizer='fmin_l_bfgs_b', random_state=None, warm_start=True)
    ##naive bayesias
    ,sk.naive_bayes.GaussianNB(priors=None)
    ##quadratic discriminant
    ,sk.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,store_covariances=False, tol=0.0001)
    ,sk.linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    ##ada boost
    ,sk.ensemble.AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,learning_rate=1.0, n_estimators=50, random_state=None)
    ]


def nCat():
    return len(catCla)

def retCat(n):
    return catCla[n]

def nBin():
    return len(binCla)

def retBin(n):
    return binCla[n]
