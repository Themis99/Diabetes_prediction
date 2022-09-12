import pandas as pd
from sklearn.datasets import*
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

def data_preparation():

    #load dataset
    diabetes = load_diabetes()
    features = pd.DataFrame(diabetes.data)
    Targets = pd.Series(diabetes.target,name='target')
    dataset = pd.concat([features,Targets],axis=1)

    #train-test split
    train_data = dataset.iloc[0:300,:]
    test_data = dataset.iloc[300:len(dataset),:]

    #min-max scaling
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train = train_scaled[:, :-1]
    y_train = train_scaled[:,-1]

    X_test = test_scaled[:, :-1]
    y_test = test_scaled[:,-1]

    return X_train, X_test, y_train, y_test

def Support_vector_machine(X_train,X_test,y_train,y_test):

    C = [0.001,0.01,0.1,1, 10, 100]
    Gamma = [0.001,0.01,0.1,1, 10, 100]

    kernel = ['rbf', 'linear', 'sigmoid', 'poly']

    for i in range(-15, 3):
        Gamma.append(2 ** i)

    param_dist = dict(kernel=kernel,gamma=Gamma, C=C)
    scoring = ['neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error','r2']

    #model training
    model = SVR()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error' ,random_state=5,n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

def Desicion_tree(X_train,X_test,y_train,y_test):

    maxdepth = [10,100,1000,10000]
    min_s_s = [5,10,100]
    min_s_l=[1,10,30,50]
    max_f = ['auto', 'sqrt', 'log2']
    spl = ['best', 'random']

    param_dist = dict(splitter=spl, max_depth=maxdepth, min_samples_split=min_s_s,
                       min_samples_leaf=min_s_l, max_features=max_f)
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

    model = DecisionTreeRegressor()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error',random_state=5, n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

def Random_forest(X_train,X_test,y_train,y_test):

    num_trees = [10,30,50,70,100]
    maxdepth = [10,100,1000,10000]
    min_s_s = [5,10,100]
    min_s_l=[1,10,30,50]
    max_f = ['auto', 'sqrt', 'log2']


    param_dist = dict(n_estimators=num_trees, max_depth=maxdepth, min_samples_split=min_s_s,
                       min_samples_leaf=min_s_l, max_features=max_f)
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

    model = RandomForestRegressor()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error',random_state=5, n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

def ADAboost(X_train,X_test,y_train,y_test):

    estim = [50, 60, 70, 80, 90, 100]
    L = ['linear', 'square', 'exponential']


    param_dist=dict(n_estimators=estim,loss=L)
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

    model = AdaBoostRegressor()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error',random_state=5, n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

def Gradientboost(X_train,X_test,y_train,y_test):
    num_est = [10,30,50,70,100]
    maxdepth = [10,100,1000,10000]
    min_s_s = [5,10,100]
    min_s_l=[1,10,30,50]
    loss = ['ls', 'lad', 'huber', 'quantile']
    lr = [0.000001,0.00001,0.0001,0.001,0.01,0.1]


    param_dist=dict(n_estimators=num_est,max_depth=maxdepth,min_samples_split=min_s_s,
                       min_samples_leaf=min_s_l,learning_rate=lr,loss=loss)

    param_dist = dict(n_estimators=num_est, max_depth=maxdepth, min_samples_split=min_s_s,
                      min_samples_leaf=min_s_l, learning_rate=lr, loss=loss)

    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

    model = GradientBoostingRegressor()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error',random_state=5, n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

def xgboost(X_train,X_test,y_train,y_test):
    learning_rate=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    max_depth=[3, 4, 5, 6, 8, 10, 12, 15]
    min_child_weight=[1, 3, 5, 7]
    gamma=[0.0, 0.1, 0.2, 0.3, 0.4]
    colsample_bytree=[0.3, 0.4, 0.5, 0.7]

    param_dist = dict(learning_rate=learning_rate,max_depth=max_depth,min_child_weight=min_child_weight,gamma=gamma,colsample_bytree=colsample_bytree)

    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']

    model = XGBRegressor()
    grid_search = RandomizedSearchCV(model, param_dist, n_iter=5, scoring=scoring, cv=10,refit='neg_mean_absolute_error',random_state=5, n_jobs = -1)
    grid_search.fit(X_train,y_train)

    ypred = grid_search.predict(X_test)

    print('Best parameters:',grid_search.best_params_)
    print('Mean absolute error:',mean_absolute_error(y_test, ypred))
    print('Mean squared error:',mean_squared_error(y_test, ypred))
    print('Root mean squared error:',mean_squared_error(y_test, ypred,squared=False))
    print('R2:',r2_score(y_test, ypred))

if __name__=="__main__":
    X_train, X_test, y_train, y_test = data_preparation()

    #Support vector machine
    print('Results for support vector machine:','\n')
    Support_vector_machine(X_train, X_test, y_train, y_test)
    print('\n')

    #Decision tree
    print('Results for decision tree:', '\n')
    Desicion_tree(X_train, X_test, y_train, y_test)
    print('\n')

    #Random forest
    print('Results for random forest:', '\n')
    Random_forest(X_train, X_test, y_train, y_test)
    print('\n')

    #ADAboost
    print('Results for ADA boost:', '\n')
    ADAboost(X_train, X_test, y_train, y_test)
    print('\n')

    #Gradient Boost
    print('Results for Grandient boost:', '\n')
    Gradientboost(X_train, X_test, y_train, y_test)
    print('\n')

    #xgboost
    print('Results for xg boost:', '\n')
    xgboost(X_train, X_test, y_train, y_test)
    print('\n')