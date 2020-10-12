import pandas as pd
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, validation_curve,cross_val_score,KFold
from sklearn.linear_model import Lasso,LassoCV,LinearRegression,RidgeCV
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import sqrt

def ols_model(X,y):
    '''
    Scale and print results summary
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    X_train_scale = X_train.copy()

    scale = StandardScaler().fit(X_train_scale)

    X_train_scale = scale.transform(X_train_scale)
    model = sm.OLS(y_train, sm.add_constant(X_train_scale))
    results = model.fit()

    return results.summary()

def train_and_test_linear(X,y):
    '''
    Scale data and perform a linear regression on it and cross validation on it
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)

    X_train_scale = X_train.values
    X_test_scale = X_test.values

    scale = StandardScaler()

    X_train_scaled = scale.fit_transform(X_train_scale)
    X_test_scaled = scale.transform(X_test_scale)

    lm = LinearRegression()
    lm.fit(X_train_scale,y_train)
    y_pred = lm.predict(X_test_scale)

    print(f'Linear Regression val R^2: {lm.score(X_train_scale, y_train):.3f}')
    print(f'Linear Regression val RME: {sqrt(mean_squared_error(y_test,y_pred)):.3f}')
    #return y_pred

def scale_test_and_train_Lasso(X,y):
    """
    Run a ridge regression on the model
    """
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=3)

    X_train_scale = X_train.values
    X_val_scale = X_val.values
    X_test_scale = X_test.values

    scale = StandardScaler()

    X_train_scale = scale.fit_transform(X_train_scale)
    X_test_scale = scale.transform(X_test_scale)
    X_val_scale = scale.transform(X_val_scale)

    lasso = LassoCV()
    lasso.fit(X_train_scale,y_train)

    lasso.score(X_val_scale,y_val)

    y_pred = lasso.predict(X_val_scale)


    print(f'Lasso Regression val R^2: {lasso.score(X_val_scale, y_val):.3f}')
    print(f'Lasso Regression val RME: {sqrt(mean_squared_error(y_val,y_pred)):.3f}')

    return lasso.coef_

def scale_test_and_train_ridge(X,y):
    """
    Run a ridge regression on the model
    """
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=3)

    X_train_scale = X_train.values
    X_val_scale = X_val.values
    X_test_scale = X_test.values

    scale = StandardScaler()

    X_train_scale = scale.fit_transform(X_train_scale)
    X_test_scale = scale.transform(X_test_scale)
    X_val_scale = scale.transform(X_val_scale)

    ridge = RidgeCV(cv=5)
    ridge.fit(X_train_scale,y_train)

    ridge.score(X_train_scale,y_train)

    y_pred = ridge.predict(X_val_scale)


    print(f'Ridge Regression val R^2: {ridge.score(X_val_scale, y_val):.3f}')
    print(f'Ridge Regression val RME: {sqrt(mean_squared_error(y_val,y_pred)):.3f}')

    return ridge.coef_


def cross_val_linear(X,y):
    '''
    Scale data and perform a linear regression on it and cross validation on it
    '''
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state = 71)
    r2_scores, rme_scores = [], [] #collect the validation results for both models

    for train_ind, val_ind in kf.split(X,y):

        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[val_ind], y.iloc[val_ind]

        scale = StandardScaler()
        X_train_scale = scale.fit_transform(X_train)
        X_val_scale = scale.transform(X_val)


        lm = LinearRegression()



        lm.fit(X_train_scale, y_train)

        y_pred = lm.predict(X_val_scale)
        r2_scores.append(lm.score(X_val_scale, y_val))
        rme_scores.append(sqrt(mean_squared_error(y_val,y_pred)))


    print('Scaled regression scores: ', r2_scores)
    print('Scaled regression RME scores: ',rme_scores)
    print(f'Scaled mean cv r^2: {np.mean(r2_scores):.3f} ')
    print(f'Scaled mean cv r^2: {np.mean(rme_scores):.3f}')
