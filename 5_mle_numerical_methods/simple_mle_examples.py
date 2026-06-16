#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:18:22 2026

@author: michael
"""


from scipy.optimize import minimize
import numpy as np
import pandas as pd
import statsmodels.api as sm
import gc

# simple example building a logistic regression
df = pd.read_csv('https://raw.githubusercontent.com/OpenSourceEcon/CompMethods/' +
      'main/data/basic_empirics/logit/titanic-train.csv')
df = pd.read_csv("/home/michael/Projects/regression_analysis/originally_forked_scripts/W10/mroz9.csv")
df.head()

# set outcome and covariates
y = "inlf"
X = ['nwifeinc', 'educ', 'exper', 'expersq', 'age', 'kidslt6']

gc.collect()

# linear regs
## using stats mod
print(sm.OLS(df[y],
             sm.add_constant(df[X]),
                data=df).fit().summary())

## define simple linear function
def linear_mod(b, y, x):
    return np.sum((y - x@b).T @ (y - x@b))
   # return np.sum((y - x@b) ** 2)

beta_0 = np.zeros(len(X) + 1) # initial guesses foe coefficients all at zero

minimize(fun = linear_mod, 
         x0 = beta_0,
         args=(df[y], 
               sm.add_constant(df[X])),
         method='BFGS'
         )


np.sqrt(minimize(fun = linear_mod, 
         x0 = beta_0,
         args=(df[y], 
               sm.add_constant(df[X])),
         method='BFGS'
         ).hess_inv[0,0])

# actually lo likelihoof function
def gausian_ll(params, x, y, n):
    # extract params
    b = params[:-1] 
    sd = params[-1]
    
    # resids
    r = y - x@b
    
    # resids^2
    r2 = np.dot(r.T, r)
    
    return (n / 2) * np.log(2 * np.pi * sd**2) + (r2 / (2 * sd**2))

# set up parameters and initial guesses
X_matrix = sm.add_constant(df[X])
n_params = X_matrix.shape[1] 

# set sigma to start its guess at 1
beta_0 = np.zeros(n_params + 1)
beta_0[-1] = 1.0

# make sure to set the constraint for sigma to ensure it's only positive (variance/sd can only be positive)
constraints = tuple((None, None) for _ in range(n_params)) + ((1e-20, None),)

output = minimize(fun=gausian_ll, 
                  x0=beta_0, 
                  args=(X_matrix, 
                        df[y], 
                        len(df)), 
                  method='L-BFGS-B', 
                  bounds=constraints)

print(output)




## using minimize


## log regs
# using stats model
# print(sm.GLM.from_formula('inlf ~ exper + expersq',
#                             data=df, 
#                             family=sm.families.Binomial()).fit().summary())

print(sm.GLM(df[y],
             sm.add_constant(df[X]),
                family=sm.families.Binomial()).fit(method = "bfgs").summary())


# using scipy

## define the log-liklihood function for log reg
def log_lik_logis_reg(beta, y, x):
    return -((-np.log(1 + np.exp(np.dot(x, beta)))).sum() + (y*(np.dot(x, beta))).sum())

beta_0 = np.zeros(len(X) + 1) # initial guesses foe coefficients all at zero

minimize(fun = log_lik_logis_reg, 
         x0 = beta_0,
         args=(df[y], 
               sm.add_constant(df[X])),
         method='BFGS'
         )


# using stochastic gradient descent
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

LinearRegression().fit(X = df[X], y = df[y]).coef_

SGDRegressor(max_iter=10000, 
             tol=1e-3, 
             loss="squared_error",  
             penalty=None, 
             fit_intercept=True).fit(X = StandardScaler().fit_transform(df[X]), 
                                     y = df[y]).coef_


SGDClassifier(max_iter=1000, tol=1e-3).fit(X = StandardScaler().fit_transform(df[X]), 
                        y = df[y]).coef_

