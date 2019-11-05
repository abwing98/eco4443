#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:03:14 2019

@author: Adam
"""
from pandas import read_csv

data = read_csv('/Users/Adam/sales.csv',delimiter=',',)

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from statsmodels.api import add_constant
from pandas import DataFrame

price = DataFrame(data.price)
year = DataFrame(data.year)
home_size = DataFrame(data.home_size)
parcel_size = DataFrame(data.parcel_size)
beds = DataFrame(data.beds)
age = DataFrame(data.age)
pool = DataFrame(data.pool)
cbd_dist = DataFrame(data.cbd_dist)
x_coord = DataFrame(data.x_coord)
y_coord = DataFrame(data.y_coord)

import pandas as pd
df = pd.DataFrame(data)
corrmatrix = df.corr()

#scale price data
data.price = data.price/1000

#split dependent data into training/testing sets
y_train = data.price[:8000]
y_test = data.price[8000:]

model = smf.ols('price ~ home_size + x_coord + y_coord + age', data=data.price[:8000])
results = model.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse1 = sum((y_train - pred_train)**2)/results.nobs
test_mse1 = sum((y_test - pred_test)**2)/len(y_test)

model1 = smf.ols('price ~ home_size + np.power(home_size, 2) + np.power(home_size, 3) + cbd_dist + np.power(cbd_dist, 2) + np.power(cbd_dist, 3)', data=(data.price[:8000]))
results = model1.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse1 = sum((y_train - pred_train)**2)/results.nobs
test_mse1 = sum((y_test - pred_test)**2)/len(y_test)

model2 = smf.ols('price ~ beds + home_size + cbd_dist + year + parcel_size + x_coord + y_coord + age + np.power(age, 2)', data=(data.price[:8000]))
results = model2.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse1 = sum((y_train - pred_train)**2)/results.nobs
test_mse1 = sum((y_test - pred_test)**2)/len(y_test)






