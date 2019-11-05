#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:03:14 2019

@author: Adam
"""
from pandas import read_csv

data = read_csv('/Users/Adam/sales.csv',delimiter=',',)
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
df = pd.DataFrame(data)
corrmatrix = df.corr()

#scale price data
data.price = data.price/1000

#split dependent data into training/testing sets
y_train = data.price[:8000]
y_test = data.price[8000:]

model = smf.ols('price ~ home_size + x_coord + y_coord + age', data=data[:8000])
results = model.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse = sum((y_train - pred_train)**2)/results.nobs
test_mse = sum((y_test - pred_test)**2)/len(y_test)

model1 = smf.ols('price ~ home_size + np.power(home_size, 2) + np.power(home_size, 3) + np.power(home_size, 4) + cbd_dist + np.power(cbd_dist, 2) + np.power(cbd_dist, 3) + beds + np.power(beds, 2) + np.power(beds, 3) + np.power(beds, 4) + parcel_size + np.power(parcel_size, 2) + np.power(parcel_size, 3) + np.power(parcel_size, 4)', data=(data[:8000]))
results = model1.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse1 = sum((y_train - pred_train)**2)/results.nobs
test_mse1 = sum((y_test - pred_test)**2)/len(y_test)

model2 = smf.ols('price ~ beds + home_size + cbd_dist + year + parcel_size + x_coord + y_coord + age + np.power(age, 2)', data=(data[:8000]))
results = model2.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse2 = sum((y_train - pred_train)**2)/results.nobs
test_mse2 = sum((y_test - pred_test)**2)/len(y_test)

model3 = smf.ols('price ~ home_size + np.multiply(home_size, beds) + cbd_dist + beds + np.multiply(beds, parcel_size) + parcel_size + np.multiply(parcel_size, home_size)', data=(data[:8000]))
results = model3.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse3 = sum((y_train - pred_train)**2)/results.nobs
test_mse3 = sum((y_test - pred_test)**2)/len(y_test)

model4 = smf.ols('price ~ home_size + np.power(home_size, 2) + np.power(home_size, 3) + np.power(home_size, 4) + cbd_dist + beds + np.power(beds, 2) + np.power(beds, 3) + np.power(beds, 4) + parcel_size + np.power(parcel_size, 2) + np.power(parcel_size, 3) + np.power(parcel_size, 4)', data=(data[:8000]))
results = model4.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse4 = sum((y_train - pred_train)**2)/results.nobs
test_mse4 = sum((y_test - pred_test)**2)/len(y_test)

model5 = smf.ols('price ~ home_size + np.power(home_size, 2) + np.power(home_size, 3) + np.power(home_size, 4) + beds + np.power(beds, 2) + np.power(beds, 3) + np.power(beds, 4) + cbd_dist + np.multiply(cbd_dist, home_size)', data=(data[:8000]))
results = model5.fit()
print(results.summary())
print(results.summary())
pred_train = results.predict(data[:8000])
pred_test  = results.predict(data[8000:])

train_mse5 = sum((y_train - pred_train)**2)/results.nobs
test_mse5 = sum((y_test - pred_test)**2)/len(y_test)
