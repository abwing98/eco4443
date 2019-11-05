
from pandas import read_csv
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from statsmodels.api import add_constant
from pandas import DataFrame
import pandas as pd
from sklearn import model_selection
from sklearn import datasets
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)

#MODEL 1
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#pandas dataframe
df = pd.DataFrame(data)
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
#score the model
print("Score:", model.score(x_test,y_test))
#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)

results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))


#MODEL 2
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create exponential variables
hsize2 = np.power(data.home_size,2)
hsize3 = np.power(data.home_size,3)
hsize4 = np.power(data.home_size,4)
beds2 = np.power(data.beds,2)
beds3 = np.power(data.beds,3)
beds4 = np.power(data.beds,4)
cbd_hsize = np.multiply(data.cbd_dist, data.home_size)
cbd_beds = np.multiply(data.cbd_dist, data.beds) 
beds_hsize = np.multiply(data.beds, data.home_size)
#create new dataframe with our models variables: home size^1,2,3,4; beds^1,2,3,4; cbd_dist*home_size; cbd_dist*beds; beds*home_size
new = [data.home_size, hsize2, hsize3, hsize4, data.beds, beds2, beds3, beds4, data.cbd_dist, cbd_hsize, cbd_beds, beds_hsize]
df = pd.DataFrame(new)
df = np.transpose(df)

#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
#score the model
print("Score:", model.score(x_test,y_test))
#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)

results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))
