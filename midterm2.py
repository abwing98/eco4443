
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
from sklearn.metrics import mean_squared_error


#MODEL 1
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
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
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 1 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 1 (Cross-Validated)")
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
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 2 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 2 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)


#MODEL 3
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create new dataframe with our models variables: home size, age, x coordinate, y coordinate
new = [data.home_size, data.age, data.x_coord, data.y_coord]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 3 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 3 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)
results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))

#MODEL 4
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create exponential variables
hsize2 = np.power(data.home_size,2)
hsize3 = np.power(data.home_size,3)
cbd2 = np.power(data.cbd_dist,2)
cbd3 = np.power(data.cbd_dist,3)
#create new dataframe with our models variables: home size^1,2,3; cbd_dist^1,2,3
new = [data.home_size, hsize2, hsize3, data.cbd_dist, cbd2, cbd3]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 4 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 4 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)
results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))

#MODEL 5
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create exponential variables
hsize2 = np.power(data.home_size,2)
beds2 = np.power(data.beds,2)
age2 = np.power(data.age,2)
xcoord2 = np.power(data.x_coord,2)
ycoord2 = np.power(data.y_coord,2)
beds_hsize = np.multiply(data.beds, data.home_size)
#create new dataframe with our models variables: beds, beds^2, home_size, home_size^2, (beds*home_size), cbd_dist, parcel_size, x_coord^2, y_coord^2, age^2
new = [data.home_size, hsize2, data.beds, beds2, beds_hsize, data.cbd_dist, data.parcel_size, xcoord2, ycoord2, age2]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 5 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 5 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)
results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))

#MODEL 6
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create exponential variables
hsize2 = np.power(data.home_size,2)
beds2 = np.power(data.beds,2)
age2 = np.power(data.age,2)
xcoord2 = np.power(data.x_coord,2)
ycoord2 = np.power(data.y_coord,2)
beds_hsize = np.multiply(data.beds, data.home_size)
beds_psize = np.multiply(data.beds, data.parcel_size)
psize_hsize = np.multiply(data.parcel_size, data.home_size)
#create new dataframe with our models variables: home_size, beds, (home_size * beds), cbd_dist, (beds * parcel_size), parcel_size, (parcelsize*homesize)
new = [data.home_size, data.beds, beds_hsize, data.cbd_dist, beds_psize, data.parcel_size, psize_hsize]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 6 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 6 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)
results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))

#MODEL 7
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
psize2 = np.power(data.parcel_size,2)
psize3 = np.power(data.parcel_size,3)
psize4 = np.power(data.parcel_size,4)
#create new dataframe with our models variables: home size^1,2,3,4; beds^1,2,3,4; parcel_size^1,2,3,4
new = [data.home_size, hsize2, hsize3, hsize4, data.beds, beds2, beds3, beds4, data.parcel_size, psize2, psize3, psize4]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 7 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)

#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 7 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)
results = model.score(x_test, y_test)
print("Accuracy: %.3f%%" % (results*100.0))

#MODEL 8
fulldata = read_csv('/Users/Adam/sales.csv',delimiter=',',)
data = read_csv('/Users/Adam/sales.csv', delimiter=',' , usecols=(1,2,3,4,5,6,7,8,9))
#define target (dependent) variable as y
y = pd.DataFrame(fulldata.price)
#create exponential variables
year2 = np.power(data.year,2)
age2 = np.power(data.age,2)
xcoord2 = np.power(data.x_coord,2)
ycoord2 = np.power(data.y_coord,2)
hsize_beds = np.multiply(data.beds, data.home_size)
psize_hsize = np.multiply(data.parcel_size, data.home_size)
psize_beds = np.multiply(data.beds, data.parcel_size)
hsize_beds_psize = np.multiply(data.home_size, psize_beds)
#create new dataframe with our models variables: home_size, year^2, parcel_size, beds, age^2, pool, cbd_dist, x_coord^2, y_coord^2, (home_size*beds), (home_size*parcel_size), (beds*parcel_size), (home_size*beds*parcel_size)
new = [data.home_size, year2, data.parcel_size, data.beds, age2, data.pool, data.cbd_dist, xcoord2, ycoord2, hsize_beds, psize_hsize, psize_beds, hsize_beds_psize]
df = pd.DataFrame(new)
df = np.transpose(df)
#create training/testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#fit a model
model = LinearRegression()
model.fit(x_train, y_train)
trainpredictions = model.predict(x_train)
testpredictions = model.predict(x_test)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 8 (Pre Cross-Validation)")
#score the model
print("Training R^2:", model.score(x_train,y_train))
print("Testing R^2:", model.score(x_test,y_test))
trainmse = np.sum((y_train - trainpredictions)**2)/9000
testmse = np.sum((y_test - testpredictions)**2)/1000
print("Train MSE=", trainmse)
print("Test MSE=", testmse)
#Perform 10-Fold Cross Validation
scores = cross_val_score(model, df, y, cv=10)
print("Cross-validated scores:", scores)
#Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=10)
plt.scatter(y, predictions)
plt.title("Model 8 (Cross-Validated)")
#Check Rsquared
accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)



#Fit all of data with superior model
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
#fit a model
model = LinearRegression()
model.fit(df,y)
#plot the model
plt.scatter(y_test, testpredictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Model 2 (Estimating Whole Dataset)")
#score the model
print("R^2:", model.score(df,y))

(4500, 12) (4500, 1)
(500, 12) (500, 1)
Training R^2: 0.6820307390779174
Testing R^2: 0.7884524680570247
Train MSE= price    3416.297967
dtype: float64
Test MSE= price    2073.140352
dtype: float64
Cross-validated scores: [0.76333    0.64328504 0.22122324 0.70067431 0.65729694 0.77119813
 0.70641923 0.7141794  0.22667466 0.55698388]
Cross-Predicted Accuracy: 0.5557643578026634
