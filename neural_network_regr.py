import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from scipy import stats
from matplotlib import pyplot as plt

"""Load data"""
dataset = np.load('processed_trip_data_5.npy')
data = pd.DataFrame(dataset)


"""Function for swapping columns"""
def swap_columns(df, c1, c2):
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)


swap_columns(data, 5, 7)


"calculate distance between pickup and drop-off"
def calc_distance(x1,x2,y1,y2):
    dist = np.sqrt(np.abs(x2-x1)**2+np.abs(y2-y1)**2)
    return dist


pu_x = data[2]
pu_y = data[3]
do_x = data[6]
do_y = data[5]

data[8] = calc_distance(pu_x, do_x, pu_y, do_y)

swap_columns(data, 7,8)
swap_columns(data, 4,7)
swap_columns(data, 4,7)
# swap_columns(data, 4,7)

print(data)


"remove outliers from data"
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

"randomize order of rows"
data = data.sample(frac=1)


"Split data in input and output"
X = data.iloc[:,0:5:]
Y = data.iloc[:,8:9:]

"normalize data"
d = preprocessing.normalize(X, axis=0)
df = pd.DataFrame(d)
df[2] = [-x for x in df[2]]
# df[6] = [-x for x in df[6]]

print(Y.describe())

X_new = X.iloc[:100000]
Y_new = Y.iloc[:100000]

Y_new = np.array(Y_new).ravel()

print(X_new, Y_new)
Xtrain , Xtest, Ytrain, Ytest = train_test_split(X_new,Y_new, test_size=0.2, random_state=0)

print("Datapoints in Training set:",len(Xtrain))
print("Datapoints in Test set:",len(Xtest))



""" sklearn model """
neural_net = MLPRegressor(max_iter=3000, hidden_layer_sizes=8, activation="relu").fit(Xtrain, Ytrain)
print('model is fitted')
pred = neural_net.predict(Xtest)
mae = mean_absolute_error(pred, Ytest)
mse = mean_squared_error(pred, Ytest, squared=False)

print(f"The MAE of this network is  {mae}")
print(f"The RMSE of this network is {mse}")

# """Multiple Linear Regression"""
LR = LinearRegression()
LR.fit(Xtrain, Ytrain)
pred_reg = LR.predict(Xtest)
mae_reg = mean_absolute_error(pred_reg, Ytest)
mse_reg = mean_squared_error(pred_reg, Ytest, squared=False)

print("Linear Regression Results")
print(f"The MAE of this network is  {mae_reg}")
print(f"The MSE of this network is {mse_reg}")


