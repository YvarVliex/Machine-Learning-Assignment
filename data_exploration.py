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
import seaborn as sns


dataset = np.load('processed_trip_data_5.npy')
data = pd.DataFrame(dataset)

print(data)
print(data.describe()) #find info such as mean and std of the dataset

"""Plot possible relations"""
# sns.catplot(x=4, y=5, data=data)
# plt.xlabel("Number of passengers")
# plt.ylabel("Trip Duration (s)")
# plt.show()

sns.catplot(x=1, y=5, data=data)
plt.xlabel("Day of the week")
plt.ylabel("Trip Duration (s)")

# sns.catplot(x=0, y=5, data=data)
# plt.xlabel("Hour of the day")
# plt.ylabel("Trip Duration (s)")
# plt.show()
plt.show()