import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import backend as K

import matplotlib.pyplot as plt
import pandas as pd
from pipeline.config import set_config
set_config()

efficient_feature_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

# drop columns we will not use at all
s1_drop_columns_for_training = ['Sensor1 (Ohms)', 'Sensor2 (Ohms)', 'Sensor2 Moisture (%)']
s1_df_train = efficient_feature_df.drop(s1_drop_columns_for_training, axis=1)

# drop columns we'll predict later
X = s1_df_train.drop(['Sensor1 Moisture (%)', ], axis=1)
y = s1_df_train[['Sensor1 Moisture (%)']]

# providing seed. we take control of the stochastic split to reproduce
# s1_X_train, s1_X_test, s1_y_train, s1_y_test = train_test_split(
#   s1_X, s1_y, test_size=0.25, random_state=42, #stratify=s1_y
# )
# Splitting the data into training and testing sets (80% training, 20% testing) in chronological order
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Checking the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# --------------------------------------------------------------
# Get a visualizatino of how our test set has been split
# --------------------------------------------------------------

# Get the count for the specific value
df_train_count = s1_df_train['Sensor1 Moisture (%)'].shape[0]
y_train_count = y_train['Sensor1 Moisture (%)'].shape[0]
y_test_count = y_test['Sensor1 Moisture (%)'].shape[0]

# Create a DataFrame for the specific value
s1_df_train_plot = pd.DataFrame({'Total': [df_train_count]})
s1_y_train_plot = pd.DataFrame({'Train': [y_train_count]})
s1_y_test_plot = pd.DataFrame({'Test': [y_test_count]})

# Plot the bar graphs
fig, ax = plt.subplots(figsize=(12, 6))
s1_df_train_plot.plot(kind='bar', ax=ax, color='lightblue', label='Total')
s1_y_train_plot.plot(kind='bar', ax=ax, color='dodgerblue', label='Train')
s1_y_test_plot.plot(kind='bar', ax=ax, color='royalblue', label='Train')
plt.legend()
plt.show()

# --------------------------------------------------------------
# We will divide our features into subsets to see if the additional
# features we engineered help our predictive performance of our model.
# --------------------------------------------------------------

basic_features = [
    'Barometer - hPa', 
    'Temp - C', 
    'High Temp - C', 
    'Low Temp - C',
    'Hum - %', 
    'Dew Point - C', 
    'Wet Bulb - C', 
    'Wind Speed - km/h',
    'Heat Index - C', 
    'THW Index - C', 
    'Rain - mm', 
    'Heating Degree Days',
    'Cooling Degree Days'
]

# TODO: go back and perform this eng. do we need them?
# square features = [] 
# cluster features = []
# pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']

# need a list comprehension to access these
time_features = [f for f in s1_df_train.columns if '_temp_' in f]
frequency_features = [f for f in s1_df_train.columns if ('_freq' in f) or ('_pse_' in f)]

print(f'Basic Features: {len(basic_features)}')
print(f'PCA Features: {len(pca_features)}')
print(f'Time Features: {len(time_features)}')
print(f'Frequency Features: {len(frequency_features)}')

# use this later on to use different data selections
feature_set_1 = list(set(basic_features))
# feature_set_2 = set(basic_features + pca_features) # + square_features 
# feature_set_3 = list(set(feature_set_2 + time_features))
# feature_set_4 = list(set(feature_set_3 + frequency_features)) # + cluster_features

efficient_feature_df[feature_set_1]

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# try one feature set at a time to compare performance
# --------------------------------------------------------------

