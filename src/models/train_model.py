import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import backend as K

from pipeline.config import set_config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
def dataset_splitter(X, y, train_size=0.8):
  train_size = int(len(X) * 0.8)
  X_train, X_test = X[:train_size], X[train_size:]
  y_train, y_test = y[:train_size], y[train_size:]
  # Checking the shape of the training and testing sets
  X_train.shape, X_test.shape, y_train.shape, y_test.shape
  
  return X_train, X_test, y_train, y_test  


# --------------------------------------------------------------
# Get a visualizatino of how our test set has been split
# --------------------------------------------------------------

# Get the count for the specific value
# df_train_count = s1_df_train['Sensor1 Moisture (%)'].shape[0]
# y_train_count = y_train['Sensor1 Moisture (%)'].shape[0]
# y_test_count = y_test['Sensor1 Moisture (%)'].shape[0]

# # Create a DataFrame for the specific value
# s1_df_train_plot = pd.DataFrame({'Total': [df_train_count]})
# s1_y_train_plot = pd.DataFrame({'Train': [y_train_count]})
# s1_y_test_plot = pd.DataFrame({'Test': [y_test_count]})

# # Plot the bar graphs
# fig, ax = plt.subplots(figsize=(12, 6))
# s1_df_train_plot.plot(kind='bar', ax=ax, color='lightblue', label='Total')
# s1_y_train_plot.plot(kind='bar', ax=ax, color='dodgerblue', label='Train')
# s1_y_test_plot.plot(kind='bar', ax=ax, color='royalblue', label='Train')
# plt.legend()
# plt.show()

# --------------------------------------------------------------
# Let's begin training
# --------------------------------------------------------------

def train_evaluate_lstm(
  X_scaled,
  y_scaled,
  n_splits=5,
  epochs=50,
  batch_size=32
):

  # Scaling the data
  scaler_X = MinMaxScaler()
  X_scaled = scaler_X.fit_transform(X_scaled)
  scaler_y = MinMaxScaler()
  y_scaled = scaler_y.fit_transform(y_scaled.values.reshape(-1, 1))

  # Function to create LSTM model
  def create_lstm_model(input_shape):
      model = Sequential()
      model.add(LSTM(50, input_shape=input_shape))
      model.add(Dense(1))
      model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
      return model

  # Initializing KFold for cross-validation
  kf = KFold(n_splits=n_splits, shuffle=False)
  cv_scores_mae = []
  cv_scores_rmse = []

  # Callbacks
  # callbacks = [
  #   EarlyStopping(monitor='val_loss', patience=20),
  #   ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
  # ]
  
  # K-fold cross-validation
  for train_index, val_index in kf.split(X_scaled):

    CV_X_train, CV_X_val = X_scaled[train_index], X_scaled[val_index]
    CV_y_train, CV_y_val = y_scaled[train_index], y_scaled[val_index]
    CV_X_train = CV_X_train.reshape(CV_X_train.shape[0], 1, CV_X_train.shape[1])
    CV_X_val = CV_X_val.reshape(CV_X_val.shape[0], 1, CV_X_val.shape[1])
    
    model = create_lstm_model((CV_X_train.shape[1], CV_X_train.shape[2]))
    model.fit(CV_X_train, CV_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred_val_scaled = model.predict(CV_X_val)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled).flatten()

    print(pd.DataFrame(data={'Train Predictions': y_pred_val, 'Actuals': CV_y_train}))

    mae = mean_absolute_error(y_scaled[val_index], y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_scaled[val_index], y_pred_val))
    cv_scores_mae.append(mae)
    cv_scores_rmse.append(rmse)
    
    K.clear_session()

  results = {
      'cv_mae': np.mean(cv_scores_mae),
      'cv_rmse': np.mean(cv_scores_rmse)
  }

  return results

# --------------------------------------------------------------
# We will divide our features into subsets to see if the additional
# features we engineered help our predictive performance of our model.
# --------------------------------------------------------------

basic_features = [
  'Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
  'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h', 'Heat Index - C', 
  'THW Index - C', 'Rain - mm', 'Heating Degree Days', 'Cooling Degree Days'
]

interaction_features = [
  'temp_hum_interaction','barometer_temp_interaction', 'wind_temp_interaction',
  'dew_hum_interaction', 'heat_cool_interaction','rain_wind_interaction', 
  'sensor1_temp_hum_interaction', 'sensor2_temp_hum_interaction', 'overall_moisture_index'
]

lag_features = [
  'Barometer - hPa_lag_1_day', 'Hum - %_lag_1_day',
  'Dew Point - C_lag_1_day', 'Wet Bulb - C_lag_1_day', 
]

# TODO: go back and perform this eng. do we need them?
# square features = [] 
# cluster features = []
# pca_features = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']
# need a list comprehension to access these
# time_features = [f for f in s1_df_train.columns if '_temp_' in f]
# frequency_features = [f for f in s1_df_train.columns if ('_freq' in f) or ('_pse_' in f)]

print(f'Basic Features: {len(basic_features)}')
print(f'Interaction Features: {len(interaction_features)}')
print(f'Lag Features: {len(lag_features)}')

# use this later on to use different data selections
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + interaction_features)) # + square_features 
feature_set_3 = list(set(feature_set_2 + lag_features))

feature_set_1_df = efficient_feature_df[feature_set_1]
X_train, X_test, y_train, y_test = dataset_splitter(feature_set_1_df, y)

lstm_results = train_evaluate_lstm(X_train, y_train)
print(lstm_results)
