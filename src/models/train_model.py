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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline.config import set_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
set_config()

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
# Baseline: Linear Regression
# --------------------------------------------------------------

# Function to train and evaluate Simple Linear Regression
def train_evaluate_linear_regression(X_train, y_train, X_test, y_test):
    # Initializing TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Initializing Linear Regression model
    model = LinearRegression()

    # Cross-validation scores
    cv_scores_mae = []
    cv_scores_rmse = []

    # Time-series cross-validation
    for train_index, val_index in tscv.split(X_train):
        CV_X_train, CV_X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        CV_y_train, CV_y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fitting the model
        model.fit(CV_X_train, CV_y_train)

        # Predicting on the validation set
        y_pred_val = model.predict(CV_X_val)

        # Calculating MAE and RMSE for the validation set
        mae = mean_absolute_error(CV_y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(CV_y_val, y_pred_val))

        # Storing the scores
        cv_scores_mae.append(mae)
        cv_scores_rmse.append(rmse)

    # Training the model on the full training set
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred_test = model.predict(X_test)

    # Calculating MAE and RMSE for the test set
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results = {
        'cv_mae': np.mean(cv_scores_mae),
        'cv_rmse': np.mean(cv_scores_rmse),
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }

    return model, results

# --------------------------------------------------------------
# LSTM Model
# --------------------------------------------------------------

def train_evaluate_kfold_lstm(
  X_train, y_train, n_splits=5,epochs=50,batch_size=32
):

  # Scaling the data
  scaler_X = MinMaxScaler()
  X_train = scaler_X.fit_transform(X_train)
  scaler_y = MinMaxScaler()
  y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

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
  
  # K-fold cross-validation. training & validation split
  for train_index, val_index in kf.split(X_train):

    # prep data for model
    CV_X_train, CV_X_val = X_train[train_index], X_train[val_index]
    CV_y_train, CV_y_val = y_train[train_index], y_train[val_index]

    CV_X_train = CV_X_train.reshape(CV_X_train.shape[0], 1, CV_X_train.shape[1])
    CV_X_val = CV_X_val.reshape(CV_X_val.shape[0], 1, CV_X_val.shape[1])
    
    # actual ML
    model = create_lstm_model((CV_X_train.shape[1], CV_X_train.shape[2]))
    model.fit(CV_X_train, CV_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred_val_scaled = model.predict(CV_X_val)

    # invert scaling for train predictions, back to original form
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled).flatten()

    mae = mean_absolute_error(y_train[val_index], y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_train[val_index], y_pred_val))
    cv_scores_mae.append(mae)
    cv_scores_rmse.append(rmse)
    
    K.clear_session()

  results = {
      'train_cv_mae': np.mean(cv_scores_mae),
      'train_cv_rmse': np.mean(cv_scores_rmse), 
  }

  return results

# LSTM
def train_evaluate_simple_lstm(X_train, y_train, X_test, epochs=50, batch_size=32):
    # Reshaping the data
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fitting the model
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predicting
    train_predictions = model.predict(X_train_reshaped)
    test_predictions = model.predict(X_test_reshaped)

    return train_predictions.flatten(), test_predictions.flatten()

# --------------------------------------------------------------
# We will divide our features into subsets to see if the additional
# features we engineered help our predictive performance of our model.
# --------------------------------------------------------------

efficient_feature_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

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

print(f'Basic Features: {len(basic_features)}')
print(f'Interaction Features: {len(interaction_features)}')
print(f'Lag Features: {len(lag_features)}')

# drop columns we will not use at all
s1_drop_columns_for_training = ['Sensor2 (Ohms)', 'Sensor1 Moisture (%)', 'Sensor2 Moisture (%)']
s1_df_train = efficient_feature_df.drop(s1_drop_columns_for_training, axis=1)

# drop columns we'll predict later
X = s1_df_train.drop(['Sensor1 (Ohms)', ], axis=1)
y = s1_df_train[['Sensor1 (Ohms)']]

# use this later on to use different data selections
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(feature_set_1 + interaction_features)) 
feature_set_3 = list(set(feature_set_1 + lag_features))
feature_set_4 = list(set(feature_set_2 + lag_features))

pretraining_set_X = X[feature_set_2]

X_train, X_test, y_train, y_test = dataset_splitter(
   pretraining_set_X, y
)

# Training and evaluating the Simple Linear Regression model
linear_model, linear_results = train_evaluate_linear_regression(X_train, y_train, X_test, y_test)
print("Linear Regression Baseline")
print(linear_results)

# kfold_lstm_results = train_evaluate_kfold_lstm(X_train, y_train)
# print("LSTM KFold Performance Results")
# print(kfold_lstm_results)

lstm_train_results, lstm_test_results = train_evaluate_simple_lstm(X_train, y_train, X_test)
lstm_mean_absolute_error = mean_absolute_error(y_test, lstm_test_results)
lstm_mean_squared_error = mean_squared_error(y_test, lstm_test_results)
print("LSTM Performance Results")
print(f"LSTM Mean Absolute Error: {lstm_mean_absolute_error}")
print(f"LSTM Mean Squared Error: {lstm_mean_squared_error}")

# residuals_train = y_train - np.mean(lstm_train_results)
# residuals_test = y_test - np.mean(lstm_test_results)

array_series = pd.Series(lstm_test_results)
array_series.index = y_test.index
result = y_test.iloc[:, 0] - array_series
result = y_test.apply(lambda x: x - array_series)

residuals = y_test - lstm_test_results

# Plot the residuals
plt.figure(figsize=(8, 6))
sns.residplot(x=lstm_test_results, y=result, lowess=True, scatter_kws={'s': 50}, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, residuals_train, alpha=0.5)
plt.title('Training Residuals')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
plt.scatter(y_test, residuals_test, alpha=0.5)
plt.title('Test Residuals')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test[:1000].values, label='Actual')
plt.plot(lstm_test_results[:1000], label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(lstm_test_results, label='Predicted')
plt.title('Zoomed In Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
