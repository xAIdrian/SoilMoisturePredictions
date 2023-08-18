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
from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from pipeline.config import set_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
set_config()

def resistance_to_moisture(resistance):
    if resistance <= 2000:
        return 100.00
    elif resistance < 55000:
        # Assuming a linear decrease from 100% at 2000 Ohms to 0% at 55,000 Ohms
        return 100 - ((resistance - 2000) / (55000 - 2000)) * 100
    else:
        return 0.00
vectorized_res_to_moist = np.vectorize(resistance_to_moisture)

# Splitting the data into training and testing sets (80% training, 20% testing) in chronological order
def dataset_splitter(X, y, train_size=0.8):
  train_size = int(len(X) * 0.8)
  X_train, X_test = X[:train_size], X[train_size:]
  y_train, y_test = y[:train_size], y[train_size:]

  # Checking the shape of the training and testing sets
  X_train.shape, X_test.shape, y_train.shape, y_test.shape
  
  return X_train, X_test, y_train, y_test  

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
    
    y_pred_val = model.predict(CV_X_val)

    # invert scaling for train predictions, back to original form
    # y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled).flatten()

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
    y_train_preds = model.predict(X_train_reshaped)
    y_test_preds = model.predict(X_test_reshaped)

    return y_train_preds.flatten(), y_test_preds.flatten()

# --------------------------------------------------------------
# We will divide our features into subsets to see if the additional
# features we engineered help our predictive performance of our model.
# --------------------------------------------------------------

def train_and_plot(X, y, prediction_features, title, full_output = False):

  pretraining_set_X = X[prediction_features]

  scaler_X = MinMaxScaler()
  X_train = scaler_X.fit_transform(pretraining_set_X)
  scaler_y = MinMaxScaler()
  y = scaler_y.fit_transform(y)

  X_train, X_test, y_train, y_test = dataset_splitter(
    pretraining_set_X, y
  )

  if full_output == True:
    # Training and evaluating the Simple Linear Regression model
    linear_model, linear_results = train_evaluate_linear_regression(X_train, y_train, X_test, y_test)
    print(f"{title} Linear Regression Baseline")
    print(linear_results)

    kfold_lstm_results = train_evaluate_kfold_lstm(X_train, y_train)
    print(f"{title} LSTM KFold Performance Results")
    print(kfold_lstm_results)

  lstm_train_results, lstm_test_results = train_evaluate_simple_lstm(X_train, y_train, X_test)

  if len(lstm_test_results) != y_test.shape[0]:
    raise ValueError("Array length does not match the number of rows in DataFrame")

  lstm_mean_absolute_error = mean_absolute_error(y_test, lstm_test_results)
  lstm_mean_squared_error = mean_squared_error(y_test, lstm_test_results)
  mape = mean_absolute_percentage_error(y_test, lstm_test_results)

  print(f"{title}: LSTM Performance Results")
  print(f"LSTM Mean Absolute Error: {lstm_mean_absolute_error}")
  print(f"LSTM Mean Squared Error: {lstm_mean_squared_error}")
  print("Mean Absolute Percentage Error (MAPE):", round(mape, 6), "%")

  test_series = pd.Series(y_test.flatten())
  pred_series = pd.Series(lstm_test_results)
  # pred_series.index = y_test.index
  residuals = test_series - pred_series
  # result = test_series.apply(lambda x: x - pred_series)
  # residuals = y_test - result

  # Plot the residuals
  plt.figure(figsize=(8, 6))
  sns.residplot(x=y_test, y=residuals, lowess=True, scatter_kws={'s': 50}, line_kws={'color': 'red', 'lw': 2})
  plt.xlabel('Fitted values')
  plt.ylabel('Residuals')
  plt.title(f'{title} Residual plot')
  plt.show()

  y_test = scaler_y.inverse_transform(y_test)
  y_pred_val = scaler_y.inverse_transform(pred_series.values.reshape(-1, 1))

  # Extracting the relevant datetime index values
  DATETIME_WINDOW = 150
  datetime_index = X_train.index[:DATETIME_WINDOW]

  plt.figure(figsize=(12, 6))
  plt.plot(datetime_index, y_test[:DATETIME_WINDOW], label='Actual')
  plt.plot(datetime_index, y_pred_val[:DATETIME_WINDOW], label='Predicted')
  plt.title(f"{title}: Early Timeframe: Actual vs Predicted Values")
  plt.xlabel('Date') 
  plt.ylabel('Sensor (Ohms)')
  plt.xticks(rotation=90)
  plt.legend()
  plt.show()

  datetime_index = X_train.index[len(X_train) - DATETIME_WINDOW:]

  plt.figure(figsize=(12, 6))
  plt.plot(datetime_index, y_test[len(y_test) - DATETIME_WINDOW:], label='Actual')
  plt.plot(datetime_index, y_pred_val[len(y_pred_val) - DATETIME_WINDOW:], label='Predicted')
  plt.title(f"{title}: Late Timeframe: Actual vs Predicted Values")
  plt.xlabel('Date') 
  plt.ylabel('Sensor (Ohms)')
  plt.xticks(rotation=90)
  plt.legend()
  plt.show()

  plt.figure(figsize=(12, 6))
  plt.plot(y_test, label='Actual')
  plt.plot(y_pred_val, label='Predicted')
  plt.title(f"{title}: Actual vs Predicted Values")
  plt.xlabel('Date') 
  plt.ylabel('Sensor (Ohms)')
  plt.xticks([])
  plt.legend()
  plt.show()

# --------------------------------------------------------------
# Actual Execution
# --------------------------------------------------------------

efficient_feature_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

basic_features = [
  'Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
  'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h', 'Heat Index - C', 
  'THW Index - C', 'Rain - mm', 'Heating Degree Days', 'Cooling Degree Days'
]

lag_features = [col for col in efficient_feature_df.columns if "_lag_" in col]
interaction_features = [col for col in efficient_feature_df.columns if "_interaction" in col]
pca_features = [col for col in efficient_feature_df.columns if "pca_" in col]

print(f'Basic Features: {len(basic_features)}')
print(f'Interaction Features: {len(interaction_features)}')
print(f'Lag Features: {len(lag_features)}')
print(f'PCA Features: {len(pca_features)}')

# use this later on to use different data selections
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + pca_features))
feature_set_3 = list(set(basic_features + interaction_features)) 
feature_set_4 = list(set(basic_features + lag_features))

# Sensor 1 Focus
sensor1_df = efficient_feature_df.copy()

s1_drop_columns_for_training = ['Sensor1 Moisture (%)', 'Sensor2 Moisture (%)']
sensor1_df.drop(s1_drop_columns_for_training, axis=1, inplace=True)

# This is our prediction column
X = sensor1_df.drop(['Sensor1 (Ohms)', ], axis=1)
y = sensor1_df[['Sensor1 (Ohms)']]

# train_and_plot(X, y, pca_features, 'Sensor 1 PCA Only')

sensor2_df = efficient_feature_df.copy()

s2_drop_columns_for_training = ['Sensor1 Moisture (%)', 'Sensor2 Moisture (%)']
sensor2_df.drop(s2_drop_columns_for_training, axis=1, inplace=True)

# This is our prediction column
X = sensor2_df.drop(['Sensor2 (Ohms)', ], axis=1)
y = sensor2_df[['Sensor2 (Ohms)']]

# train_and_plot(X, y, pca_features, 'Sensor 2 PCA')
# train_and_plot(X, y, feature_set_1, 'Sensor 2 Basic Features')
# train_and_plot(X, y, feature_set_2, 'Sensor 2 Basic & PCA Features')

# --------------------------------------------------------------
# Isolate sets to make sure predictions are generalized
# --------------------------------------------------------------

isolated_set_df = efficient_feature_df.copy()

isolated_set_df['month'] = isolated_set_df.index.month
isolated_set_df['day_of_week'] = isolated_set_df.index.dayofweek
isolated_set_df['week_of_year'] = (isolated_set_df.index.dayofyear - 1) // 7 + 1

drop_columns_for_training = ['Sensor1 Moisture (%)', 'Sensor2 Moisture (%)']
isolated_set_df.drop(drop_columns_for_training, axis=1, inplace=True)

# This is our prediction column
MONTH_TO_TEST = 4
pre_X_train = isolated_set_df[isolated_set_df['month'] != MONTH_TO_TEST].drop(['Sensor2 (Ohms)', ], axis=1)[pca_features]
pre_y_train = isolated_set_df[isolated_set_df['month'] != MONTH_TO_TEST][['Sensor2 (Ohms)']]

pre_X_test = isolated_set_df[isolated_set_df['month'] == MONTH_TO_TEST].drop(['Sensor2 (Ohms)', ], axis=1)[pca_features]
pre_y_test = isolated_set_df[isolated_set_df['month'] == MONTH_TO_TEST]['Sensor2 (Ohms)']

scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(pre_X_train)
X_test = scaler_X.transform(pre_X_test)
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(pre_y_train)
y_test = scaler_y.fit_transform(pre_y_test.values.reshape(-1, 1))

# Training and evaluating the Simple Linear Regression model
# linear_model, linear_results = train_evaluate_linear_regression(X_train, y_train, X_test, y_test)
# print(f"Linear Regression Baseline")
# print(linear_results)

# kfold_lstm_results = train_evaluate_kfold_lstm(X_train, y_train)
# print(f"LSTM KFold Performance Results")
# print(kfold_lstm_results)

# lstm_train_results, lstm_test_results = train_evaluate_simple_lstm(X_train, y_train, X_test)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fitting the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=25, verbose=0)

# Predicting
lstm_train_results = model.predict(X_train_reshaped).flatten()
lstm_test_results = model.predict(X_test_reshaped).flatten()

if len(lstm_test_results) != y_test.shape[0]:
  raise ValueError("Array length does not match the number of rows in DataFrame")

lstm_mean_absolute_error = mean_absolute_error(y_test, lstm_test_results)
lstm_mean_squared_error = mean_squared_error(y_test, lstm_test_results)
mape = mean_absolute_percentage_error(y_test, lstm_test_results)

print(f"LSTM Performance Results")
print(f"LSTM Mean Absolute Error: {lstm_mean_absolute_error}")
print(f"LSTM Mean Squared Error: {lstm_mean_squared_error}")
print("Mean Absolute Percentage Error (MAPE):", round(mape, 6), "%")

test_series = pd.Series(y_test.flatten())
pred_series = pd.Series(lstm_test_results)
# pred_series.index = y_test.index
residuals = test_series - pred_series
# result = test_series.apply(lambda x: x - pred_series)
# residuals = y_test - result

# Plot the residuals
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=residuals, lowess=True, scatter_kws={'s': 50}, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title(f'Residual plot')
plt.show()

y_test = scaler_y.inverse_transform(y_test)
y_pred_val = scaler_y.inverse_transform(pred_series.values.reshape(-1, 1))

# Apply the custom function to your dataframe (assuming df is your dataframe containing resistance values)
plot_y_test = vectorized_res_to_moist(y_test)
plot_y_pred_val = vectorized_res_to_moist(y_pred_val)

# Extracting the relevant datetime index values
DATETIME_WINDOW = 150
datetime_index = pre_X_test.index[:DATETIME_WINDOW]

plt.figure(figsize=(12, 6))
plt.plot(datetime_index, plot_y_test[:DATETIME_WINDOW], label='Actual')
plt.plot(datetime_index, plot_y_pred_val[:DATETIME_WINDOW], label='Predicted')
plt.title(f"Early Timeframe: Month Omitted: Actual vs Predicted Values")
plt.xlabel('Date') 
plt.ylabel('Moisture (%)')
plt.xticks(rotation=90)
plt.legend()
plt.show()

datetime_index = pre_X_test.index[len(pre_X_test) - DATETIME_WINDOW:]

plt.figure(figsize=(12, 6))
plt.plot(datetime_index, plot_y_test[len(y_test) - DATETIME_WINDOW:], label='Actual')
plt.plot(datetime_index, plot_y_pred_val[len(y_pred_val) - DATETIME_WINDOW:], label='Predicted')
plt.title(f"Late Timeframe: Month Omitted: Actual vs Predicted Values")
plt.xlabel('Date') 
plt.ylabel('Moisture (%)')
plt.xticks(rotation=90)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(plot_y_test, label='Actual')
plt.plot(plot_y_pred_val, label='Predicted')
plt.title(f"Month Omitted: Actual vs Predicted Values")
plt.xlabel('Index') 
plt.ylabel('Mositure (%)')
plt.legend()
plt.show()
