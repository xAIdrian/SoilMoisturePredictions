import sys
import os

# Get the absolute path of the root directory (adjust according to your specific structure)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root path to sys.path
sys.path.append(root_path)

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from data.datetime_utils import set_datetime_as_index
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -----------------------------------------------------------------
# Organize our two sensor datasets and optimize the datetime
# -----------------------------------------------------------------

all_files = glob('../../data/raw/*.csv')
print(f"Total number of files: {len(all_files)}")

metea_model_df = pd.DataFrame()
sensor1_df = pd.DataFrame()
sensor2_df = pd.DataFrame()

for filename in all_files:
  with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
    if 'meteo_data for model' in filename:
      metea_model_df = pd.read_csv(file)
    elif 'Sensor1' in filename:
      sensor1_df = pd.read_csv(file)
    elif 'Sensor2' in filename:
      sensor2_df = pd.read_csv(file)

# set dates as index
metea_model_df = set_datetime_as_index(metea_model_df, 'Date & Time')
sensor1_df = set_datetime_as_index(sensor1_df, 'Date & Time')
sensor2_df = set_datetime_as_index(sensor2_df, 'Date&Time')

sensor1_df = sensor1_df.groupby('Date & Time').mean()
sensor2_df = sensor2_df.groupby('Date&Time').mean()

# resampling to 15 min intervals. setting the average within each 15 min bucket
# we could use reset_index() to get the date & time back as a column
meteo_model_resampled = metea_model_df.resample('15T').mean()
sensor1_resampled = sensor1_df.resample('15T').mean()
sensor2_resampled = sensor2_df.resample('15T').mean()

# Create two new dataframes, one for each sensor. we will also drop end info that showed errors
full_sensor_one_df = pd.merge(meteo_model_resampled, sensor1_resampled, left_index=True, right_index=True, how='left')
full_sensor_two_df = pd.merge(meteo_model_resampled, sensor2_resampled, left_index=True, right_index=True, how='left')

# remove the null values
full_sensor_one_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
full_sensor_two_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# --------------------------------------------------------------
# Let's Standardize The Whole Dataframe and Try Corrolation Again
# As a sanity check, we should see the same results
# --------------------------------------------------------------
scaled_df = final_df.copy()

# just the column names
cols_to_minmax_scale = ['Barometer - hPa']
cols_to_standard_scale = [col for col in scaled_df.columns if col not in cols_to_minmax_scale]

# Scale all of our features, 2 types
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

cols_to_minmax_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_minmax_scale])
scaled_df[cols_to_minmax_scale] = cols_to_minmax_scale_values

cols_to_standard_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_standard_scale])
scaled_df[cols_to_standard_scale] = cols_to_standard_scale_values

# Plotting a heatmap for visualization
scaled_correlation_matrix = scaled_df.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(scaled_correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Scaled Correlation Heatmap")
plt.show()
