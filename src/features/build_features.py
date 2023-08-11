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
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data.datetime_utils import set_datetime_as_index
from DataTransformation import PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
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

predictor_columns = ['Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
       'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h',
       'Heat Index - C', 'THW Index - C', 'Rain - mm', 'Heating Degree Days',
       'Cooling Degree Days']      

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

# feature engineering
meteo_model_resampled['month'] = meteo_model_resampled.index.month
meteo_model_resampled['day_of_week'] = meteo_model_resampled.index.dayofweek
meteo_model_resampled['week_of_year'] = (meteo_model_resampled.index.dayofyear - 1) // 7 + 1

# Create two new dataframes, one for each sensor. we will also drop end info that showed errors
full_sensor_one_df = pd.merge(meteo_model_resampled, sensor1_resampled, left_index=True, right_index=True, how='left')
full_sensor_two_df = pd.merge(meteo_model_resampled, sensor2_resampled, left_index=True, right_index=True, how='left')

# --------------------------------------------------------------
# Dealing with scaling and missing values (imputation)
# --------------------------------------------------------------
# remove the missing values
full_sensor_one_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
full_sensor_two_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# proper scaling of all values 
scaled_df = full_sensor_one_df.copy()

# just the column names
cols_feature_eng = ['month', 'day_of_week', 'week_of_year']
cols_to_minmax_scale = ['Barometer - hPa']
cols_to_standard_scale = [col for col in scaled_df.columns if col not in cols_to_minmax_scale and col not in cols_feature_eng]

# Scale all of our features, 2 types
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

cols_to_minmax_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_minmax_scale])
scaled_df[cols_to_minmax_scale] = cols_to_minmax_scale_values

cols_to_standard_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_standard_scale])
scaled_df[cols_to_standard_scale] = cols_to_standard_scale_values

# --------------------------------------------------------------
# --------------------------------------------------------------
# We'll start by analyzing sensor 1 all on its own then we will
# come back to look at sensor 2
# --------------------------------------------------------------
# --------------------------------------------------------------

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = scaled_df.copy()
pca = PrincipalComponentAnalysis()

# Paper reports ptimal amount of principle components is 5. 
# Let's test this

pc_values = pca.determine_pc_explained_variance(full_sensor_one_df, predictor_columns)

# elbow techniques to find best number of PCs
# capture the most variance without incorporating too much noise
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pc_values) + 1), pc_values, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.show()

# looks like 4 components is what we want
# we may need to come back and look at 
df_pca = pca.apply_pca(full_sensor_one_df, predictor_columns, 5)
df_pca.head(1000)[['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']].plot()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_pca.copy()
NumAbs = NumericalAbstraction()

# how many values we want to look back
# 96 window size is one day. 4 window = 1 hour * 24 hours = 96 window
window_size = 96 

for col in predictor_columns:
  df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, 'mean')
  df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, 'std')

feb_subset = df_temporal[df_temporal['month'] == 2]

ax1 = feb_subset[['Temp - C', 'Temp - C_temp_std_ws_96', 'Temp - C_temp_mean_ws_96']].plot()
ax1.set_title('February - Temperature Analysis Temporal Abstraction Daily')
ax1.set_xlabel('Date & Time')
ax1.set_ylabel('Temperature (°C)')

march_subset = df_temporal[df_temporal['month'] == 3]

ax2 = march_subset[['Hum - %', 'Hum - %_temp_std_ws_96', 'Hum - %_temp_mean_ws_96']].plot()
ax2.set_title('March - Humidity Analysis Temporal Abstraction Daily')
ax2.set_xlabel('Date & Time')
ax2.set_ylabel('Humidity (%)')

april_subset = df_temporal[df_temporal['month'] == 4]

ax3 = april_subset[['THW Index - C', 'THW Index - C_temp_std_ws_96', 'THW Index - C_temp_mean_ws_96']].plot()
ax3.set_title('April - Temperature Analysis Temporal Abstraction Daily')
ax3.set_xlabel('Date & Time')
ax3.set_ylabel('THW Index (°C)')

may_subset = df_temporal[df_temporal['month'] == 5]

ax4 = may_subset[['Dew Point - C', 'Dew Point - C_temp_std_ws_96', 'Dew Point - C_temp_mean_ws_96']].plot()
ax4.set_title('May - Humidity Analysis Temporal Abstraction Daily')
ax4.set_xlabel('Date & Time')
ax4.set_ylabel('Dew Point (C)')

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_frequency = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# sampling rate is one hour
# window size is one day
sampling_rate = 4
window_size = 96

df_frequency = FreqAbs.abstract_frequency(df_frequency, ['Temp - C'], window_size, sampling_rate)

subset = df_frequency[df_frequency['week_of_year'] == 11]
subset[['Temp - C']].plot()
subset[[
  'Temp - C_max_freq',
  'Temp - C_freq_weighted',
  'Temp - C_pse',
  'Temp - C_freq_1.625_Hz_ws_96',
  'Temp - C_freq_2.0_Hz_ws_96'
]].plot()
subset[[
  'Temp - C_max_freq',
  'Temp - C_freq_weighted',
  'Temp - C_pse'
]].plot()
subset[[
  'Temp - C_freq_1.625_Hz_ws_96',
  'Temp - C_freq_2.0_Hz_ws_96'
]].plot()

df_freq_list = []
for col in predictor_columns:
  df_frequency = FreqAbs.abstract_frequency(df_frequency, [col], window_size, sampling_rate)
  df_freq_list.append(col + '_freq_1.625_Hz_ws_96')
  df_freq_list.append(col + '_freq_2.0_Hz_ws_96')

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
