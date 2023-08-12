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
from sklearn.cluster import KMeans
from data.datetime_utils import set_datetime_as_index
from DataTransformation import PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100

complete_meteo_sensor_df = pd.read_pickle('../../data/interim/02_complete_meteo_sensors_datetime_df.pkl')

predictor_columns = list(['Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
       'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h',
       'Heat Index - C', 'THW Index - C', 'Rain - mm', 'Heating Degree Days',
       'Cooling Degree Days']    )  

# feature engineering
complete_meteo_sensor_df['month'] = complete_meteo_sensor_df.index.month
complete_meteo_sensor_df['day_of_week'] = complete_meteo_sensor_df.index.dayofweek
complete_meteo_sensor_df['week_of_year'] = (complete_meteo_sensor_df.index.dayofyear - 1) // 7 + 1

# proper scaling of all values 
scaled_df = complete_meteo_sensor_df.copy()

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
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = scaled_df.copy()
pca = PrincipalComponentAnalysis()

# Paper reports ptimal amount of principle components is 5. 
# Let's test this

pc_values = pca.determine_pc_explained_variance(complete_meteo_sensor_df, predictor_columns)

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
df_pca = pca.apply_pca(complete_meteo_sensor_df, predictor_columns, 5)
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

# --------------------------------------------------------------
# Plotting

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
for week in df_frequency['week_of_year'].unique():
  print(f"Applying Fourier Transformation to Week #{week}")
  curr_subset = df_frequency[df_frequency['week_of_year'] == week].reset_index(drop=True).copy()
  curr_subset = FreqAbs.abstract_frequency(curr_subset, predictor_columns, window_size, sampling_rate)
  df_freq_list.append(curr_subset)

df_frequency = pd.concat(df_freq_list).set_index('Date & Time', drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows to reduce overfitting
# use the 50% window overlap rule. allowed with larger datasets
# --------------------------------------------------------------

df_frequency.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# :: means every other row
df_frequency = df_frequency.iloc[::2]

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_frequency.to_pickle('../../data/interim/03_data_features.pkl')
