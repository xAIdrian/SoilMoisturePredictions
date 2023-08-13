from pipeline.config import set_config
set_config()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from DataTransformation import PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

complete_meteo_sensor_df = pd.read_pickle('../../data/interim/02.5_outlier_safe_complete_datetime_df.pkl')

predictor_columns = list(['Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
       'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h',
       'Heat Index - C', 'THW Index - C', 'Rain - mm', 'Heating Degree Days',
       'Cooling Degree Days']    )  

# feature engineering
complete_meteo_sensor_df['month'] = complete_meteo_sensor_df.index.month
complete_meteo_sensor_df['day_of_week'] = complete_meteo_sensor_df.index.dayofweek
complete_meteo_sensor_df['week_of_year'] = (complete_meteo_sensor_df.index.dayofyear - 1) // 7 + 1


# --------------------------------------------------------------
# Let's Standardize The Whole Dataframe and Try Corrolation Again
# As a sanity check, we should see the same results
# --------------------------------------------------------------

scaled_df = complete_meteo_sensor_df.copy()

# Scale all of our features, 2 types
min_max_scaler = MinMaxScaler()

cols_to_minmax_scale_values = min_max_scaler.fit_transform(scaled_df[predictor_columns])
scaled_df[predictor_columns] = cols_to_minmax_scale_values

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
df_frequency.dropna(inplace=True)
# :: means every other row
# this is just another way to smooth out data
df_frequency = df_frequency.iloc[::2]

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_frequency.to_pickle('../../data/interim/03_data_features.pkl')
