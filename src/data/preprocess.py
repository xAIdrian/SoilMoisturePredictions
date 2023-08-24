import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
import numpy as np
from glob import glob
from data.data_utils import set_datetime_as_index, resistance_to_moisture
import data.remove_outliers as remove_outliers
from pipeline.config import set_config
set_config()

def load_comparative_analysis_data():
  accuweather_df = pd.DataFrame()
  accuweather_meteo_df = pd.DataFrame()

  # the beginning of processing all our files by filename
  all_files = glob('../../data/raw/*.csv')
  print(f"Total number of files: {len(all_files)}")

  for filename in all_files:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
      if 'accuweather_hourly' in filename:
        accuweather_df = pd.read_csv(file)
      elif 'meteo_data_for_accuweather_comparison' in filename:
        accuweather_meteo_df = pd.read_csv(file)
      else:
        print(f'File {filename} is not being processed') 

  return accuweather_df, accuweather_meteo_df       

def load_correlation_analysis_data():     
  meteo_model_df = pd.DataFrame()
  sensor1_df = pd.DataFrame()
  sensor2_df = pd.DataFrame()

  all_files = glob('../../data/raw/*.csv')
  print(f"Total number of files: {len(all_files)}")

  for filename in all_files:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
      if 'meteo_data for model' in filename:
        meteo_model_df = pd.read_csv(file)
      elif 'Sensor1' in filename:
        sensor1_df = pd.read_csv(file)
      elif 'Sensor2' in filename:
        sensor2_df = pd.read_csv(file)

  return meteo_model_df, sensor1_df, sensor2_df

# --------------------------------------------------------------
# Prepare our comparison dataframes
# --------------------------------------------------------------

accuweather_df, accuweather_meteo_df = load_comparative_analysis_data()

# Column cleaning. Strip white spaces
accuweather_df.columns = accuweather_df.columns.str.strip()
accuweather_meteo_df.columns = accuweather_meteo_df.columns.str.strip()

# Set date column to proper datetime format
accuweather_df = set_datetime_as_index(accuweather_df, 'Date & Time')
accuweather_meteo_df = set_datetime_as_index(accuweather_meteo_df, 'Date & Time')

# Save to pickle file
accuweather_df.to_pickle('../../data/interim/01_accuweather_comparison_datetime_df.pkl')
accuweather_meteo_df.to_pickle('../../data/interim/01_accuweather_metero_comparison_datetime_df.pkl')

# --------------------------------------------------------------
# Prepare our Correlation dataframes
# --------------------------------------------------------------

meteo_model_df, sensor1_df, sensor2_df = load_correlation_analysis_data()

sensor2_df.rename(columns={'Date&Time': 'Date & Time'}, inplace=True)

# set dates as index
meteo_model_df = set_datetime_as_index(meteo_model_df, 'Date & Time')
sensor1_df = set_datetime_as_index(sensor1_df, 'Date & Time')
sensor2_df = set_datetime_as_index(sensor2_df, 'Date & Time')

sensor1_df = sensor1_df.groupby('Date & Time').mean()
sensor2_df = sensor2_df.groupby('Date & Time').mean()

# resampling to 15 min intervals. setting the average within each 15 min bucket
# we could use reset_index() to get the date & time back as a column
meteo_model_resampled = meteo_model_df.resample('15T').mean()
sensor1_resampled = sensor1_df.resample('15T').mean()
sensor2_resampled = sensor2_df.resample('15T').mean()

# Merge the two sensor dataframes then merge that with weather data
sensor_merged = pd.merge(sensor1_resampled, sensor2_resampled, left_index=True, right_index=True)
complete_meteo_sensor_df = pd.merge(meteo_model_resampled, sensor_merged, left_index=True, right_index=True, how='left')

# remove the null values
complete_meteo_sensor_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

#--------------------------------------------------------------
# Remove The highest values and set moisture
#--------------------------------------------------------------

pre_clean_df = complete_meteo_sensor_df.copy()

# Remove the highest values
moist_meteo_sensor_df = pre_clean_df[pre_clean_df['Sensor1 (Ohms)'] < 50000]
moist_meteo_sensor_df = moist_meteo_sensor_df[moist_meteo_sensor_df['Sensor2 (Ohms)'] < 50000]

# --------------------------------------------------------------
# Removing Outliers
# --------------------------------------------------------------

outlier_columns = list(moist_meteo_sensor_df.columns)

print('IQR method')
for col in outlier_columns:
  iqr_outlier_plot_dataset = remove_outliers.mark_outliers_iqr(moist_meteo_sensor_df, col)
  remove_outliers.plot_binary_outliers(iqr_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

print('Chauvenet method')
for col in outlier_columns:
  chauvenet_outlier_plot_dataset = remove_outliers.mark_outliers_chauvenet(moist_meteo_sensor_df, col)
  remove_outliers.plot_binary_outliers(chauvenet_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

print('LOF method')
lof_outlier_plot_dataset, outliers, X_scores = remove_outliers.mark_outliers_lof(moist_meteo_sensor_df, outlier_columns)
for col in outlier_columns:
  remove_outliers.plot_binary_outliers(dataset=lof_outlier_plot_dataset, col=col, outlier_col="outlier_lof", reset_index=True)

# We will choose LOF and remove the outliers
real_outlier_columns = ['Sensor1 (Ohms)', 'Sensor2 (Ohms)']

outlier_safe_df, outliers, X_scores = remove_outliers.mark_outliers_lof(moist_meteo_sensor_df, real_outlier_columns)
for column in real_outlier_columns:
  outlier_safe_df.loc[outlier_safe_df['outlier_lof'], column] = np.nan

# remove the null values
outlier_safe_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
outlier_safe_df.drop('outlier_lof', axis=1, inplace=True)

outlier_safe_df.to_pickle('../../data/interim/02_outlier_safe_complete_datetime_df.pkl')

