import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
import numpy as np
from glob import glob
from data.data_utils import set_datetime_as_index, resistance_to_moisture
import data.remove_outliers as remove_outliers

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import set_config
set_config()

class FileLoader(BaseEstimator, TransformerMixin):
    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X=None):
        return self

    def transform(self, X=None):
        load_files = glob(self.pattern)

        if not load_files:
            raise ValueError(f"No files found for the pattern: {self.pattern}")

        load_file = load_files[0]

        df = pd.DataFrame()

        print(f"Loading: {load_file}")

        with open(load_file, 'r', encoding='utf-8', errors='ignore') as file:
          df = pd.read_csv(file)

        return df
    
class ColumnCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.columns = X.columns.str.strip()
        return X


class SimpleDateTimeIndexSetter(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = set_datetime_as_index(X, self.datetime_column)
        return X    

class ToPickleSaver(BaseEstimator, TransformerMixin):
    def __init__(self, pickle_name):
        self.pickle_name = pickle_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.to_pickle(self.pickle_name)
        return X
    
accuweather_pipeline = Pipeline([
    ('file_loader', FileLoader('../../data/raw/accuweather_hourly_1.29_to_6.15.csv')),
    ('column_cleaner', ColumnCleaner()),
    ('datetime_index_setter', SimpleDateTimeIndexSetter('Date & Time')),
    ('pickle_saver', ToPickleSaver('../../data/interim/01_accuweather_comparison_datetime_df.pkl'))
])  

meteo_data_pipeline = Pipeline([
    ('file_loader', FileLoader('../../data/raw/meteo_data_for_accuweather_comparison_1.30_to_6.15.csv')),
    ('column_cleaner', ColumnCleaner()),
    ('datetime_index_setter', SimpleDateTimeIndexSetter('Date & Time')),
    ('pickle_saver', ToPickleSaver('../../data/interim/01_accuweather_metero_comparison_datetime_df.pkl'))
])  

accuweather_df = accuweather_pipeline.fit_transform(X=None)
accuweather_meteo_df= meteo_data_pipeline.fit_transform(X=None)

correlation_pipeline = Pipeline([])

# --------------------------------------------------------------
# Prepare our comparison dataframes
# --------------------------------------------------------------

accuweather_df, accuweather_meteo_df = load_comparative_analysis_data()


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

