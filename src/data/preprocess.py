import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
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
    
class ColumnSpaceCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.columns = X.columns.str.strip()
        return X

class DateTimeIndexSetter(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = set_datetime_as_index(X, self.datetime_column)
        return X    

class Saver(BaseEstimator, TransformerMixin):
    def __init__(self, pickle_name, csv_name):
        self.pickle_name = pickle_name
        self.csv_name = csv_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.to_pickle(self.pickle_name)
        X.to_csv(self.csv_name)
        return X

# --------------------------------------------------------------
# Prepare our Correlation dataframes
# --------------------------------------------------------------

class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, old_col_name, new_col_name):
        self.old_col_name = old_col_name
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.rename(columns={ self.old_col_name: self.new_col_name }, inplace=True)
        return X
    
class SensorGroupBy(BaseEstimator, TransformerMixin):
    def __init__(self, group_by_column):
        self.group_by_column = group_by_column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.groupby(self.group_by_column).mean()
        return X
    
class Resampler(BaseEstimator, TransformerMixin):
    def __init__(self, resample_interval):
        self.resample_interval = resample_interval

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.resample(self.resample_interval).mean()
        # remove the null values
        X.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
        return X

#--------------------------------------------------------------
# Remove The highest values and remove outliers
#--------------------------------------------------------------

class ExtremeValueRemover(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, max_value):
        self.column_name = column_name
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[X[self.column_name] < self.max_value]
        return X

class RemoveOutliersWithLOF(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

      outlier_safe_df, outliers, X_scores = remove_outliers.mark_outliers_lof(X, self.columns)
      for column in self.columns:
        outlier_safe_df.loc[outlier_safe_df['outlier_lof'], column] = np.nan

      # remove the null values
      outlier_safe_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
      outlier_safe_df.drop('outlier_lof', axis=1, inplace=True)

      return outlier_safe_df    

def plot_outliers(source_df):
  moist_meteo_sensor_df = source_df.copy()
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

def preprocess_comparison_pipeline(accuweather_file_path, meteo_data_file_path):
    accuweather_pipeline = Pipeline([
      # ('file_loader', FileLoader('../../data/raw/accuweather_hourly_1.29_to_6.15.csv')),
      ('file_loader', FileLoader(accuweather_file_path)),
      ('column_cleaner', ColumnSpaceCleaner()),
      ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
      ('pickle_saver', Saver(
        '../../data/interim/01_accuweather_comparison_datetime_df.pkl',
        '../../data/processed/01_accuweather_comparison_datetime_df.csv'
      ))
    ])  

    meteo_data_pipeline = Pipeline([
        # ('file_loader', FileLoader('../../data/raw/meteo_data_for_accuweather_comparison_1.30_to_6.15.csv')),
        ('file_loader', FileLoader(meteo_data_file_path)),
        ('column_cleaner', ColumnSpaceCleaner()),
        ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
        ('pickle_saver', Saver(
          '../../data/interim/01_accuweather_metero_comparison_datetime_df.pkl',
          '../../data/processed/01_accuweather_metero_comparison_datetime_df.csv'
        ))
    ])  
    # accuweather_pipeline.fit_transform(X=None)
    # meteo_data_pipeline.fit_transform(X=None)
    return accuweather_pipeline, meteo_data_pipeline

def preprocess_pipeline(
        metero_data_file_path,
        sensor1_file_path,
        sensor2_file_path,
        plot=False
):
      
  meteo_model_pipeline = Pipeline([
      ('file_loader', FileLoader(metero_data_file_path)),
      ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
      ('resampler', Resampler('15T'))
  ])  

  sensor1_pipeline = Pipeline([
      ('file_loader', FileLoader(sensor1_file_path)),
      ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
      ('resampler', Resampler('15T'))
  ]) 

  sensor2_pipeline = Pipeline([
      ('file_loader', FileLoader(sensor2_file_path)),
      ('datetime_name_cleaner', ColumnNameCleaner('Date&Time', 'Date & Time')),
      ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
      ('resampler', Resampler('15T'))
  ]) 

  outlier_pipeline = Pipeline([
      ('sensor1_extreme_value_remover', ExtremeValueRemover('Sensor1 (Ohms)', 50000)),
      ('sensor2_extreme_value_remover', ExtremeValueRemover('Sensor2 (Ohms)', 50000)),
      ('remove_outliers_with_lof', RemoveOutliersWithLOF(['Sensor1 (Ohms)', 'Sensor2 (Ohms)'])),
      ('pickle_saver', Saver(
        f"../../data/interim/02_outlier_safe_complete_datetime_df.pkl",
        '../../data/processed/02_outlier_safe_complete_datetime_df.csv'
      ))
  ])

  meteo_model_df = meteo_model_pipeline.fit_transform(X=None)
  sensor1_df= sensor1_pipeline.fit_transform(X=None)        
  sensor2_df= sensor2_pipeline.fit_transform(X=None)     

  # Merge the two sensor dataframes then merge that with weather data
  sensor_merged = pd.merge(sensor1_df, sensor2_df, left_index=True, right_index=True)
  complete_meteo_sensor_df = pd.merge(meteo_model_df, sensor_merged, left_index=True, right_index=True, how='left')

  # remove the null values
  complete_meteo_sensor_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
  complete_meteo_sensor_df.to_pickle('../../data/interim/02_complete_datetime_df.pkl')
  complete_meteo_sensor_df.to_csv('../../data/processed/02_complete_datetime_df.csv')

  if plot:
    plot_outliers(complete_meteo_sensor_df)
  
  outlier_pipeline_output_path = '../../data/processed/02_outlier_safe_complete_datetime_df.csv'
  return outlier_pipeline, outlier_pipeline_output_path

if __name__ == "__main__":
    outlier_pipeline, outlier_pipeline_output_path = preprocess_pipeline(
        '../../data/raw/meteo_data for model 30.1.2023. - 31.7.2023..csv',
        '../../data/raw/Sensor1 data 21.1.2023..csv',
        '../../data/raw/Sensor2 data 21.1.2023..csv'
    )
    
    outlier_pipeline.fit_transform(
        X=pd.read_pickle('../../data/interim/02_complete_datetime_df.pkl')
    )

