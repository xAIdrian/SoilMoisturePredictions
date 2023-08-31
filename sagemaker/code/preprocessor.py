
import os
import numpy as np
import json
import numpy as np
import tempfile
from pathlib import Path
from constants import *
from pickle import dump
import pandas as pd
from glob import glob
from data.data_utils import set_datetime_as_index
import data.remove_outliers as remove_outliers

from config.config import set_config
set_config()

from sagemaker.inputs import FileSystemInput
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from sagemaker.processing import ProcessingInput, ProcessingOutput
# from sagemaker.sklearn.processing import SKLearnProcessor
# from sagemaker.workflow.steps import ProcessingStep
# from sagemaker.workflow.parameters import ParameterString
# from sagemaker.workflow.pipeline import Pipeline
# from sagemaker.workflow.steps import CacheConfig
# from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig

BASE_DIRECTORY = "/opt/ml/processing"
WEATHER_DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "meteo_weather.csv"
SENSOR1_DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "sensor1.csv"
SENSOR2_DATA_FILEPATH = Path(BASE_DIRECTORY) / "input" / "sensor2.csv"

def _save_splits(base_directory, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)    
    
def _save_source_dataframe(base_directory, dataframe):
    """
    We will take a complete dataframe prior to the test train split and save it to the directory
    """
    data_path = Path(base_directory) / "source"
    data_path.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(data_path / "source.csv", header=False, index=False)
    
def _save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))

def _save_classes(base_directory, classes):
    """
    Saves the list of classes from the dataset. 
    We will need this if we ever want to use a LabelEncoder.
    """
    path = Path(base_directory) / "classes"
    path.mkdir(parents=True, exist_ok=True)

    np.asarray(classes).tofile(path / "classes.csv", sep=",")

def _save_baseline(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need a baseline
    to compute constraints and statistics. This function will save that
    baseline to the disk.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()
        df.to_json(
            baseline_path / f"{split}-baseline.json", orient="records", lines=True
        )
        

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

# --------------------------------------------------------------
# Prepare our Correlation dataframes
# --------------------------------------------------------------
    
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
    
class Saver(BaseEstimator, TransformerMixin):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _save_source_dataframe(self.base_dir, X)  
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

def preprocess_pipeline(
    base_directory,
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
      ('datetime_index_setter', DateTimeIndexSetter('Date & Time')),
      ('resampler', Resampler('15T'))
  ]) 

  outlier_pipeline = Pipeline([
      ('sensor1_extreme_value_remover', ExtremeValueRemover('Sensor1 (Ohms)', 50000)),
      ('sensor2_extreme_value_remover', ExtremeValueRemover('Sensor2 (Ohms)', 50000)),
      ('remove_outliers_with_lof', RemoveOutliersWithLOF(['Sensor1 (Ohms)', 'Sensor2 (Ohms)'])),
      ('source_saver', Saver(base_directory))
  ])

  meteo_model_df = meteo_model_pipeline.fit_transform(X=None)
  sensor1_df= sensor1_pipeline.fit_transform(X=None)        
  sensor2_df= sensor2_pipeline.fit_transform(X=None)     

  # Merge the two sensor dataframes then merge that with weather data
  sensor_merged = pd.merge(sensor1_df, sensor2_df, left_index=True, right_index=True)
  complete_meteo_sensor_df = pd.merge(meteo_model_df, sensor_merged, left_index=True, right_index=True, how='left')

  # remove the null values
  complete_meteo_sensor_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
  outlier_pipeline.fit_transform(X=complete_meteo_sensor_df)  
    
  _save_pipeline(base_directory, outlier_pipeline)

  if plot:
    plot_outliers(complete_meteo_sensor_df)
  
if __name__ == "__main__":
  preprocess_pipeline(
      BASE_DIRECTORY,
      WEATHER_DATA_FILEPATH,
      SENSOR1_DATA_FILEPATH,
      SENSOR2_DATA_FILEPATH
  )      
