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
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from data.datetime_utils import set_datetime_as_index

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

# --------------------------------------------------------------
# We'll start by analyzing sensor 1 all on its own then we will
# come back to look at sensor 2
# --------------------------------------------------------------

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# remove the missing values
full_sensor_one_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
full_sensor_two_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = full_sensor_one_df.copy()
pca = PrincipalComponentAnalysis()

# Paper reports ptimal amount of principle components is 5. 
# Let's test this

pc_values = pca.determine_pc_explained_variance(full_sensor_one_df, full_sensor_one_df.columns)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pc_values) + 1), pc_values, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.show()

# looks like 4 components is what we want
# we may need to come back and look at 
df_pca = pca.apply_pca(full_sensor_one_df, full_sensor_one_df.columns, 4)
df_pca.head(100)[['pca_1', 'pca_2', 'pca_3', 'pca_4']].plot()

# elbow techniques to find best number of PCs
# capture the most variance without incorporating too much noise


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
