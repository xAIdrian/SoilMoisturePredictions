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
from statsmodels.tsa.seasonal import seasonal_decompose
import data.remove_outliers as remove_outliers

# -----------------------------------------------------------------
# Corrolation Research To Find The Best Features For Model Training
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

# --------------------------------------------------------------
# Get Our Whole View Of Corrolations
# --------------------------------------------------------------

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

# Compare our sensor measures
plt.plot(full_sensor_one_df['Sensor1 (Ohms)'], label='Sensor 1')
plt.plot(full_sensor_two_df['Sensor2 (Ohms)'], label='Sensor 2')
plt.xlabel('Sensors')
plt.ylabel('Count')
plt.title('Comparison of Two Sensor Features')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Removing Outliers
# --------------------------------------------------------------

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100

outlier_columns = list(full_sensor_one_df.columns)

# IQR method
for col in outlier_columns:
  iqr_outlier_plot_dataset = remove_outliers.mark_outliers_iqr(full_sensor_one_df, col)
  remove_outliers.plot_binary_outliers(iqr_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

for col in outlier_columns:
  chauvenet_outlier_plot_dataset = remove_outliers.mark_outliers_chauvenet(full_sensor_one_df, col)
  remove_outliers.plot_binary_outliers(chauvenet_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

lof_outlier_plot_dataset, outliers, X_scores = remove_outliers.mark_outliers_lof(full_sensor_one_df, outlier_columns)
for col in outlier_columns:
  remove_outliers.plot_binary_outliers(dataset=lof_outlier_plot_dataset, col=col, outlier_col="outlier_lof", reset_index=True)

# --------------------------------------------------------------
# Find additional insights through seasonal decomposition
# --------------------------------------------------------------

# this is good but takes a long time to load
# print('loading pairplot...')
# sns.pairplot(full_sensor_one_df.head(450), kind='reg')
# sns.pairplot(full_sensor_two_df.head(450), kind='reg')

# Applying seasonal decomposition to the sorted 'Sensor' series
# Using a seasonal frequency of 365 days
decomposition_sorted = seasonal_decompose(full_sensor_one_df['Sensor1 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

decomposition_sorted = seasonal_decompose(full_sensor_two_df['Sensor2 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

# --------------------------------------------------------------
# Plotting a heatmap for visualization of correlation
# --------------------------------------------------------------

# Calculating the correlation matrix for all columns
corr_matrix_sensor_one = full_sensor_one_df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_sensor_one, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap Sensor 1")
plt.show()

corr_matrix_sensor_two = full_sensor_two_df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_sensor_two, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap Sensor 2")
plt.show()

# Returning the correlation values with the target variable 
correlation_with_sensor1 = corr_matrix_sensor_one['Sensor1 (Ohms)']
correlation_with_sensor2 = corr_matrix_sensor_two['Sensor2 (Ohms)']

correlation_df = pd.DataFrame([correlation_with_sensor1, correlation_with_sensor2]).T
correlation_df.sort_values(by=['Sensor1 (Ohms)', 'Sensor2 (Ohms)'], ascending=False)



