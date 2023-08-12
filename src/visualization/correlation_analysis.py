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

complete_meteo_sensor_df = pd.read_pickle('../../data/interim/02_complete_meteo_sensors_datetime_df.pkl')

# Compare our sensor measures
plt.plot(complete_meteo_sensor_df['Sensor1 (Ohms)'], label='Sensor 1')
plt.plot(complete_meteo_sensor_df['Sensor2 (Ohms)'], label='Sensor 2')
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

outlier_columns = list(complete_meteo_sensor_df.columns)

# IQR method
for col in outlier_columns:
  iqr_outlier_plot_dataset = remove_outliers.mark_outliers_iqr(complete_meteo_sensor_df, col)
  remove_outliers.plot_binary_outliers(iqr_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

for col in outlier_columns:
  chauvenet_outlier_plot_dataset = remove_outliers.mark_outliers_chauvenet(complete_meteo_sensor_df, col)
  remove_outliers.plot_binary_outliers(chauvenet_outlier_plot_dataset, col, outlier_col=col + '_outlier', reset_index=True)

lof_outlier_plot_dataset, outliers, X_scores = remove_outliers.mark_outliers_lof(complete_meteo_sensor_df, outlier_columns)
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
decomposition_sorted = seasonal_decompose(complete_meteo_sensor_df['Sensor1 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

decomposition_sorted = seasonal_decompose(complete_meteo_sensor_df['Sensor2 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

# --------------------------------------------------------------
# Plotting a heatmap for visualization of correlation
# --------------------------------------------------------------

# Calculating the correlation matrix for all columns
corr_matrix_sensor_df = complete_meteo_sensor_df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_sensor_df, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Returning the correlation values with the target variable 
correlation_with_sensor1 = corr_matrix_sensor_df['Sensor1 (Ohms)']
correlation_with_sensor2 = corr_matrix_sensor_df['Sensor2 (Ohms)']

correlation_df = pd.DataFrame([correlation_with_sensor1, correlation_with_sensor2]).T
correlation_df.sort_values(by=['Sensor1 (Ohms)', 'Sensor2 (Ohms)'], ascending=False)



