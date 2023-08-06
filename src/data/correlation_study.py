import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from build_features import set_datetime_as_index
import seaborn as sns
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

#
# Get Our Whole View Of Corrolations
#

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

# Merge the two sensor dataframes then merge that with weather data
sensor_merged = pd.merge(sensor1_resampled, sensor2_resampled, left_index=True, right_index=True)
final_data = pd.merge(meteo_model_resampled, sensor_merged, left_index=True, right_index=True, how='left')

# Calculating the correlation matrix for all columns
correlation_matrix = final_data.corr(method='pearson')

# Plotting a heatmap for visualization
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Returning the correlation values with the target variable 'clAndijk'
correlation_with_clAndijk = correlation_matrix['clAndijk']
correlation_with_clAndijk.sort_values(ascending=False)

