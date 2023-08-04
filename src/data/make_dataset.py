import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# the beginning of processing all our files by filename
all_files = glob('../../data/raw/*.csv')
print(f"Total number of files: {len(all_files)}")

accuweather_df = pd.DataFrame()
accuweather_meteo_df = pd.DataFrame()
meteo_model_df = pd.DataFrame()

sensor1_df = pd.DataFrame()
sensor2_df = pd.DataFrame()

for filename in all_files:
  curr_filename = filename.replace(' ', '_').replace('. ', '_')

  with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
    if 'accuweather_hourly' in filename:
      accuweather_df = pd.read_csv(file)
    elif 'meteo_data_for_accuweather_comparison' in filename:
      accuweather_meteo_df = pd.read_csv(file)
    elif 'meteo_data_for_model' in filename:
      meteo_model_df = pd.read_csv(file)  
    elif 'Sensor1' in filename:
      sensor1_df = pd.read_csv(file)
    elif 'Sensor2' in filename:
      sensor2_df = pd.read_csv(file)
    else:
      print(f'File {filename} is not being processed')

# ---------------------------------------------------------------
# Validate Accuweather meets our 5% margin of error for accuracy
# ---------------------------------------------------------------

# Column cleaning. Strip white spaces
accuweather_df.columns = accuweather_df.columns.str.strip()
accuweather_meteo_df.columns = accuweather_meteo_df.columns.str.strip()

#
# Dates
#
# Set date column to proper datetime format
accuweather_df['Date & Time'] = pd.to_datetime(accuweather_df['Date & Time'], format='mixed', infer_datetime_format=True)
accuweather_meteo_df['Date & Time'] = pd.to_datetime(accuweather_meteo_df['Date & Time'], format='mixed', infer_datetime_format=True)

# standardize to the format: 'Year-Month-Day Hour:Minute:Second'
accuweather_df['Date & Time'] = accuweather_df['Date & Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
accuweather_meteo_df['Date & Time'] = accuweather_meteo_df['Date & Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

accuweather_df.set_index('Date & Time', inplace=True)
accuweather_meteo_df.set_index('Date & Time', inplace=True)

accuweather_df.sort_index(inplace=True)
accuweather_meteo_df.sort_index(inplace=True)

accuweather_df.index = pd.to_datetime(accuweather_df.index)
accuweather_meteo_df.index = pd.to_datetime(accuweather_meteo_df.index)

#
# Understand the Alignment of our Data Sources
#
accuweather_comp_columns = ['Temperature', 'Humidity (%)', 'DewPoint', 'Wind Speed (km/h)', 'WindGust  (km/h)', 'UVIndex', 'WetBulb', 'Pressure (mb)']
accuweather_meteo_comp_columns = ['Temp - C', 'Hum - %', 'Dew Point - C', 'Wind Speed - km/h', 'High Wind Speed - km/h', 'UV Index', 'Wet Bulb - C', 'Barometer - hPa']
# Use zip() to create pairs and dict() to convert these pairs to a dictionary
comp_dict =  dict(zip(accuweather_comp_columns, accuweather_meteo_comp_columns))

# start visually. TODO different graphs and drop irrelevant columns
accuweather_comp_hourly = accuweather_df[accuweather_comp_columns].resample('H').mean()
accuweather_meteo_comp_hourly = accuweather_meteo_df[accuweather_meteo_comp_columns].resample('H').mean()

# Replace all irregular values and NaN to smooth out dataset
accuweather_comp_hourly.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
accuweather_meteo_comp_hourly.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# loop through to save to file and then show the graph
for accuweather_col, accuweather_meteo_col in comp_dict.items():
  plt.figure(figsize=(20,10))
  accuweather_comp_hourly.head(400)[accuweather_col].plot(label='Accuweather')
  accuweather_meteo_comp_hourly.head(400)[accuweather_meteo_col].plot(label='Weather Station')
  plt.xlabel('Date & Time')
  plt.ylabel(accuweather_col)
  plt.title('Data Source Comparison')
  plt.legend()

  plt.savefig(f"../../reports/figures/accuweather_weather_station_comparison_{accuweather_col.replace('/', '')}.png")

  plt.show()

# calculate the percentage difference between the two data sources by feature
calculate_percent_difference_df = pd.DataFrame()
threshold_percentage_df = pd.DataFrame()

for accuweather_col, accuweather_meteo_col in comp_dict.items():
  difference_col = f"{accuweather_col} Percent Difference"
  threshold_column = f"{accuweather_col} Within Threshold"

  calculate_percent_difference_df[difference_col] = abs(accuweather_comp_hourly[accuweather_col].sub(accuweather_meteo_comp_hourly[accuweather_meteo_col]) / accuweather_comp_hourly[accuweather_col]) * 100
  calculate_percent_difference_df[difference_col] = calculate_percent_difference_df[difference_col].round(2)
  calculate_percent_difference_df.dropna(inplace=True)

  calculate_percent_difference_df[threshold_column] = calculate_percent_difference_df[difference_col] <= 5
  percentage_within_threshold = (calculate_percent_difference_df[threshold_column].mean() * 100).round(2)

  threshold_percentage_df[threshold_column] = [percentage_within_threshold]

print(threshold_percentage_df)
# -----------------------------------------------------------------
# Corrolation Research To Find The Best Features For Model Training
# -----------------------------------------------------------------

#
# Get Our Whole View Of Corrolations
#

# Calculating the correlation matrix for all columns
correlation_matrix = accuweather_df.corr(method='pearson')

# Plotting a heatmap for visualization
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Returning the correlation values with the target variable 'clAndijk'
correlation_with_clAndijk = correlation_matrix['clAndijk']
correlation_with_clAndijk.sort_values(ascending=False)

