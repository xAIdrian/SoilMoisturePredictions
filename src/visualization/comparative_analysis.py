import sys
import os

# Get the absolute path of the root directory (adjust according to your specific structure)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root path to sys.path
sys.path.append(root_path)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

accuweather_df = pd.read_pickle('../../data/interim/01_accuweather_comparison_datetime_df.pkl')
accuweather_meteo_df = pd.read_pickle('../../data/interim/01_accuweather_metero_comparison_datetime_df.pkl')

# plot raw data
plt.figure()
accuweather_df['Temperature'].plot(figsize=(20,10), label='Accuweather')
accuweather_meteo_df[:"2023-06-15 14:42:00"]['Temp - C'].plot(figsize=(20,10), label='Weather Station')   
plt.legend()
plt.title('At a Glance: Accuweather vs Weather Station Accuracy')

# --------------------------------------------------------------
# Understand the Alignment of our Data Sources
# --------------------------------------------------------------
accuweather_comp_columns = ['Temperature', 'Humidity (%)', 'DewPoint', 'Wind Speed (km/h)', 'WindGust  (km/h)', 'UVIndex', 'WetBulb', 'Pressure (mb)']
accuweather_meteo_comp_columns = ['Temp - C', 'Hum - %', 'Dew Point - C', 'Wind Speed - km/h', 'High Wind Speed - km/h', 'UV Index', 'Wet Bulb - C', 'Barometer - hPa']

# Use zip() to create pairs and dict() to convert these pairs to a dictionary
comp_dict =  dict(zip(accuweather_comp_columns, accuweather_meteo_comp_columns))

# start visually 
#resampling will gill up those slots for every hour
accuweather_comp_hourly = accuweather_df[accuweather_comp_columns].resample('H').mean()
accuweather_meteo_comp_hourly = accuweather_meteo_df[accuweather_meteo_comp_columns].resample('H').mean()

# Replace all irregular values and NaN to smooth out dataset
accuweather_comp_hourly.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
accuweather_meteo_comp_hourly.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

# scaling our pressure data to normalize comparison
scaler = MinMaxScaler()
accuweather_comp_hourly['Pressure (mb)'] = scaler.fit_transform(accuweather_comp_hourly[['Pressure (mb)']])
accuweather_meteo_comp_hourly['Barometer - hPa'] = scaler.fit_transform(accuweather_meteo_comp_hourly[['Barometer - hPa']])

# drop the last row of accuweather_comp_hourly to ensure size equality for calculation
accuweather_greater_row_num = accuweather_comp_hourly.shape[0] - accuweather_meteo_comp_hourly.shape[0]
accuweather_comp_hourly.drop(accuweather_comp_hourly.tail(accuweather_greater_row_num).index, inplace=True)

# ensure size equality for calculation
assert accuweather_comp_hourly.shape[0] == accuweather_meteo_comp_hourly.shape[0]

# calculate the percentage difference between the two data sources by feature
calculate_percent_difference_df = pd.DataFrame()
threshold_percentage_df = pd.DataFrame()

# histogram line graph
for accuweather_col, accuweather_meteo_col in comp_dict.items():
  # Calculate the percentage difference between the two data sources by feature
  difference_col = f"{accuweather_col} Percent Difference"
  threshold_column = f"{accuweather_col} Within Threshold"

  calculate_percent_difference_df[difference_col] = abs(accuweather_comp_hourly[accuweather_col].sub(accuweather_meteo_comp_hourly[accuweather_meteo_col]) / accuweather_comp_hourly[accuweather_col]) * 100
  
  calculate_percent_difference_df[difference_col] = calculate_percent_difference_df[difference_col].round(2)
  calculate_percent_difference_df.dropna(inplace=True)

  calculate_percent_difference_df[threshold_column] = calculate_percent_difference_df[difference_col] <= 5
  percentage_within_threshold = (calculate_percent_difference_df[threshold_column].mean() * 100).round(2)

  threshold_percentage_df[threshold_column] = [percentage_within_threshold]
  
  # Drawing the plot
  plt.figure(figsize=(30,10))

  accuweather_comp_hourly[accuweather_col].plot(label='Accuweather')
  accuweather_meteo_comp_hourly[accuweather_meteo_col].plot(label='Weather Station')
  
  plt.xlabel('Date & Time')
  plt.ylabel(accuweather_col)
  plt.title(f"{threshold_percentage_df[threshold_column].values[0]}% of {accuweather_col} values within 5% margin of error")
  plt.legend()

  plt.savefig(f"../../reports/figures/accuweather_weather_station_comparison_{accuweather_col.replace('/', '')}.png")

  plt.show()

  plt.figure(figsize=(30,10))

  accuweather_comp_hourly.head(500)[accuweather_col].plot(label='Accuweather')
  accuweather_meteo_comp_hourly.head(500)[accuweather_meteo_col].plot(label='Weather Station')
  
  plt.xlabel('Date & Time')
  plt.ylabel(accuweather_col)
  plt.title(f"Zoomed In to Month: {threshold_percentage_df[threshold_column].values[0]}% of {accuweather_col} values within 5% margin of error")
  plt.legend()

  plt.savefig(f"../../reports/figures/zoom_400_accuweather_weather_station_comparison_{accuweather_col.replace('/', '')}.png")

  plt.show()
