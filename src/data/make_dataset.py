import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# the beginning of processing all our files by filename
all_files = glob('../../data/raw/*.csv')
print(f"Total number of files: {len(all_files)}")

accuweather_df = pd.DataFrame()
accuweather_meteo_df = pd.DataFrame()
metea_model_df = pd.DataFrame()

sensor1_df = pd.DataFrame()
sensor2_df = pd.DataFrame()

for filename in all_files:
  curr_filename = filename.replace(' ', '_').replace('. ', '_')
  print(filename)

  with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
    if 'accuweather_hourly' in filename:
      accuweather_df = pd.read_csv(file)
    elif 'meteo_data_for_accuweather_comparison' in filename:
      accuweather_meteo_df = pd.read_csv(file)
    elif 'meteo_data_for_model' in filename:
      metea_model_df = pd.read_csv(file)  
    elif 'Sensor1' in filename:
      sensor1_df = pd.read_csv(file)
    elif 'Sensor2' in filename:
      sensor2_df = pd.read_csv(file)
    else:
      print(f'File {filename} is not being processed')

# ---------------------------------------------------------------
# Validate Accuweather meets our 5% margin of error for accuracy
# ---------------------------------------------------------------

print(f"Accuweather: { accuweather_df.columns }")
print(f"Accuweather Meteo Baseline: { accuweather_meteo_df.columns }")

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

plt.figure(figsize=(20,10))

# plot df1
plt.plot(accuweather_df.index, accuweather_df['Temperature'], label='Temperature 1')

# plot df2
plt.plot(accuweather_meteo_df.index, accuweather_meteo_df['Temp - C'], label='Temperature 2')

plt.xlabel('Date & Time')
plt.ylabel('Temperature')
plt.title('Temperature Comparison')
plt.legend()
plt.show()


#
# Understand the Weather Data
#
