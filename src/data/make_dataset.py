import pandas as pd
import numpy as np
from glob import glob
import re

# let's get familiar with our dataset and see what we have
# with open('../../data/raw/accuweather_hourly_1.29.23_to_6.15.23.csv', 'r', encoding='utf-8', errors='ignore') as file:
#   single_file_accuweather = pd.read_csv(file)

# with open('../../data/raw/meteo_data_for_accuweather_comparison_1.30.23_to_6.15.23.csv', 'r', encoding='utf-8', errors='ignore') as file:
#   single_file_meteo_compare = pd.read_csv(file)

# with open('../../data/raw/meteo_data_for_model_1.30.23_to_7.31.23.csv', 'r', encoding='utf-8', errors='ignore') as file:
#   single_file_metea_model = pd.read_csv(file)

# with open('../../data/raw/Sensor1_ohms_data_1.21.23_to_7.31.23.csv', 'r', encoding='utf-8', errors='ignore') as file:
#   single_file_sensor1 = pd.read_csv(file)

# with open('../../data/raw/Sensor1_ohms_data_1.21.23_to_7.31.23.csv', 'r', encoding='utf-8', errors='ignore') as file:
#   single_file_sensor2 = pd.read_csv(file)

# # these can be removed later
# single_file_accuweather.columns
# single_file_meteo_compare.info()
# single_file_metea_model.columns

# single_file_sensor1.head()
# single_file_sensor2

# the beginning of processing all our files by filename
all_files = glob('../../data/raw/*.csv')
print(f"Total number of files: {len(all_files)}")

accuweather_df = pd.DataFrame()
meteo_compare_df = pd.DataFrame()
metea_model_df = pd.DataFrame()

sensor1_df = pd.DataFrame()
sensor2_df = pd.DataFrame()

for filename in all_files:
  curr_filename = filename.replace(' ', '_').replace('. ', '_')
  print(filename)

  with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
    if 'accuweather_hourly' in filename:
      accuweather_df = pd.read_csv(file)
    elif 'meteo_data_for_model' in filename:
      single_file_meteo_compare = pd.read_csv(file)
    elif 'meteo_data_for_accuweather_comparison' in filename:
      metea_model_df = pd.read_csv(file)  
    elif 'Sensor1' in filename:
      sensor1_df = pd.read_csv(file)
    elif 'Sensor2' in filename:
      sensor2_df = pd.read_csv(file)
    else:
      print(f'File {filename} is not being processed')

