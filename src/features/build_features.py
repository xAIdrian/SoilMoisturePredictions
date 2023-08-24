import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
from pipeline.config import set_config
set_config()

moist_complete_meteo_sensor_df = pd.read_pickle('../../data/interim/02_outlier_safe_complete_datetime_df.pkl')

#--------------------------------------------------------------
# Understanding a lag in our time series. Narrow scope, we try
# 1, 3, 5, 7, 10 day features
# We'll start with 1 & 7 days
# --------------------------------------------------------------

lagged_df = moist_complete_meteo_sensor_df.copy()

core_feature_columns = ['Barometer - hPa', 'Hum - %', 'Dew Point - C', 'Wet Bulb - C']
# 96, 15 minute intervals in a day
for col in core_feature_columns:
    lagged_df[f"{col}_lag_1_day"] = lagged_df[col].shift(96)

# 672, 15 minute intervals in a week
for col in core_feature_columns:
    moist_complete_meteo_sensor_df[f"{col}_lag_7_days"] = lagged_df[col].shift(672)

# ---------------------------------------------------------
# Cleaning our Data
# ---------------------------------------------------------
# data_cleaned = action_df.dropna()
data_cleaned = lagged_df.dropna()

# Confirming that there are no remaining missing values in the cleaned dataset
missing_values_after_removal = data_cleaned.isnull().any().any()
num_rows_removed = lagged_df.shape[0] - data_cleaned.shape[0]

missing_values_after_removal, num_rows_removed, data_cleaned.shape

if missing_values_after_removal == False:
  scaled_df = data_cleaned
else:
  raise Exception('Missing values still exist in the dataset')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_cleaned.to_pickle('../../data/interim/03_data_features.pkl')
