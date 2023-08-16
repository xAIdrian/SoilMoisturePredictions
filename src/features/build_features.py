import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

# --------------------------------------------------------------
# Feature Interactions
# --------------------------------------------------------------

action_df = lagged_df.copy()
# Interaction between Temperature and Humidity (Likely to Influence Soil Moisture)
action_df['temp_hum_interaction'] = action_df['Temp - C'] * action_df['Hum - %']
# Barometer and Temperature Interaction (Pressure and Temperature Correlation)
action_df['barometer_temp_interaction'] = action_df['Barometer - hPa'] * action_df['Temp - C']
# Wind Speed and Temperature Interaction (Possible Influence on Evaporation)
action_df['wind_temp_interaction'] = action_df['Wind Speed - km/h'] * action_df['Temp - C']
# Dew Point and Humidity Interaction (Moisture Interaction)
action_df['dew_hum_interaction'] = action_df['Dew Point - C'] * action_df['Hum - %']
# Heating and Cooling Degree Days Interaction with Temperature (Energy Considerations)
action_df['heat_cool_interaction'] = action_df['Heating Degree Days'] * action_df['Cooling Degree Days'] * action_df['Temp - C']
# Rain and Wind Speed Interaction (Weather Condition Correlation)
action_df['rain_wind_interaction'] = action_df['Rain - mm'] * action_df['Wind Speed - km/h']
# Sensor Readings (Ohms) Interaction with Temperature and Humidity (Soil Condition)
action_df['sensor1_temp_hum_interaction'] = action_df['Sensor1 (Ohms)'] * action_df['Temp - C'] * action_df['Hum - %']
action_df['sensor2_temp_hum_interaction'] = action_df['Sensor2 (Ohms)'] * action_df['Temp - C'] * action_df['Hum - %']
# Overall Moisture Index (Combining Soil and Atmospheric Moisture)
action_df['overall_moisture_index'] = (action_df['Sensor1 Moisture (%)'] + action_df['Sensor2 Moisture (%)'])/2 * action_df['Hum - %']


# --------------------------------------------------------------
# Let's Normalilze The Whole Dataframe and Try Corrolation Again
# As a sanity check, we should see the same results
# --------------------------------------------------------------

scaled_df = action_df.copy()
datetime = scaled_df.index

# Scale all of our features, 2 types
scaler = MinMaxScaler()

# Fit and transform the scaler on the DataFrame
scaled_values = scaler.fit_transform(scaled_df)
# Convert back to DataFrame w/ datetime index
scaled_df = pd.DataFrame(scaled_values, columns=scaled_df.columns)
scaled_df.index = datetime

# Checking for any remaining missing values in the dataset
missing_values_remaining = scaled_df.isnull().sum().sort_values(ascending=False)
missing_values_remaining[missing_values_remaining > 0]

missing_values_after_scale = scaled_df.isnull().any().any()
missing_values_after_scale

# Removing rows with any missing values associated with LAG
data_cleaned = scaled_df.dropna()

# Confirming that there are no remaining missing values in the cleaned dataset
missing_values_after_removal = data_cleaned.isnull().any().any()
num_rows_removed = scaled_df.shape[0] - data_cleaned.shape[0]

missing_values_after_removal, num_rows_removed, data_cleaned.shape

if missing_values_after_removal == False:
  scaled_df = data_cleaned
else:
  raise Exception('Missing values still exist in the dataset')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

scaled_df.to_pickle('../../data/interim/03_data_features.pkl')
