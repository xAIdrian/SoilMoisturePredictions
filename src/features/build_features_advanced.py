import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
import matplotlib.pyplot as plt
from features.DataTransformation import PrincipalComponentAnalysis
from features.TemporalAbstraction import NumericalAbstraction
from config.config import set_config
set_config()

complete_meteo_sensor_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

# lagged_df['month'] = lagged_df.index.month
# lagged_df['day_of_week'] = lagged_df.index.dayofweek
# lagged_df['week_of_year'] = (lagged_df.index.dayofyear - 1) // 7 + 1

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = complete_meteo_sensor_df.copy()
pca = PrincipalComponentAnalysis()

# Paper reports ptimal amount of principle components is 5. 
# Let's test this

pc_values = pca.determine_pc_explained_variance(complete_meteo_sensor_df, df_pca.columns)

# elbow techniques to find best number of PCs
# capture the most variance without incorporating too much noise
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pc_values) + 1), pc_values, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.show()

# looks like 4 components is what we want
# we may need to come back and look at 
df_pca = pca.apply_pca(df_pca, df_pca.columns, 5)
df_pca.head(1000)[['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']].plot()
df_pca[['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5']].plot()

# --------------------------------------------------------------
# Temporal abstraction - NOT READY
# --------------------------------------------------------------

# df_temporal = df_pca.copy()
# NumAbs = NumericalAbstraction()

# # how many values we want to look back
# # 96 window size is one day. 4 window = 1 hour * 24 hours = 96 window
# window_size = 96 

# for col in predictor_columns:
#   df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, 'mean')
#   df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, 'std')

# # --------------------------------------------------------------
# # Plotting

# feb_subset = df_temporal[df_temporal['month'] == 2]

# ax1 = feb_subset[['Temp - C', 'Temp - C_temp_std_ws_96', 'Temp - C_temp_mean_ws_96']].plot()
# ax1.set_title('February - Temperature Analysis Temporal Abstraction Daily')
# ax1.set_xlabel('Date & Time')
# ax1.set_ylabel('Temperature (°C)')

# march_subset = df_temporal[df_temporal['month'] == 3]

# ax2 = march_subset[['Hum - %', 'Hum - %_temp_std_ws_96', 'Hum - %_temp_mean_ws_96']].plot()
# ax2.set_title('March - Humidity Analysis Temporal Abstraction Daily')
# ax2.set_xlabel('Date & Time')
# ax2.set_ylabel('Humidity (%)')

# april_subset = df_temporal[df_temporal['month'] == 4]

# ax3 = april_subset[['THW Index - C', 'THW Index - C_temp_std_ws_96', 'THW Index - C_temp_mean_ws_96']].plot()
# ax3.set_title('April - Temperature Analysis Temporal Abstraction Daily')
# ax3.set_xlabel('Date & Time')
# ax3.set_ylabel('THW Index (°C)')

# may_subset = df_temporal[df_temporal['month'] == 5]

# ax4 = may_subset[['Dew Point - C', 'Dew Point - C_temp_std_ws_96', 'Dew Point - C_temp_mean_ws_96']].plot()
# ax4.set_title('May - Humidity Analysis Temporal Abstraction Daily')
# ax4.set_xlabel('Date & Time')
# ax4.set_ylabel('Dew Point (C)')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_pca.to_pickle('../../data/interim/03_data_features.pkl')
