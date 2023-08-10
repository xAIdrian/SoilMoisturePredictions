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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import data.remove_outliers as remove_outliers

# --------------------------------------------------------------
# Let's Standardize The Whole Dataframe and Try Corrolation Again
# As a sanity check, we should see the same results
# --------------------------------------------------------------
scaled_df = final_df.copy()

# just the column names
cols_to_minmax_scale = ['Barometer - hPa']
cols_to_standard_scale = [col for col in scaled_df.columns if col not in cols_to_minmax_scale]

# Scale all of our features, 2 types
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

cols_to_minmax_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_minmax_scale])
scaled_df[cols_to_minmax_scale] = cols_to_minmax_scale_values

cols_to_standard_scale_values = min_max_scaler.fit_transform(scaled_df[cols_to_standard_scale])
scaled_df[cols_to_standard_scale] = cols_to_standard_scale_values

# Plotting a heatmap for visualization
scaled_correlation_matrix = scaled_df.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(scaled_correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Scaled Correlation Heatmap")
plt.show()
