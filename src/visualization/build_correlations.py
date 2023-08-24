import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pipeline.config import set_config
set_config()

# -----------------------------------------------------------------
# Corrolation Research To Find The Best Features For Model Training
# -----------------------------------------------------------------

moist_complete_meteo_sensor_df = pd.read_pickle('../../data/interim/02_outlier_safe_complete_datetime_df.pkl')

# Compare our sensor measures
plt.plot(moist_complete_meteo_sensor_df['Sensor1 (Ohms)'], label='Sensor 1')
plt.plot(moist_complete_meteo_sensor_df['Sensor2 (Ohms)'], label='Sensor 2')
plt.xlabel('Sensors')
plt.ylabel('Ohms Count')
plt.title('Comparison of Two Sensor Raw')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Find additional insights through seasonal decomposition
# --------------------------------------------------------------

# this is good but takes a long time to load
# print('loading pairplot...')
# sns.pairplot(full_sensor_one_df.head(450), kind='reg')
# sns.pairplot(full_sensor_two_df.head(450), kind='reg')

# Applying seasonal decomposition to the sorted 'Sensor' series
# Using a seasonal frequency of 365 days
decomposition_sorted = seasonal_decompose(moist_complete_meteo_sensor_df['Sensor1 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

decomposition_sorted = seasonal_decompose(moist_complete_meteo_sensor_df['Sensor2 (Ohms)'], period=3000, model='additive')
decomposition_plot_sorted = decomposition_sorted.plot()
decomposition_plot_sorted.set_size_inches(15, 12)
plt.show()

# --------------------------------------------------------------
# Plotting a heatmap for visualization of correlation
# --------------------------------------------------------------

# Calculating the correlation matrix for all columns
corr_matrix_sensor_df = moist_complete_meteo_sensor_df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_sensor_df, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Returning the correlation values with the target variable 
correlation_with_sensor1 = corr_matrix_sensor_df['Sensor1 (Ohms)']
correlation_with_sensor2 = corr_matrix_sensor_df['Sensor2 (Ohms)']

correlation_df = pd.DataFrame([correlation_with_sensor1, correlation_with_sensor2]).T
correlation_df.sort_values(by=['Sensor1 (Ohms)', 'Sensor2 (Ohms)'], ascending=False)

def color_high(val):
    
    if val < -0.5:                     
        return 'background: pink'
    
    elif val > 0.5:
        return 'background: skyblue'   
    
corr_matrix_sensor_df.T.style.applymap(color_high)

