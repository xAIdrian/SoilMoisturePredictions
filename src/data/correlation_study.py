import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

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

