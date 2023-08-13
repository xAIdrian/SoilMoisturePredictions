from pipeline.config import set_config
set_config()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

efficient_feature_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

# drop columns we will not use at all
s1_drop_columns_for_training = ['Sensor1 (Ohms)', 'Sensor2 (Ohms)', 'Sensor2 Moisture (%)']
s1_df_train = efficient_feature_df.drop(s1_drop_columns_for_training, axis=1)

# drop columns we'll predict later
s1_X = s1_df_train.drop(['Sensor1 Moisture (%)', ], axis=1)
s1_y = s1_df_train[['Sensor1 Moisture (%)']]

# providing seed. we take control of the stochastic split to reproduce
s1_X_train, s1_X_test, s1_y_train, s1_y_test = train_test_split(s1_X, s1_y, test_size=0.25, random_state=42)


