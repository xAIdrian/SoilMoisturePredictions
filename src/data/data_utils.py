import pandas as pd

def set_datetime_as_index(dataframe: pd.DataFrame, datetime_column: str):
    dataframe[datetime_column] = pd.to_datetime(dataframe[datetime_column], format='mixed', infer_datetime_format=True)

    # standardize to the format: 'Year-Month-Day Hour:Minute:Second'
    dataframe[datetime_column] = dataframe[datetime_column].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    dataframe.set_index(datetime_column, inplace=True)

    dataframe.sort_index(inplace=True)

    # update the index to be datetime
    dataframe.index = pd.to_datetime(dataframe.index)

    return dataframe

def resistance_to_moisture(resistance):
    return 100 - ((resistance - 2000) / (50000 - 2000)) * 100

