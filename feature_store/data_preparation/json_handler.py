import pandas as pd
import numpy as np

import orjson

def get_json_value(df: pd.DataFrame, col: str):
    """
    Takes a pandas dataframe and a string column-name.
    Extracts json object from specified column in dataframe.
    Returns original dataframe joined with normalized json as columns.
    Args:
        df (DataFrame): Dataframe to extract json col from
        col (str): Column name with json data.
    
    Returns:
        DataFrame: Data with json column normalized
    """

    try:
        df = df.copy()
    except Exception as e:
        raise(e)

    try:
        data = pd.json_normalize(
            df[col].apply(
                orjson.loads), max_level=0)
    except KeyError as e:
        return df
    else:
        col_lst = data.columns.difference(df.columns)
        return df.join(data[col_lst])

def extract_value_dict(data: dict, key: str, default=np.nan):
    """
    Function receives dictionary with key string
    and returns value. If default is provided, returns
    default value when key does not exist, otherwise returns nan.
    Args:
        data (dict): Dictionary to extract value
        key (str): Key to exctract value
        default (np.nan): default value to provide if key not present in dict
    Returns:

    """

    try:
        status = data.get(key, default)
    except AttributeError as e:
        raise(e)

    return status

def map_normalize_dict(df: pd.DataFrame, col: str, map:dict):
    """
    Receives Pandas dataframe, column name and
    dictionary containing new column names as keys and dict
    keys as values. Normalizes dict column in dataframe and returns
    original dataframe with new columns.
    Args:
        df (DataFrame): Dictionary to extract value
        key (str): Key to exctract value
        default (np.nan): default value to provide if key not present in dict
    Returns:
        DataFrame: Data with json normalized as new columns
    """
    df = df.copy()

    for new_col_name, dict_key in map.items():
        df.loc[:, new_col_name] = df[col].apply(
            lambda x: extract_value_dict(
                data=x, key=dict_key))
                
    return df.fillna(value=np.nan)