# -*- coding: utf-8 -*-
import os, sys
import click
import logging
import pandas as pd
import numpy as np
import os, datetime
from pathlib import Path


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include='category').columns
    for col in categorical_cols:
        encoded_cols = pd.get_dummies(df[col], prefix=col, drop_first=False) 
        df = pd.concat([df, encoded_cols], axis=1)
        df = df.drop(columns=[col])
    
    return df

def extract_datetime(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Extracts the year, month, day of the month, weekday, and hour from the datetime column of a dataframe
    
    Args:
    df (pd.DataFrame): dataframe containing the datetime column
    
    Returns:
    pd.DataFrame: dataframe with new columns for year, month, day, weekday, and hour
    '''
    # Convert the datetime column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Extract the year, month, day, weekday, and hour columns
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    # Drop the original datetime column
    df.drop('datetime', axis=1, inplace=True)
    
    return df

def select_variables(X_train):
    X_train_regression = X_train[["season", "holiday", "workingday", "weather", "time_range",  "temp", "atemp", "humidity", "windspeed"]]
    print(X_train_regression.dtypes)
    for i in ["season","holiday","workingday","weather"]: 
        X_train_regression[i] = X_train_regression[i].astype("category")

    X_train_regression= one_hot_encode(X_train_regression)
    return X_train_regression

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_train= pd.read_csv(os.path.join(input_filepath,"train.csv"))
    data_test=pd.read_csv(os.path.join(input_filepath,"test.csv"))

    X_train= data_train.iloc[:,:-1]
    Y_train= data_train['count']    
    X_train = extract_datetime(X_train)

    time_intervals = [-1, 6, 10,15, 20, 24]
    # Create a column of time ranges
    X_train['time_range'] = pd.cut(X_train['hour'], bins=time_intervals, labels=[0, 1, 2, 3,4])

    # Change the categorical variables to 0,1,2... (some of them start with one)
    X_train["season"] = X_train["season"].replace({1: 0, 2: 1, 3: 2, 4: 3})
    X_train["weather"] = X_train["weather"].replace({1: 0, 2: 1, 3: 2, 4: 3})
    

    #Non-probablistic approach
    X_train_reg= select_variables(X_train)

    #Save data 
    X_train_reg.to_csv(os.path.join(output_filepath, 'X_train.csv'), index=False)
    Y_train.to_csv(os.path.join(output_filepath, 'Y_train.csv'), index=False)
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    input_filepath= os.path.join(os.getcwd(),"data","raw")
    output_filepath= os.path.join(os.getcwd(),"data","processed")

    main(input_filepath, output_filepath)
