
import pandas as pd
import numpy as np
import os, datetime

from visualizations import samples_from_posterior,plot_numerical_variable,plot_categorical_variables,compare_yhat_ytrue

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



def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include='category').columns
    
    for col in categorical_cols:
        encoded_cols = pd.get_dummies(df[col], prefix=col, drop_first=False) 
        df = pd.concat([df, encoded_cols], axis=1)
        df = df.drop(columns=[col])
    
    return df



def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, r2


def save_results_figures(alpha_samples,beta_samples,y_hat,errors,Y_train,X_train):
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f_path=os.path.join(os.getcwd,"trained_models",folder_name)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    
    # Save the NumPy arrays in the folder
    np.save(os.path.join(f_path, 'alpha_samples.npy'), alpha_samples)
    np.save(os.path.join(f_path, 'beta_samples.npy'), beta_samples)
    np.save(os.path.join(f_path, 'y_hat.npy'), y_hat)

    with open(f_path+"errors.txt",'w') as file:
        for key, value in errors.items():           
            file.write(f"{key}: {value}\n")

    
    samples_from_posterior(alpha_samples,f_path)
    plot_numerical_variable(beta_samples,f_path)
    plot_categorical_variables(beta_samples, X_train,f_path)
    compare_yhat_ytrue(y_hat,Y_train,f_path)
    compare_yhat_ytrue(y_hat[Y_train< 500], Y_train[Y_train< 500],f_path)
    


