import pandas as pd
from utils import extract_datetime


def preprocessing_Xdata(X_train):
    X_train = extract_datetime(X_train)

    time_intervals = [-1, 6, 10,15, 20, 24]
    # Create a column of time ranges
    X_train['time_range'] = pd.cut(X_train['hour'], bins=time_intervals, labels=[0, 1, 2, 3,4])


    # Change the categorical variables to 0,1,2... (some of them start with one)
    X_train["season"] = X_train["season"].replace({1: 0, 2: 1, 3: 2, 4: 3})
    X_train["weather"] = X_train["weather"].replace({1: 0, 2: 1, 3: 2, 4: 3})

    return X_train