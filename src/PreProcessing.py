import numpy as np
import re
import pandas as pd
from datetime import datetime,timedelta

def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
    return df

def get_dates(df):    
    df = df.reset_index()
    last = df['Date'][len(df)-1]
    last.strftime("%Y-%m-%d")
    dates=[]
    for i in range(0,100):
        y = datetime.now() + timedelta(days=i)
        date = y.strftime("%Y-%m-%d 00:00:00")
        date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        dates.append(date_obj)
    return dates

def expand_train_dataset(df):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    
    add_datepart(new_data, 'Date')
    new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp
    new_data['mon_fri'] = 0
    for i in range(0,len(new_data)):
        if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
            new_data['mon_fri'][i] = 1
        else:
            new_data['mon_fri'][i] = 0
    
    return new_data

def expand_test_dataset(dates):
    input_data = pd.DataFrame(dates,columns=['Date'])
    df = input_data
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
    
    add_datepart(new_data, 'Date')
    new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp
    new_data['mon_fri'] = 0
    for i in range(0,len(new_data)):
        if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
            new_data['mon_fri'][i] = 1
        else:
            new_data['mon_fri'][i] = 0
    return new_data