# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%


import pandas as pd
import pandas_datareader as pdr
from yahoo_finance import Share
import datetime as dt
from datetime import datetime, timedelta
from yahoo_finance import Share
import numpy as np

#%%

df =  pdr.get_data_yahoo('^AXJO')
#%% Day of week
df['date'] = df.index
df['dayofweek'] =df['date'].dt.dayofweek
df['month'] =df['date'].dt.month
df['year'] =df['date'].dt.year
df
#%% Change and Direction

df['change_real'] = df['Open'] - df['Close']
df['change_pct'] = df['change_real'] / df['Open']
df['change_pct_abs'] = abs(df['change_pct'])

df['change_real_t2'] = df['Open'].shift(periods=1) - df['Close']
df['change_pct_t2'] = df['change_real_t2'] / df['Open'].shift(periods=1)

df['change_real_t3'] = df['Open'].shift(periods=2) - df['Close']
df['change_pct_t3'] = df['change_real_t3'] / df['Open'].shift(periods=2)

df['change_real_t4'] = df['Open'].shift(periods=3) - df['Close']
df['change_pct_t4'] = df['change_real_t4'] / df['Open'].shift(periods=3)


df['change_real_t5'] = df['Open'].shift(periods=4) - df['Close']
df['change_pct_t5'] = df['change_real_t5'] / df['Open'].shift(periods=4)


df['change_real_t6'] = df['Open'].shift(periods=5) - df['Close']
df['change_pct_t6'] = df['change_real_t6'] / df['Open'].shift(periods=5)

df['change_real_t7'] = df['Open'].shift(periods=6) - df['Close']
df['change_pct_t7'] = df['change_real_t7'] / df['Open'].shift(periods=6)


df['change_real_t8'] = df['Open'].shift(periods=7) - df['Close']
df['change_pct_t8'] = df['change_real_t8'] / df['Open'].shift(periods=7)

def sign(row):
    val=0
    if row['change_real'] > 0:
        val = 1
    elif row['change_real'] <= 0:
        val = 0
    return val

df['change_direction']=df.apply(sign, axis=1)
df
#%% If the market went up by 1,2 or 3%
def one_percenter_up(row):
    val=0
    if row['change_pct'] >= .01:
        val = 1
    else: 
        val = 0
    return val
df['one_percent_up']=df.apply(one_percenter_up, axis=1)
df


def two_percenter_up(row):
    val=0
    if row['change_pct'] >= .02:
        val = 1
    else: 
        val = 0
    return val
df['two_percent_up']=df.apply(two_percenter_up, axis=1)
df


def three_percenter_up(row):
    val=0
    if row['change_pct'] >= .03:
        val = 1
    else: 
        val = 0
    return val
df['three_percent_up']=df.apply(three_percenter_up, axis=1)
df
#%% Consecutive Days of movement in the same direction
def rolling_count(val):
    if val == rolling_count.previous:
        rolling_count.count +=1
    else:
        rolling_count.previous = val
        rolling_count.count = 1
    return rolling_count.count
rolling_count.count = 0 #static variable
rolling_count.previous = None #static variable


df['consecutive_change'] = df['change_direction'].apply(rolling_count) #new column in dataframe
#%% 
run_change_list = []
opens = [i for i in df['Open']]
closes = [i for i in df['Close']]
shifter = [i-1 for i in df['consecutive_change']]



counter = 1
for i,k in zip(shifter[1:],closes[1:]):
    dnom = counter-i
    run_change_list.append(( k-opens[dnom])/ opens[dnom])
    counter += 1
run_change_list.insert(0,0)

df['run_change'] =run_change_list
#%% A quick look at the data so far

df.plot.scatter(x='consecutive_change', y='run_change')

#%% 
df['y_one_pct'] = df['one_percent_up'].shift(periods=-1)

#%% A very small and weak learner
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = df.replace([np.inf, -np.inf], np.nan)
df1 = df.dropna()

x_cols = ['consecutive_change','run_change','change_direction',
          'change_real','change_pct',
          'change_real_t2','change_pct_t2',
          'change_real_t3','change_pct_t3',
          'change_real_t4','change_pct_t4',
          'change_real_t5','change_pct_t5']


y = [i for i in df1['y_one_pct']]

X_train, X_test, y_train, y_test = train_test_split( df1[x_cols], y, test_size=0.33, random_state=42)
X_train_m = X_train.as_matrix(columns=None)
#%%
np.isfinite(X_train.all())
X_train.isnull().any()
np.isnan(y_train)==True
np.isinf(y_train).sum()

X_train.dtypes
#%%

clf = RandomForestClassifier(n_jobs=2)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
accuracy_score(y_test, preds)
accuracy_score(y_test, preds, normalize=False)
print(classification_report(y_test, preds))


