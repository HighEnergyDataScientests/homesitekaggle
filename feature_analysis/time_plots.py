# -----------------------------------------------------------------------------
#  Name: time_plots
#  Purpose: Plot fields vs time
#
#
# -----------------------------------------------------------------------------
"""
Make a plot of non-categorical fields vs time.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg")  # Needed to save figures
import matplotlib.pyplot as plt

# Import data
print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')

# Process data
print("## Data Processing")
train = train.drop('QuoteNumber', axis=1)

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))

# Split data into 2 dataframes, based on whether the insurance was bought
df_pos = train.loc[train['QuoteConversion_Flag'] == 1] # bought
df_neg = train.loc[train['QuoteConversion_Flag'] == 0] # not bought

# Create list of features
features = [s for s in train.columns.ravel().tolist() if s != 'QuoteConversion_Flag']

# Create plots
for f in features:
    if train[f].dtype.name != 'object' and f != 'Date':
        plt.clf() # Clear figure

        plt.xlabel('Date')
        plt.ylabel(f)
        plt.title(f + ' vs. Date')

        df_pos.plot(x='Date', y=f, color='b')
        df_neg.plot(x='Date', y=f, color='r')

        plt.legend(labels=['Converted', 'Not Converted'])

        plt.savefig('plots_time/' + f + '.png')
