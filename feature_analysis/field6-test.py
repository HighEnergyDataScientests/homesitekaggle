# -----------------------------------------------------------------------------
# Name: field6-test
# Purpose: testing out plotting
#
# -----------------------------------------------------------------------------
"""
Make a bar graph for Field6.
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

# Create plot
plt.figure()
width = 0.4
plt.xlabel('Field6')
plt.ylabel('Freq')
plt.title('Field6 Bar Graph')

df_pos['Field6'].value_counts().plot(kind='bar', color='blue', width=width, position=1)
df_neg['Field6'].value_counts().plot(kind='bar', color='red', width=width, position=0)

plt.legend(labels=['Converted', 'Not Converted'])

plt.savefig('Field6.png')