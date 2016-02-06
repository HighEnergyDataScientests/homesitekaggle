# -----------------------------------------------------------------------------
#  Name: correlation
#  Purpose: Calculate correlations
#
#
# -----------------------------------------------------------------------------
"""
Calculate correlations
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
train.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

# Calculate matrices
print("## Calculating matrices")
correlation = train.corr()
covariance = train.cov()

# Save to csv
correlation.to_csv('stats/correlation_matrix.csv')
covariance.to_csv('stats/covariance_matrix.csv')

# Plot matrices
print("## Plotting")
plt.matshow(correlation)
plt.savefig('stats/correlation_matrix.png')
plt.clf()

plt.matshow(covariance)
plt.savefig('stats/covariance_matrix.png')
plt.clf()