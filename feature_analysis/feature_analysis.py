#!/usr/bin/python
###############################################################################
# This code is developed by HighEnergyDataScientests Team.
# Do not copy or modify without written approval from one of the team members.
###############################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg")  # Needed to save figures
import matplotlib.pyplot as plt

# seed = 260681


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')


print("## Data Processing")
train = train.drop('QuoteNumber', axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

# train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
# train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
# train['weekday'] = train['Date'].dt.dayofweek

# test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
# test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
# test['weekday'] = test['Date'].dt.dayofweek

# train = train.drop('Date', axis=1)
# test = test.drop('Date', axis=1)

# Missing data
# train = train.fillna(-1)
# test = test.fillna(-1)

print("## Data Encoding")
for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# Create list of features
features = [s for s in train.columns.ravel().tolist() if s != 'QuoteConversion_Flag']
print("Features: ", features)

# Split data into 2 dataframes, based on whether the quote was bought or not
df_pos = train.loc[train['QuoteConversion_Flag'] == 1]
df_neg = train.loc[train['QuoteConversion_Flag'] == 0]

# Remove QuoteConversion_Flag column
df_pos.drop('QuoteConversion_Flag', axis=1)
df_neg.drop('QuoteConversion_Flag', axis=1)

# Plot each column against Date
for f in features:
    if f != 'Date':
        fig = plt.figure()

        # Labels
        plt.title(f + 'vs. Date')
        plt.xlabel('Date')
        plt.ylabel(f)

        # Plot data
        plt.scatter(df_neg['Date'], df_neg[f], 'rs')
        plt.scatter(df_pos['Date'], df_pos[f], 'b^')

        # Save plot
        fig.savefig('PlotsTime/' + f + '.png')



# featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# fig_featp = featp.get_figure()
# fig_featp.savefig('feature_importance_xgb.png',bbox_inches='tight',pad_inches=1)
# df.to_csv("feature_importance.csv")
#
