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

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))

train[['PersonalField7','PropertyField3','PropertyField4']].fillna('N',inplace=True)
train[['PersonalField7','PropertyField3','PropertyField32']].fillna('Y',inplace=True)

# Create list of features
features = [s for s in train.columns.ravel().tolist() if s != 'QuoteConversion_Flag']
print("Features: ", features)

# Split data into 2 dataframes, based on whether the quote was bought or not
#df_pos = train.loc[train['QuoteConversion_Flag'] == 1]
#df_neg = train.loc[train['QuoteConversion_Flag'] == 0]

# Plot each column against Date
for f in features:
    if f != 'Date':
    	if len(pd.unique(train[f])) == 2:
        	print("Unique value for ", f, " : " , pd.unique(train[f]))
		
		plt.clf() # Clear figure
        	
		colors = np.random.rand(2)
		
		lbl = preprocessing.LabelEncoder()
        	lbl.fit(list(train[f].values))
        	train[f] = lbl.transform(list(train[f].values))
		
		corr = train[f].corr(train['QuoteConversion_Flag'])
		
		print("Correlation betweeen ", f, " and the output is ", corr)
		
		#train[[f,'QuoteConversion_Flag']].plot(style=['o','rx'])
		#x = train[f].values
		#y = train['QuoteConversion_Flag'].values
		
        	#plt.xlabel(f)
		#plt.ylabel('QuoteConversion_Flag')
        	
		#plt.scatter(x, y, c=colors, alpha=0.5)
		
		#lbl = preprocessing.LabelEncoder()
        	#lbl.fit(list(x))
        	#x = lbl.transform(list(x))
		
		#print("X = ", x)
		#print("Y = ", y)
		#plt.plot(x, y,'ro')
		#plt.savefig('plot_scatter/' + f + '.png')


