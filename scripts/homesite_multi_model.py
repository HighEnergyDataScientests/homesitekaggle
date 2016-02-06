#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import time


seeds = [234,99034,536536,1145256]


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')


print("## Data Processing")
y = train.QuoteConversion_Flag.values
train = train.drop('QuoteNumber', axis=1)
#test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

print("## Data Encoding")
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

features = [s for s in train.columns.ravel().tolist() if s != 'QuoteConversion_Flag']
#print("Features: ", features)
#for f in sorted(set(features)):
#    print f
#exit()

preds_df = pd.DataFrame()
caseNum = 1

for seedValue in seeds:
	timestr = time.strftime("%Y%m%d-%H%M%S")
	print("## Training First Model ", caseNum)
	params = {"objective": "binary:logistic",
        	  "eta": 0.3,
        	  "nthread":3,
        	  "max_depth": 10,
        	  "subsample": 0.8,
        	  "colsample_bytree": 0.8,
        	  "eval_metric": "auc",
        	  "silent": 1,
        	  "seed": seedValue
        	  }
	num_boost_round = 1000

	print("## Train a XGBoost model ", caseNum)
	X_train, X_valid = train_test_split(train, test_size=0.05)
	y_train = X_train['QuoteConversion_Flag']
	y_valid = X_valid['QuoteConversion_Flag']
	dtrain = xgb.DMatrix(X_train[features], y_train)
	dvalid = xgb.DMatrix(X_valid[features], y_valid)

	watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, maximize=True, early_stopping_rounds=100, verbose_eval=True)


	print("## Predicting test data for model", caseNum)
	preds = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=gbm.best_ntree_limit)
	preds_df["model_"+timestr] = preds
	print preds_df
	
	print("## Saving Model ", caseNum)
	gbm.save_model("../models/" + timestr + '.model')
	test["QuoteConversion_Flag"] = preds
	test[['QuoteNumber',"QuoteConversion_Flag"]].to_csv("../predictions/pred_" + timestr + ".csv", index=False)
	caseNum += 1


print(preds_df)

preds_df.to_csv("models_preds.csv", index=False)
