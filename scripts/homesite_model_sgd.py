#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

#seed = 260681

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
print("Features: ", features)


print("Train a SGDClassifier model")
X_train, X_valid = train_test_split(train, test_size=0.01)
y_train = X_train['QuoteConversion_Flag']
y_valid = X_valid['QuoteConversion_Flag']

clf = SGDClassifier(loss="hinge", penalty="l2", n_jobs=-1)
clf.fit(X_train[features].values, y_train.values)


print("## Validating Data")
preds = clf.decision_function(X_valid[features])
auc_value = roc_auc_score(y_valid, preds)
print("ROC Score : " + str(auc_value))

print("## Predicting test data")
preds = clf.decision_function(test[features].values)
test["QuoteConversion_Flag"] = preds
test[['QuoteNumber',"QuoteConversion_Flag"]].to_csv('test_predictions.csv', index=False)
