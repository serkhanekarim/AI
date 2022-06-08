#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:04:42 2022

@author: serkhane
"""

import xgboost as xgb
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

from modules.reader.reader import DataReader
import csv

from sklearn.preprocessing import OneHotEncoder

# create sample dataset
X_mul, y_mul = make_multilabel_classification(n_samples=100, n_features=3, n_classes=6, n_labels=1,
                                      allow_unlabeled=False, random_state=42)

obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
data_lottery = DataReader("/home/serkhane/Repositories/AI/DATA/lottery_fr-FR.tsv").read_data_file(**obj)

X = data_lottery[[0,1,2]]
y = data_lottery[[3,4,5,6,7,8]]

enc_X = OneHotEncoder()
enc_X.fit(X)
enc_X.categories_
X_array = enc_X.transform(X).toarray()

enc_y = OneHotEncoder()
enc_y.fit(y)
enc_y.categories_
y_array = enc_y.transform(y).toarray()

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

evallist = [(dtrain, 'eval'), (dtest, 'train')]
evals_result = {}    
param = {'num_round' : 5000,
         'max_depth': 6, 
         'eta': 0.01, 
         'objective': 'binary:logistic', 
         'nthread' : -1,
         'num_class' : 2}

bst = xgb.train(param, 
                X_array,
                evallist, 
                early_stopping_rounds=10, 
                evals_result=evals_result)

# create XGBoost instance with default hyper-parameters
xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', verbosity=3, n_jobs=-1, n_estimators=10000, learning_rate=0.01)

# create MultiOutputClassifier instance with XGBoost model inside
multilabel_model = MultiOutputClassifier(xgb_estimator)

# fit the model
multilabel_model.fit(X_train, y_train, verbose=True)

y_pred = multilabel_model.predict(X_test)
y_pred_proba = multilabel_model.predict_proba(X_test)

enc_y.inverse_transform(y)

# evaluate on test data
print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test, multilabel_model.predict(X_test))*100))















import xgboost as xgb
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# create sample dataset
X_mul, y_mul = make_multilabel_classification(n_samples=3000, n_features=45, n_classes=20, n_labels=1,
                                      allow_unlabeled=False, random_state=42)

# split dataset into training and test set
X_train_mul, X_test_mul, y_train_mul, y_test_mul = train_test_split(X_mul, y_mul, test_size=0.2, random_state=123)

# create XGBoost instance with default hyper-parameters
xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', num_class=49, verbosity=3, n_jobs=-1)

# create MultiOutputClassifier instance with XGBoost model inside
multilabel_model = MultiOutputClassifier(xgb_estimator)

# fit the model
multilabel_model.fit(X_train, y_train)

# evaluate on test data
print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test, multilabel_model.predict(X_test))*100))



