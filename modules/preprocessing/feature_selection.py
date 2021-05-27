#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

class DataSelector:
    '''
    Class used to make feature selection
    '''
    
    def __init__(self, dataframe, label):
        self.dataframe = dataframe
        self.label = label
        
    def features_selection(self, max_num_features=5):
        '''
        Get the importances of features

        Returns
        -------
        DataFrame
            Converted categorical variable into dummy/indicator variables.

        '''
        max_num_features = min(max_num_features,self.dataframe.shape[1])
        clf = ExtraTreesClassifier(n_estimators=500,n_jobs=-1,verbose=1)
        clf = clf.fit(self.dataframe,self.label)
        feat_importances = pd.Series(clf.feature_importances_, index=self.dataframe.columns)
        feat_importances = feat_importances.nlargest(max_num_features)
        
        return clf, self.dataframe[feat_importances.index]