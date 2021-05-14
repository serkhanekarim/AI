#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

class DataSelector:
    '''
    Class used to make feature selection
    '''
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def dummy_variable(self):
        '''
        Convert categorical variable into dummy/indicator variables.

        Returns
        -------
        DataFrame
            Converted categorical variable into dummy/indicator variables.

        '''
        
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(df_train.drop('embauche', axis=1).fillna(0), df_train['embauche'].fillna(0))
        clf.feature_importances_