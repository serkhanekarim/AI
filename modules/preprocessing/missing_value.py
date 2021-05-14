#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from modules.Global import variable

class DataImputation:
    '''
    Class used to impute missing data
    '''
    
    MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = variable.Var().MAX_NUMBER_OF_CATEGORICAL_OCCURENCES
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def _feature_type_detector(self, column_name):
        '''
        Find the type of feature, categorical, continuous or class number

        Parameters
        ----------
        column_name : string
            Name of the columnn to determine the type of feature.

        Returns
        -------
        string
            Type of feature

        '''
        length_value = len(self.dataframe[column_name])
        length_unique_value = len(set(self.dataframe[column_name])) 
        
        boolean_feature_class = self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES >= length_value and \
             length_unique_value <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES and \
                  self.dataframe[column_name].dtype == "int64"
        
        boolean_feature_continuous = self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES < length_value and \
             length_unique_value > self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES and \
                  self.dataframe[column_name].dtype == "float64"
                  
        boolean_feature_categorical = self.dataframe[column_name].dtype == "object"
                
        if boolean_feature_class: return 'class'
        if boolean_feature_continuous: return 'continous'
        if boolean_feature_categorical: return 'categorical'
        return -1
            
        
    def imputer(self, column_name, method='knn'):
        '''
        Impute missing data to a missing data

        Parameters
        ----------
        column_name : string
            Name of the column to impute data

        Returns
        -------
        DataFrame
            Return updated dataframe of the missing data from the column.

        '''
        
        feature_type = self._feature_type_detector(column_name)
        if feature_type == "class": strategy = "median"
        if feature_type == "continuous": strategy = "mean"
        if feature_type == "categorical": strategy = "most_frequent"
        
        if method == 'simple': imp = SimpleImputer(strategy=strategy)
        if method == 'iterative': imp = IterativeImputer(max_iter=10, initial_strategy=strategy)
        if method == 'knn': imp = KNNImputer(n_neighbors=2, weights="uniform")
        
        imp.fit(self.dataframe[column_name])
        self.dataframe[column_name] = imp.transform(self.dataframe[column_name])
        
        return self.dataframe
    