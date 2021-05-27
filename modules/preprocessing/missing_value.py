#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

#from sklearn.impute import SimpleImputer
#from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from modules.Global.method import DataMethod

class DataImputation:
    '''
    Class used to impute missing data
    '''
        
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def imputer(self, method='knn'):
        '''
        Impute missing data to a missing data

        Parameters
        ----------
        column_name : string
            Name of the column to impute data

        Returns
        -------
        dataframe : DataFrame
            Return updated dataframe of the missing data from the column.
        imp : object
            imputer created with the data.

        '''
        print("Impute missing data using: " + method)
        # feature_type = self._feature_type_detector(column_name)
        # if feature_type == "class": strategy = "median"
        # if feature_type == "continuous": strategy = "mean"
        # if feature_type == "categorical": strategy = "most_frequent"
        
        # if method == 'simple': imp = SimpleImputer(strategy=strategy)
        # if method == 'iterative': imp = IterativeImputer(max_iter=10, initial_strategy=strategy)
        if method == 'knn': imp = KNNImputer(n_neighbors=5, weights="uniform", add_indicator=True)
        
        imp.fit(self.dataframe)
        transformed_data = imp.transform(self.dataframe)
        
        new_length_added = len(transformed_data[0]) - self.dataframe.shape[1]
        new_column_name = DataMethod.get_new_column_name(length_new_matrix = new_length_added, 
                                                              prefix = "kNN_NaN_indicator")
        
        self.dataframe = pd.DataFrame(transformed_data, columns = list(self.dataframe.columns) + new_column_name)
        
        return self.dataframe, imp
    