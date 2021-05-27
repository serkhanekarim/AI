#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from modules.Global.method import DataMethod


class DataGenerator:
    '''
    Class used to generate features
    '''
    
    def __init__(self, dataframe, selected_dataframe):
        self.dataframe = dataframe
        self.selected_dataframe = selected_dataframe
        
    def polynomial_features(self, degree):
        '''
        Generate polynomial feature from dataframe

        Parameters
        ----------
        degree : int
            Number of polynomial degree to use for polynomial generation

        Returns
        -------
        dataframe : DataFrame
            Return updated dataframe with generated polynomial features.
            
        '''
        
        poly = PolynomialFeatures(degree=degree)
        poly.fit(self.selected_dataframe)
        transformed_data = poly.transform(self.selected_dataframe)
        
        new_length_added = (len(transformed_data[0]) - 1) - self.selected_dataframe.shape[1]
        new_column_name = DataMethod.get_new_column_name(length_new_matrix = new_length_added, 
                                                              prefix = "polynomial_features")
        
        self.selected_dataframe = pd.DataFrame(transformed_data, columns = ['Unit_Feature_Generated'] + list(self.selected_dataframe.columns) + new_column_name)
        
        self.dataframe = pd.concat([self.dataframe, self.selected_dataframe[self.selected_dataframe.columns.difference(self.dataframe.columns)]], axis=1)
        
        return self.dataframe, poly