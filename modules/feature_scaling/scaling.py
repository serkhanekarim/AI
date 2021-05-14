#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import preprocessing

class DataScaler:
    '''
    Class used to scale data
    '''
    
    def __init__(self, dataframe:
        self.dataframe = dataframe
        
    def scaler(self, column_name, method='standard'):
    
        
        if method == 'standard': scaler = preprocessing.StandardScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.MinMaxScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.MaxAbsScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.RobustScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.RobustScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.RobustScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.RobustScaler().fit(self.dataframe)
        if method == 'standard': scaler = preprocessing.RobustScaler().fit(self.dataframe)
        
        self.dataframe[column_name] = scaler.transform(self.dataframe[column_name])
        
        return self.dataframe
        