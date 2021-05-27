#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import preprocessing

class DataScaler:
    '''
    Class used to scale data
    '''
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def _check_sign_feature(self):
        '''
        Check if there is a negative value or only positive value in the dataframe

        Parameters
        ----------
        None

        Returns
        -------
        str
            positive if there are only positive value or an existing negative value.

        '''
        
        for column_name in self.dataframe.columns:
            if min(self.dataframe[column_name]) < 0: 
                return 'negative'
        return 'positive'
        
    def scaler(self, method='yeo-johnson'):
        '''
        Scale data to gaussian distribution N(0,1)

        Parameters
        ----------
        column_name : string
            Name of the column to scale data.
        method : string, optional
            Method to use for scaling transformation. The default is 'yeo-johnson'.

        Returns
        -------
        dataframe : DataFrame
            Return updated dataframe of the missing data from the column.
        scaler: object
            scaler created with the data.
            
        '''
        
        if method == 'standard': scaler = preprocessing.StandardScaler()
        if method == 'minmax': scaler = preprocessing.MinMaxScaler()
        if method == 'maxabs': scaler = preprocessing.MaxAbsScaler()
        if method == 'robust': scaler = preprocessing.RobustScaler()
        if method == 'quantile': scaler = preprocessing.QuantileTransformer(output_distribution='normal')
        
        if method == 'l1': scaler = preprocessing.normalize(method)
        if method == 'l2': scaler = preprocessing.normalize(method)
        if method == 'max': scaler = preprocessing.normalize(method)
        
        feature_sign = self._check_sign_feature()
        if method == 'box-cox' or feature_sign == 'positive': scaler = preprocessing.PowerTransformer(method)
        if method == 'yeo-johnson' or feature_sign == 'negative': scaler = preprocessing.PowerTransformer(method)
        
        scaler.fit(self.dataframe)
        self.dataframe = scaler.transform(self.dataframe)
        
        return self.dataframe, scaler
        