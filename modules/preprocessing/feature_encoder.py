#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm

class DataEncoder:
    '''
    Class used to encode categorigal data
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
        
        return pd.get_dummies(data=self.dataframe, 
                           prefix=None, 
                           prefix_sep='_', 
                           dummy_na=False, 
                           columns=None, 
                           sparse=False, 
                           drop_first=False, 
                           dtype=None)
    
    def preprocess_feature_embedding(self,columns_name,more_data=None,separator=","):
        
        '''
        Make dummy variable on a feature from training and test dataframe containing word separated by comma
        
        Parameters
        ----------
        more_data : Pandas DataFrame
            In any case there is value in test data not present in training data on feature
        column_name : string
            column name (str) to make the dummy variable
                
        Returns
        -------
        DataFrame 
            DataFrame with expanded categorical variable (dummy variable) from feature
        
        Examples
        --------
        >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
        >>> df_drugs_test = pd.read_csv(path_drugs_test, sep=",")  
        >>> preprocess_feature_embedding(data = df_drugs_train,more_data = df_drugs_test,feature="route_of_administration")
                      drug_id  ...  route_of_administration_orale
            0          0_test  ...                             1
            1         0_train  ...                             1
            2       1000_test  ...                             1
            3      1000_train  ...                             1
            4       1001_test  ...                             1  
        '''
        
        if isinstance(columns_name,str): columns_name = [columns_name]
        for column_name in set(columns_name):
            if more_data is not None:
                list_column_name = [word for value in set(pd.concat([self.dataframedata[column_name], more_data[column_name]], ignore_index=True)) for word in set(value.split(separator))]
            else:
                list_column_name = [word for value in set(self.dataframedata[column_name]) for word in set(value.split(separator))]
            
            for value in tqdm(set(list_column_name)):
                self.dataframedata[column_name + "_" + value] = 0
                for index in range(self.dataframe.shape[0]):
                    if value in set(self.dataframe[column_name][index]):
                        self.dataframedata[column_name + "_" + value][index] = 1
                        
        return self.dataframe