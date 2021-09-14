#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DataPreprocessor:
    '''
    Class used to make data preprocessor
    '''
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def _find_unique_user(self, user_column, element_column):
        '''
        Method used to find unique user

        Parameters
        ----------
        user_column : string
            Name of the column containing user name, id or anything else...
        element_column : string
            Name of the column containg the element of the user

        Returns
        -------
        list
            list containg list of user and number of elemeent

        '''
        
        unique_users = self.dataframe[user_column].unique()
        res = []
        for user in set(unique_users):
            res.append([user,len(self.dataframe[self.dataframe[user_column]==user])])
        return res
    
    def _find_max_user(self,list_unique_user):
        '''
        

        Parameters
        ----------
        list_unique_user : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''