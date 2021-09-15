#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

class DataPreprocessor:
    '''
    Class used to make data preprocessor
    '''
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def _find_unique_user(self, user_column, element_column, option_column=None):
        '''
        Method used to find unique user

        Parameters
        ----------
        user_column : string
            Name of the column containing user name, id or anything else...
        element_column : string
            Name of the column containg the element of the user
        option_column : string
            Name of an additional column to consider to find user

        Returns
        -------
        tuple
            tuple containg list of ([user], [number of element], [option element])

        '''
        
        unique_users = self.dataframe[user_column].unique()
        list_user = list(unique_users)
        list_number_element = []
        list_option_element = []
        for user in list_user:
            list_number_element.append(self.dataframe[self.dataframe[user_column]==user].shape[0])
            if option_column is not None:
                #Sometimes gender contain male and/or female and/or NaN
                list_option_element += [list(self.dataframe[self.dataframe[user_column]==user][option_column].unique())[0]]
        return (list_user, list_number_element, list_option_element)
    
    def _find_max_user(self, list_unique_user, option=None):
        '''
        Find the user who have the biggest amount of elements

        Parameters
        ----------
        list_unique_user : list
            list containg list of [user, number of element, list_option_element=None]
        option : string
            option use to specify parameter on optional list from list_unique_user to find max user

        Returns
        -------
        string
            String containg user who has te biggest amount of elements

        '''
        
        list_user = list_unique_user[0]
        list_number_element = list_unique_user[1]
        list_option_element = list_unique_user[2]
        
        print(len(list_user))
        print(len(list_number_element))
        print(len(list_option_element))
        
        if option is None:
            biggest_user = list_user[list_number_element.index(max(list_number_element))]
        else:
            index_option = [i for i,x in enumerate(list_option_element) if str(x).lower() == option]
            list_user_option = [list_user[index] for index in index_option]
            list_number_element_option = [list_number_element[index] for index in index_option]
            biggest_user = list_user_option[list_number_element_option.index(max(list_number_element_option))]

        return biggest_user
    
    def _get_information_from_user(self, 
                                   user, 
                                   user_column, 
                                   path_column, 
                                   element_column,
                                   data_directory):
        '''
        Get information from an user

        Parameters
        ----------
        user : string
            User id or user name...
        user_column : string
            Name of the column containing user name, id or anything else...
        path_column : string
            Name of the column containing file path
        element_column : string
            Name of the column containing element like sentences...
        data_directory : string
            directory containg audio data

        Returns
        -------
        Pandas dataframe
            dataframe containg information about user in LJ Speech Dataset format

        '''
        
        table = self.dataframe[self.dataframe[user_column]==user][[path_column,element_column]]
        table[path_column] = table[path_column].apply(lambda x : os.path.join(data_directory,x))
        table = pd.DataFrame({path_column : list(table[path_column]), element_column : list(table[element_column])},
                             columns=[path_column,element_column])
        return table[path_column].astype(str) + "|" + table[element_column].astype(str)
    
    def convert_data_mcv_to_lsj(self,
                                user_column, 
                                path_column, 
                                element_column,
                                data_directory,
                                option_column=None,
                                option=None):
        '''
        From a dataframe containg audio information in Mozilla Common voice format, convert it
        into LJ Speech Dataset audio information format

        Parameters
        ----------
        user_column : string
            Name of the column containing user name, id or anything else...
        path_column : string
            Name of the column containing file path
        element_column : string
            Name of the column containg the element of the user
        data_directory : string
            directory containg audio data
        option_column : string
            Name of an additional column to consider to find user
        option : string
            option use to specify parameter on optional list from list_unique_user to find max user
            
        Returns
        -------
        Pandas dataframe
            dataframe containg information about user in LJ Speech Dataset format

        '''
        
        list_unique_user = self._find_unique_user(user_column, element_column, option_column)
        user = self._find_max_user(list_unique_user, option)
        
        return self._get_information_from_user(user, 
                                               user_column, 
                                               path_column, 
                                               element_column,
                                               data_directory)
    