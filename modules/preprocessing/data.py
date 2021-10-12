#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm

from modules.preprocessing.audio import AudioPreprocessor
import librosa

class DataPreprocessor:
    '''
    Class used to make data preprocessor
    '''
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def _find_unique_user(self, user_column, element_column, option_column, path_column, data_directory, format_conversion):
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
        path_column : string
            Name of the column containing file path
        data_directory : string
            directory containg audio data
        format_conversion : string
            format used by tacotron2 training (always .wav)

        Returns
        -------
        tuple
            tuple containg list of ([user], [number of element], [option element])

        '''
        
        unique_users = self.dataframe[user_column].unique()
        list_user = list(unique_users)
        list_number_element = []
        list_option_element = []
        list_element_length = []
        print("List creation of number of element and option from user")
        for user in tqdm(list_user):
            list_number_element.append(self.dataframe[self.dataframe[user_column]==user].shape[0])
            #Find a way to make faster audio length computation usinf data info and soxi -D directly in console...
            #list_element_length.append(self.dataframe[self.dataframe[user_column]==user][path_column].apply(lambda path_audio : librosa.get_duration(filename=os.path.join(data_directory,path_audio))).sum())            
            #print(max(list_element_length))
            if option_column is not None:
                #Sometimes gender contain male and/or female and/or NaN
                list_option_element += [list(self.dataframe[self.dataframe[user_column]==user][option_column].unique())[0]]
                
        return (list_user, list_number_element, list_option_element, list_element_length)
    
    def _find_max_user(self, list_unique_user, option):
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
                                   data_directory_converted,
                                   format_conversion):
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
        data_directory_converted : string
            directory containing converted audio data
        format_conversion : string
            format used by tacotron2 training (always .wav)

        Returns
        -------
        Pandas dataframe
            dataframe containg information about user in LJ Speech Dataset format

        '''
        
        table = self.dataframe[self.dataframe[user_column]==user][[path_column,element_column]]
        return table.apply(lambda x : os.path.join(data_directory_converted,os.path.splitext(x[path_column])[0]+format_conversion) + "|" + x[element_column], axis=1).reset_index(drop=True)
    
    def convert_data_mcv_to_lsj(self,
                                user_column, 
                                path_column, 
                                element_column,
                                data_directory,
                                data_directory_converted,
                                option_column=None,
                                option=None,
                                format_conversion=".wav"):
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
            directory containing audio data
        data_directory_converted : string
            directory containing converted audio data
        option_column : string
            Name of an additional column to consider to find user
        option : string
            option use to specify parameter on optional list from list_unique_user to find max user
        format_conversion : string
            format used by tacotron2 training (always .wav)
            
        Returns
        -------
        Pandas dataframe
            dataframe containg information about user in LJ Speech Dataset format

        '''
        
        list_unique_user = self._find_unique_user(user_column, element_column, option_column, path_column, data_directory, format_conversion)
        #(list_user, list_number_element, list_option_element, list_element_length)
        data_info = pd.DataFrame({user_column:list_unique_user[0],element_column:list_unique_user[1],option_column:list_unique_user[2],path_column:list_unique_user[3]},
                                 columns=[user_column, element_column, option_column, path_column])
        
        user = self._find_max_user(list_unique_user, option)
        
        return (self._get_information_from_user(user, 
                                               user_column, 
                                               path_column, 
                                               element_column,
                                               data_directory_converted,
                                               format_conversion),data_info)
    