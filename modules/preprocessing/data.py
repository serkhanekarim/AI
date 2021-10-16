#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
import re

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
        # Waiting for a fast audio length computation
        # data_info = pd.DataFrame({user_column:list_unique_user[0],element_column:list_unique_user[1],option_column:list_unique_user[2],path_column:list_unique_user[3]},
        #                          columns=[user_column, element_column, option_column, path_column])
        
        data_info = pd.DataFrame({user_column:list_unique_user[0],element_column:list_unique_user[1],option_column:list_unique_user[2]},
                                  columns=[user_column, element_column, option_column])
        
        user = self._find_max_user(list_unique_user, option)
        
        return (self._get_information_from_user(user, 
                                               user_column, 
                                               path_column, 
                                               element_column,
                                               data_directory_converted,
                                               format_conversion),data_info)
    
    def _useless_data(self, data):
        '''
        Find data index to keep for training taflowtron by removing useless data

        Parameters
        ----------
        data : list
            list of data containing string

        Returns
        -------
        list
            list of index to remove from data

        '''
        
        index_to_remove = [index for index,element in enumerate(data) if len(element) > 0 if element[0]=='[']
        index_to_remove += [index for index,element in enumerate(data) if len(element) == 0]
        
        return index_to_remove
    
    def _concatenate_subtitle(self, list_time, list_subtitle):
        '''
        Concatenate subtitle to get long sentences and not cut sentences

        Parameters
        ----------
        list_time : list
            list of data containing time
        list_subtitle : list
            list of data containing string

        Returns
        -------
        None.

        '''
        new_list_time = []
        new_list_subtitle = []
        
        index = 0
        while index < len(list_subtitle)-1:
            compt = 0
            subtitle = list_subtitle[index]
            end_time = list_time[index][0][1]
            while list_time[index+compt][0][1] == list_time[index+compt+1][0][0] and list_subtitle[index+compt][-1] != ".":
                subtitle += list_subtitle[index+compt+1]
                end_time = list_time[index+compt+1][0][1]
                compt += 1
            new_list_time.append((list_time[index][0][0],list_time[index+compt][0][1]))
            new_list_subtitle.append(subtitle)
            index += compt + 1
        
    
    def get_info_from_vtt(self, data):
        '''
        Get time of subtitile and subtitle

        Parameters
        ----------
        data : list
            list of vtt data

        Returns
        -------
        list
            list of list containing start and end time of the subtitle and the subtitle [(xxxx,xxxx),xxxx]

        '''
        
        list_time = []
        list_subtitle = []
        
        index = 0
        while index < len(data):
            element = data[index]
            compt = 1
            subtitle = ''
            if re.search(r'\d\d\:\d\d\:\d\d\.\d\d\d --> \d\d\:\d\d\:\d\d\.\d\d\d', element):
                list_time.append(re.findall(r'(\d\d\:\d\d\:\d\d\.\d\d\d) --> (\d\d\:\d\d\:\d\d\.\d\d\d)', element))
                while data[index + compt] != '\n':
                    subtitle += data[index + compt].replace('\n',' ')
                    compt += 1
                subtitle = re.sub(r'\[.*\]|^\s+|\s+$','',subtitle)
                subtitle = re.sub(r'\s{2,]',' ',subtitle)
                list_subtitle.append(subtitle)
            index += 1
            
        index_to_remove = self._useless_data(list_subtitle)
        list_time = [element for index,element in enumerate(list_time) if index not in index_to_remove]
        list_subtitle = [element for index,element in enumerate(list_subtitle) if index not in index_to_remove]
        
        '''
        Improvement merge separetaed time from vtt
        '''
        
        
        
        return [list_time, list_subtitle]
        
        
        