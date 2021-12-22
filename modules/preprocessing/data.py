#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
import re

from modules.Global.variable import Var
from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.cleaner import DataCleaner
from modules.preprocessing.time import TimePreprocessor
import librosa

class DataPreprocessor:
    '''
    Class used to make data preprocessor
    '''
        
    END_CHARS = Var().END_CHARS
    NB_LIMIT_FILE_CLUSTER = Var().NB_LIMIT_FILE_CLUSTER
    
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
        lists
            [user], [number of element], [option element]

        '''
        
        unique_users = self.dataframe[user_column].unique()
        list_user = list(unique_users)
        list_number_element = []
        list_option_element = []
        list_element_length = []
        print("List creation of number of element and option from user")
        for user in tqdm(list_user):
            list_number_element.append(self.dataframe[self.dataframe[user_column]==user].shape[0])
            if option_column is not None:
                #Sometimes gender contain male and/or female and/or NaN
                list_option_element += [list(self.dataframe[self.dataframe[user_column]==user][option_column].unique())[0]]
                
        return list_user, list_number_element, list_option_element, list_element_length
    
    def _find_max_user(self, list_user, list_number_element, list_option_element, option):
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
        
        if option is None:
            biggest_user = list_user[list_number_element.index(max(list_number_element))]
        else:
            index_option = [i for i,x in enumerate(list_option_element) if str(x).lower() == option]
            list_user_option = [list_user[index] for index in index_option]
            list_number_element_option = [list_number_element[index] for index in index_option]
            biggest_user = list_user_option[list_number_element_option.index(max(list_number_element_option))]

        return biggest_user
    
    
    def convert_data_mcv_to_taflowtron(self,
                                       user_column, 
                                       path_column, 
                                       element_column,
                                       data_directory,
                                       data_directory_preprocessed,
                                       path_cleaner,
                                       tts,
                                       option_column=None,
                                       option=None,
                                       format_conversion=".wav"):
        '''
        From a dataframe containg audio information in Mozilla Common voice format, convert it
        into taflowtron fileslist format

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
        data_directory_preprocessed : string
            directory containing futur preprocessed audio data
        path_cleaner : string
            path of a cleaner (.tsv file)    
        tts : string
            name of the tts to use; flowtron or tacotron2
        option_column : string
            Name of an additional column to consider to find user
        option : string
            option use to specify parameter on optional list from list_unique_user to find max user
        format_conversion : string
            format used by tacotron2 training (always .wav)
            
        Returns
        -------
        Pandas dataframe
            taflowtron filelist (dataframe or list), dataframe containg information about user, number of speaker

        '''
        
        list_user, list_number_element, list_option_element, _ = self._find_unique_user(user_column, element_column, option_column, path_column, data_directory, format_conversion)
        data_info = pd.DataFrame({user_column:list_user,element_column:list_number_element,option_column:list_option_element},columns=[user_column, element_column, option_column])
        
        if tts == "tacotron2":
            user = self._find_max_user(list_user, list_number_element, list_option_element, option)
            table = self.dataframe[self.dataframe[user_column]==user][[path_column,element_column]]
            table[element_column] = DataCleaner().clean_text(data=table[element_column],
                                                             path_cleaner=path_cleaner)
            table_filelist = table.apply(lambda x : os.path.join(data_directory_preprocessed,os.path.splitext(x[path_column])[0]+format_conversion) + "|" + x[element_column], axis=1).reset_index(drop=True)
            return table_filelist, data_info, 0

        if tts == "flowtron":
            total_filelist = []
            dir_to_create = []
            list_original_path = []
            for index, user in enumerate(list_user):
                table = self.dataframe[self.dataframe[user_column]==user][[path_column,element_column]]
                len_table = table.shape[0]
                table[element_column] = DataCleaner().clean_text(data=table[element_column],
                                                                 path_cleaner=path_cleaner)
                nb_part = len_table // (self.NB_LIMIT_FILE_CLUSTER + 1)
                part_extension = ["part_" + str(index) for index in range(nb_part+1)]
                dir_to_create += [os.path.join(data_directory_preprocessed,user,part) for part in part_extension]
                list_original_path += [os.path.join(data_directory,audio_path) for audio_path in table[path_column]]
                list_path = [os.path.join(data_directory_preprocessed,user,part_extension[index//(self.NB_LIMIT_FILE_CLUSTER+1)],os.path.splitext(list(table[path_column])[index])[0]+format_conversion) for index in range(len_table)]
                filelist = [list_path[index] + "|" + list(table[element_column])[index] + "|" + str(index) for index in range(len_table)]
                total_filelist += filelist
            return total_filelist, data_info, len(user)-1, dir_to_create, list_original_path
    
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
    
    def _concatenate_subtitle(self, list_time, list_subtitle, max_limit_duration, min_limit_duration):
        '''
        Concatenate subtitle to get long sentences and not cut sentences

        Parameters
        ----------
        list_time : list
            list of data containing time
        list_subtitle : list
            list of data containing string
        max_limit_duration : int
            maximum audio length/duration authorized
        min_limit_duration : int
            minimum audio length/duration authorized

        Returns
        -------
        None.

        '''
        align_duration = 10
        new_list_time = []
        new_list_subtitle = []
        
        index = 0
        while index < len(list_subtitle):
            compt = 0
            subtitle = list_subtitle[index]
            beg_time = list_time[index][0]
            end_time = list_time[index][1]
            if index+compt < len(list_subtitle)-1 and (TimePreprocessor().convert_time_format(list_time[index+compt][1]) - TimePreprocessor().convert_time_format(list_time[index+compt][0])) != align_duration:
                #If not out of index range and duration is not the aligned 10 ms one
                while (TimePreprocessor().convert_time_format(list_time[index+compt+1][1]) - TimePreprocessor().convert_time_format(beg_time)) <= max_limit_duration \
                    and list_time[index+compt][1] == list_time[index+compt+1][0] \
                        and list_subtitle[index+compt][-1] not in self.END_CHARS:
                            #Concatenate next subtitles if:
                            #If limit of 10 second
                            #If beginning of next timestamp is the end of the actual timestamp or FOR FUTURE MAYBE ADD CONCATENATION IF DIIF < 10 MS
                            #If actual subtitle does not end with an end chars 
                            if (TimePreprocessor().convert_time_format(list_time[index+compt+1][1]) - TimePreprocessor().convert_time_format(list_time[index+compt+1][0])) != align_duration:
                                #If duration is not 10ms, concatenation is done
                                subtitle += " " + list_subtitle[index+compt+1]
                                end_time = list_time[index+compt+1][1]
                            compt += 1                                    
                            if index+compt >= len(list_subtitle)-1:
                                break
                new_list_time.append((beg_time,end_time))
                new_list_subtitle.append(subtitle)
            index += compt + 1
        
        #Remove audio smaller than min_limit_duration
        new_list_subtitle = [subtitle for index,subtitle in enumerate(new_list_subtitle) if TimePreprocessor().convert_time_format(new_list_time[index][1]) - TimePreprocessor().convert_time_format(new_list_time[index][0]) >= min_limit_duration]
        new_list_time = [time for time in new_list_time if TimePreprocessor().convert_time_format(time[1]) - TimePreprocessor().convert_time_format(time[0]) >= min_limit_duration]
        
        return new_list_time, new_list_subtitle
        
    
    def get_info_from_vtt(self, data, path_cleaner, concatenate=False, max_limit_duration=10000, min_limit_duration=1000):
        '''
        Get time of subtitile and subtitle

        Parameters
        ----------
        data : list
            list of vtt data
        path_cleaner : string
            path of a cleaner (.tsv file)
        concatenate : boolean
            concatenate vtt sentences/subtitles by using time and end characters to make bigger sentence/subtitle
        max_limit_duration : int
            maximum audio length/duration authorized
        min_limit_duration : int
            minimum audio length/duration authorized

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
                    subtitle += data[index + compt]
                    compt += 1
                list_subtitle.append(subtitle)
            index += 1
        
        list_subtitle = DataCleaner().clean_text(data=list_subtitle,
                                            path_cleaner=path_cleaner)
            
        index_to_remove = self._useless_data(list_subtitle)
        #print(index_to_remove)
        list_time = [element[0] for index,element in enumerate(list_time) if index not in index_to_remove]
        list_subtitle = [element for index,element in enumerate(list_subtitle) if index not in index_to_remove]
        #print(list_subtitle[100:400])
        
        '''
        Concatenation of sentence/subtitle
        '''
        if concatenate:
            list_time, list_subtitle = self._concatenate_subtitle(list_time, list_subtitle, max_limit_duration, min_limit_duration)
        
        #print(list_subtitle[100:200])
        return list_time, list_subtitle
    
    def get_ITN_data(self, data_text, data_option=None, regex_match=re.compile('[^a-zA-Z-\']+')):
        '''
        Find ITN/symbols elements in text

        Parameters
        ----------
        data_text : list
            list containing text
        data_option : list
            list containing other data for instance audio path related to the data_text        
        regex_match : string
            regex to use to match required symbols
        
        Returns
        -------
        list
            list of ITN/symbols found

        '''
        
        regex_match_only_digit = re.compile('^\d+\.?$')
        regex_match_punctuation = re.compile('[a-zA-Z]{3,}[,.;:]')
        
        if data_option is not None:
            return [word + "\t" + sentence + "\t" + data_option[index] for index,sentence in enumerate(tqdm(data_text)) for word in sentence.split() if re.search(regex_match,word) is not None and re.search(regex_match_only_digit,word) is None and re.search(regex_match_punctuation,word) is None]
        else:
            return [word + "\t" + sentence for sentence in tqdm(data_text) for word in sentence.split() if re.search(regex_match,word) is not None and re.search(regex_match_only_digit,word) is None and re.search(regex_match_punctuation,word) is None]
        
        