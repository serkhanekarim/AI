#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from tqdm import tqdm

class Method:
    '''
    Class used to store general method
    '''
    
    def call_func(self, dispatcher, func):
        '''
        Call a function by using a string

        Parameters
        ----------
        dispatcher : dictionnary
            Dictionnary which contains function name to string key.
        func : string
            function to be called

        Returns
        -------
        function value or string
            Return the function to be called or error string

        '''
        try:
            return dispatcher[func]
        except:
            return "Invalid function"
        
    def copy_dir(self, dir_source, dir_destination):
        '''
        Copy paste a directory

        Parameters
        ----------
        dir_source : string
            Source directory.
        dir_destination : string
            Destination directory.

        Returns
        -------
        None.
            Copy paste a directory into a destination directory

        '''
        os.makedirs(dir_destination,exist_ok=True)
        [shutil.copy(os.path.join(dir_source,file_name), dir_destination) for file_name in tqdm(os.listdir(dir_source))]
        
    def get_filename(self, path):
        '''
        Return filename of a path

        Parameters
        ----------
        path : string
            path of a file

        Returns
        -------
        string
            Name of the file without the extension

        '''
        base = os.path.basename(path.split('|')[0])
        return os.path.splitext(base)[0]
    
    @staticmethod    
    def get_new_column_name(length_new_matrix, prefix):
        '''
        Give new column name for a matrix whitout column name

        Parameters
        ----------
        length_new_matrix : int
            Length (number of columns) of a new matrix.
        prefix : string
            Prefix to use for the new column names.

        Returns
        -------
        list
            Return a list of new column name.

        '''
        
        return [prefix + "_" + str(i) for i in range(length_new_matrix)]
    
    @staticmethod
    def feature_type_detector(dataframe, column_name, max_number_of_categorical_occurence):
        '''
        Find the type of feature, categorical, continuous or class number

        Parameters
        ----------
        column_name : string
            Name of the columnn to determine the type of feature.

        Returns
        -------
        string
            Type of feature

        '''
        
        length_value = len(dataframe[column_name])
        length_unique_value = len(set(dataframe[column_name])) 
        
        boolean_feature_class = max_number_of_categorical_occurence >= length_value and \
             length_unique_value <= max_number_of_categorical_occurence and \
                  dataframe[column_name].dtype == "int64"
        
        boolean_feature_continuous = max_number_of_categorical_occurence < length_value and \
             length_unique_value > max_number_of_categorical_occurence and \
                  dataframe[column_name].dtype == "float64"
                  
        boolean_feature_categorical = dataframe[column_name].dtype == "object"
                
        if boolean_feature_class: return 'class'
        if boolean_feature_continuous: return 'continous'
        if boolean_feature_categorical: return 'categorical'
        return -1