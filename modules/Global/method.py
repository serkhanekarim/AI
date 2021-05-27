#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DataMethod:
    '''
    Class used to store general method
    '''
    
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