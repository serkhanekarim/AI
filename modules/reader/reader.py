#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt


class DataReader:
    '''
    Class used to read file data using pandas module
    '''    
    def __init__(self, path_file, filetype=None, separator=None):
        self.path_file = path_file
        self.filetype = filetype
        self.separator = separator
        
    def _extension_to_separator(self, file_extension):
        '''
        Give separator from a related data file extension
        
        Parameters
        ----------
        file_extension : string
            File extension of a data file
                
        Returns
        -------
        string : Separator related to the data file extension
        
        '''
        
        switcher = {
            "csv": ",",
            "tsv": "\t"
        }
        
        return switcher.get(file_extension)
    
    def _extension_to_filetype(self,file_extension):
        '''
        Give file type from a related data file extension
        
        Parameters
        ----------
        file_extension : string
            File extension of a data file
                
        Returns
        -------
        string : File type related to the data file extension
        
        '''
        
        switcher = {
            "csv": "sv",
            "tsv": "sv",
            "xlsx": "excel"
        }
        
        return switcher.get(file_extension)
    
    def _separator_finder(self):
        '''
        Find separator from path data file
        
        Parameters
        ----------
        path_file : string
            Path of data file
                
        Returns
        -------
        Dataframe : Pandas dataframe from the file data
        
        '''
        
        return self._extension_to_separator(self.path_file.split(".")[-1])
    
    def _filetype_finder(self):
        '''
        Find type of file data
        
        Parameters
        ----------
        path_file : string
            Path of data file
                
        Returns
        -------
        Dataframe : Pandas dataframe from the file data
        
        '''
        
        return self._extension_to_filetype(self.path_file.split(".")[-1])
    
    def _extension_filetype_to_reader(self, filetype, separator):
        '''
        Read data file from a related data file extension and type
        
        Parameters
        ----------
        file_extension : string
            File extension of a data file
                
        Returns
        -------
        string : File type related to the data file extension
        
        '''
        if filetype == "sv":
            return pd.read_csv(filepath_or_buffer=self.path_file, sep=separator)
        if filetype == "excel":
            return pd.read_excel(io=self.path_file)
            
    def read_data_file(self):        
        '''
        Return dataframe from data file data
        
        Parameters
        ----------
        path : string
            Path of data file
        feature : string
            separator used in the file data (',' or '\t' or any else string)
                
        Returns
        -------
        Dataframe : Pandas dataframe from the file data
        
        '''
        
        print("Reading files...")        
        self.filetype = self.filetype or self._filetype_finder()
        self.separator = self.separator or self._separator_finder() 
        print("Reading files - DONE") 
        
        return self._extension_filetype_to_reader(filetype=self.filetype,separator=self.separator)