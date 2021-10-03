#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from modules.Global.variable import Var

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
        
        switcher = Var().SWITCHER_EXTENSION_SEPARATOR
        
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
        
        switcher = Var().SWITCHER_EXTENSION_FILETYPE
        
        return switcher.get(file_extension)
    
    def _separator_finder(self):
        '''
        Find separator from path data file
        
        Parameters
        ----------
        None
                
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
        None
                
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
        filetype : string
            File extension of a data file
        separator : string
            File extension of a data file
                
        Returns
        -------
        string : File type related to the data file extension
        
        '''
        if filetype == "sv":
            return pd.read_csv(filepath_or_buffer=self.path_file, sep=separator)
        if filetype == "excel":
            return pd.read_excel(io=self.path_file)
        if filetype in ["text", "python"]:
            with open(self.path_file,'r') as FileObj:
                return FileObj.readlines()
            
    def read_data_file(self):        
        '''
        Return dataframe from data file data
        
        Parameters
        ----------
        None
                
        Returns
        -------
        Dataframe : Pandas dataframe from the file data
        
        '''
        
        print("Reading files...")        
        self.filetype = self.filetype or self._filetype_finder()
        self.separator = self.separator or self._separator_finder() 
        print("Reading files - DONE") 
        
        return self._extension_filetype_to_reader(filetype=self.filetype,separator=self.separator)
    
    def read_data_value(self, key):
        '''
        Return value of a key from a parameter or any other kind of file

        Parameters
        ----------
        key : string
            key to find its value from a file

        Returns
        -------
        string
            value of the key

        '''
        
        for i,element in enumerate(self.read_data_file()):
            if key in element:
                return element.split(key)[-1]
        
        
        