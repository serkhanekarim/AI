#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import xlsxwriter

from modules.Global.variable import Var

class DataWriter:
    '''
    Class used to write file
    '''    
    def __init__(self, data, path_file, filetype=None, separator=None, index=False, header=False):
        self.data = data
        self.path_file = path_file
        self.filetype = filetype
        self.separator = separator
        self.index = index
        self.header = header
        
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
    
    def _extension_filetype_to_writer(self, filetype, separator):
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
            return self.data.to_csv(filepath_or_buf=self.path_file,index=self.index,header=self.header,quoting=csv.QUOTE_NONE,escapechar='\\')
        if filetype == "excel":
            '''
            https://www.geeksforgeeks.org/python-create-and-write-on-excel-file-using-xlsxwriter-module/
            
            # Workbook() takes one, non-optional, argument
            # which is the filename that we want to create.
            workbook = xlsxwriter.Workbook('hello.xlsx')
             
            # The workbook object is then used to add new
            # worksheet via the add_worksheet() method.
            worksheet = workbook.add_worksheet()
             
            # Use the worksheet object to write
            # data via the write() method.
            worksheet.write('A1', 'Hello..')
            worksheet.write('B1', 'Geeks')
            worksheet.write('C1', 'For')
            worksheet.write('D1', 'Geeks')
             
            # Finally, close the Excel file
            # via the close() method.
            workbook.close()
            '''
        if filetype in ["text", "python"]:
            with open(self.path_file) as FileObj:
                return FileObj.writelines([element + "\n" for element in self.data])
            
    def write_data_file(self):        
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
        
        return self._extension_filetype_to_writer(filetype=self.filetype,separator=self.separator)