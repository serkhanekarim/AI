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
    
    def _extension_filetype_to_writer(self, filetype, separator, newline):
        '''
        Read data file from a related data file extension and type
        
        Parameters
        ----------
        filetype : string
            File extension of a data file
        separator : string
            File extension of a data file
        newline : boolean
            indicate to add new line for each element of data or not (means that \n are already present in data)
                
        Returns
        -------
        string : File type related to the data file extension
        
        '''
        if filetype == "sv":
            return self.data.to_csv(path_or_buf=self.path_file,sep=separator,index=self.index,header=self.header,quoting=csv.QUOTE_NONE)
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
            with open(self.path_file,'w') as FileObj:
                if newline:
                    return FileObj.writelines([element + "\n" for element in self.data])
                else:
                    return FileObj.writelines([element for element in self.data])
                
    def write_data_file(self, newline=True):        
        '''
        Return dataframe from data file data
        
        Parameters
        ----------
        newline : boolean
            indicate to add new line for each element of data or not (means that \n are already present in data)
                
        Returns
        -------
        Dataframe : Pandas dataframe from the file data
        
        '''
        
        print("Writing files...")        
        self.filetype = self.filetype or self._filetype_finder()
        self.separator = self.separator or self._separator_finder() 
        print("Writing files - DONE") 
        
        return self._extension_filetype_to_writer(filetype=self.filetype,separator=self.separator,newline=newline)
    
    def write_edit_data(self, key, value, newline=False):
        '''
        Function that edit some parameter file

        Parameters
        ----------
        key : string
            key to find in the paramater file to change its value.
        value : string
            value used to replace the old value of a paramater wanted to be edited.
        newline : boolean
            indicate to add new line for each element of data or not (means that \n are already present in data)

        Returns
        -------
        list or pandas
            data edited.

        '''
        
        for i,element in enumerate(self.data):
            if key in element:
                self.data[i] = key + value
        self.write_data_file(newline=newline)
        return self.data
                
        
        
        
        
        
        
        
        
        