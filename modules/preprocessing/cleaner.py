#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
import re
import csv

from modules.Global.variable import Var
from modules.reader.reader import DataReader

class DataCleaner:
    '''
    Class used to clean text
    '''
    
    def clean_text(self, data, path_cleaner):
        '''
        Method to clean data using cleaner (tsv file with regex subs)

        Parameters
        ----------
        data : list or pandas dataframe
            table containing text
        path_cleaner : string
            path of a cleaner (.tsv file)

        Returns
        -------
        list or pandas dataframe
            cleaned data using cleaner

        '''
        
        print('Start cleaning...')
        obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
        cleaner = DataReader(path_cleaner).read_data_file(**obj)
        len_cleaner = cleaner.shape[0]

        for index in range(len_cleaner):
            regex = re.compile(cleaner[0][index])
            substitution = str(cleaner[1][index])
            print("Cleaner - Applying regex substitution:" + str(cleaner[0][index]) + "|||" + substitution)
            if type(data) == list:
                data = [re.sub(regex,substitution,element) for element in data]
            if type(data) == pd.core.frame.DataFrame:
                data = data.apply(lambda element : re.sub(regex,substitution,element))
                
        return data
        
                