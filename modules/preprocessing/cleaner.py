#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
import re

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
        cleaner = DataReader(path_cleaner).read_data_file()
        len_cleaner = cleaner.shape[0]

        for index in range(len_cleaner):
            print("Cleaner - Applying regex substitution:" + str(cleaner[0][index]) + "|||" + str(cleaner[1][index]))
            if type(data) == list:
                data = [re.sub(cleaner[0][index],cleaner[1][index],element) for element in data]
            if type(data) == pd.core.frame.DataFrame:
                data = data.apply(lambda element : re.sub(cleaner[0][index],cleaner[1][index],element))
                
        return data
        
                