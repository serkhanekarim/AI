#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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
    
    def clean_text(self, data, cleaner):
        '''
        Method to clean data using cleaner (tsv file with regex subs)

        Parameters
        ----------
        data : list or pandas dataframe
            table containing text
        cleaner : dataframe
            table containing regex and substitution

        Returns
        -------
        list or pandas dataframe
            cleaned data using cleaner

        '''
        
        len_cleaner = cleaner.shape[0]
        for index in range(len_cleaner):
            regex = re.compile(cleaner[0][index])
            substitution = str(cleaner[1][index])
            #print("Cleaner - Applying regex substitution:" + str(cleaner[0][index]) + "|||" + substitution)
            if type(data) == list:
                data = [re.sub(regex,substitution,element) for element in data]
            elif type(data) == pd.core.series.Series:
                data = data.apply(lambda element : re.sub(regex,substitution,element))
            else:
                sys.exit('Data type not detected for cleaning')
        return data
        
    
            # nb_max_parallelized_process = min(len(data), os.cpu_count())
            # list_arg = [(regex,substitution,element) for element in data]
            # with Pool(processes=nb_max_parallelized_process) as pool:
            #     if type(data) == list:
            #         data = pool.starmap(re.sub, list_arg)
            #     elif type(data) == pd.core.series.Series:
            #         data = pd.Series(pool.starmap(re.sub, list_arg),index=data.index)
            #     else:
            #         sys.exit('Data type not detected for cleaning')
                