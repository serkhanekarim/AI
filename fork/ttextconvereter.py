#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:49:52 2022

@author: serkhane
"""

from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method
from tqdm import tqdm

import csv
import pandas as pd
import os

list_filename = ["transcripts_test.txt","transcripts_dev.txt","transcripts_train.txt"]
dir_file = "/home/serkhane/Downloads"
list_file = []
list_sentence = []
obj = {'na_filter':False, 'quoting':csv.QUOTE_NONE}
for filename in list_filename:
    data = DataReader(os.path.join(dir_file,filename),filetype="sv",separator="\t").read_data_file(**obj)
    print(data.head())
    exit()
    list_file += list(data[data.columns[0]])
    list_sentence += list(data[data.columns[1]])
    

dir_data_files_converted = os.path.join("/home/serkhane/converted_text",'label')
os.makedirs(dir_data_files_converted,exist_ok=True)
for index, filename in tqdm(enumerate(list_file)):
    DataWriter([list_sentence[index]], os.path.join(dir_data_files_converted,filename+".txt")).write_data_file()
    
# list_arg = [(str(data[data.columns[1]]),
#             ) for audio_path in list_audio_path]


# with Pool(processes=nb_max_parallelized_process) as pool:
#     pool.starmap(AudioPreprocessor().convert_audio, tqdm(list_arg))