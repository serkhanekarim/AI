#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:25:07 2022

@author: serkhane
"""

from modules.preprocessing.audio import AudioPreprocessor

from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method

from tqdm import tqdm

from multiprocessing import Pool

import csv
import os

AUDIO_FORMAT = "wav"

list_filename = ["test/txt/segments","train/txt/segments","valid/txt/segments"]
list_text = ["test/txt/test.fr","train/txt/train.fr","valid/txt/valid.fr"]
list_audio = ["test/wav","train/wav","valid/wav"]
dir_file = "/home/serkhane/Downloads/mtedx_fr/fr-fr/data"
# list_audio_path = [os.path.join(dir_file,list_audio[0],file) for file in os.listdir(os.path.join(dir_file, list_audio[0]))]
# list_audio_path += [os.path.join(dir_file,list_audio[1],file) for file in os.listdir(os.path.join(dir_file, list_audio[1]))]
# list_audio_path += [os.path.join(dir_file,list_audio[2],file) for file in os.listdir(os.path.join(dir_file, list_audio[2]))]

list_file = []
list_time = []
list_youtube_code = []
list_data = []
obj = {'na_filter':False, 'quoting':csv.QUOTE_NONE, 'header':None}
for index, filename in enumerate(list_filename):
    data_info = DataReader(os.path.join(dir_file,filename),filetype="sv",separator=" ").read_data_file(**obj)
    list_file.append(list(data_info[data_info.columns[0]]))
    list_youtube_code.append(list(data_info[data_info.columns[1]]))
    list_time.append([[time_0*1000, time_1*1000] for time_0, time_1 in zip(list(data_info[data_info.columns[2]]), list(data_info[data_info.columns[3]]))])
    list_data.append(DataReader(os.path.join(dir_file,list_text[index])).read_data_file(keep_line_break=False))


dir_audio_data_files = os.path.join("/home/serkhane/converted_tedx",'wav')
os.makedirs(dir_audio_data_files,exist_ok=True)
dir_text_data_files = os.path.join("/home/serkhane/converted_tedx",'label')
os.makedirs(dir_text_data_files,exist_ok=True)

# nb_max_parallelized_process = min(len(list_audio_path), os.cpu_count())
# print("Convert audio...")
# list_arg = [(audio_path,
#             os.path.join(os.path.dirname(audio_path),Method().get_filename(audio_path) + "." + AUDIO_FORMAT),
#             16000,
#             1,
#             16,
#             False) for audio_path in list_audio_path]


# with Pool(processes=nb_max_parallelized_process) as pool:
#     pool.starmap(AudioPreprocessor().convert_audio, tqdm(list_arg))

print("Trimming audio...")
for index, sub_directory in enumerate(list_audio):
    for filename in tqdm(list(dict.fromkeys(list_youtube_code[index]))):
        list_good_index = [index for index, element in enumerate(list_youtube_code[index]) if element==filename]
        path_audio = os.path.join(dir_file,sub_directory,filename + "." + AUDIO_FORMAT)
        path_audio_output = os.path.join(dir_audio_data_files,filename + "." + AUDIO_FORMAT)
        list_trimmed_audio_path = AudioPreprocessor().trim_audio_wav(path_input=path_audio,
                                                                     path_output=path_audio_output,
                                                                     list_time=[list_time[index][i] for i in list_good_index])
        
        for index_trim, trimmed_audio_path in enumerate(list_trimmed_audio_path):
            DataWriter([[list_data[index][i] for i in list_good_index][index_trim]], os.path.join(dir_text_data_files,Method().get_filename(trimmed_audio_path) + ".txt")).write_data_file()
            
            
        