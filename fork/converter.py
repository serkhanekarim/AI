#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:55:57 2022

@author: serkhane
"""


# token_chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
#                "é", "è", "ê", "ë", 
#                "â", "à", "ä"
#                "ù", "û", "ü",
#                "î","ï",
#                "ô", "ö", 
#                "ÿ",
#                "æ", "œ",
#                "ç",
#                "'", "-"]

# from modules.reader.reader import DataReader

# data = DataReader('/home/serkhane/Downloads/mls_lm_french/data.txt').read_data_file(keep_line_break=False)

# text = set(" ".join(data).lower())

# {' ',
#  "'",
#  '-',
#  'a',
#  'b',
#  'c',
#  'd',
#  'e',
#  'f',
#  'g',
#  'h',
#  'i',
#  'j',
#  'k',
#  'l',
#  'm',
#  'n',
#  'o',
#  'p',
#  'q',
#  'r',
#  's',
#  't',
#  'u',
#  'v',
#  'w',
#  'x',
#  'y',
#  'z',
#  'à',
#  'â',
#  'æ',
#  'ç',
#  'è',
#  'é',
#  'ê',
#  'ë',
#  'î',
#  'ï',
#  'ô',
#  'ù',
#  'û',
#  'ü',
#  'ÿ',
#  'œ'}

import os
import shutil

from modules.Global.method import Method
from modules.preprocessing.audio import AudioPreprocessor

from multiprocessing import Pool
from tqdm import tqdm


AUDIO_FORMAT = "wav"

dir_audio = "/home/serkhane/Downloads/FR"
list_audio_path = [os.path.join(dir_audio, file) for file in os.listdir(dir_audio) if file.endswith(".flac")]

#Fix number of max parallelized process
nb_max_parallelized_process = min(len(list_audio_path), os.cpu_count())

print("Audio conversion...")
dir_audio_data_files_converted = os.path.join("/home/serkhane/converted_mediaspeech",'wav')
os.makedirs(dir_audio_data_files_converted,exist_ok=True)
list_arg = [(audio_path,
            os.path.join(dir_audio_data_files_converted,Method().get_filename(audio_path) + "." + AUDIO_FORMAT),
            16000,
            1,
            16,
            False) for audio_path in list_audio_path]


with Pool(processes=nb_max_parallelized_process) as pool:
    pool.starmap(AudioPreprocessor().convert_audio, tqdm(list_arg))