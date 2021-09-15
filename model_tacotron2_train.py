#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scipy.io.wavfile import write
import torch
import pandas as pd

from modules.preprocessing.data import DataPreprocessor
from modules.reader.reader import DataReader

    
def main(args):
    '''
    Training Tacotron2 with Mozilla common voice data 
    
    Model Description
    The Tacotron 2 and WaveGlow model form a text-to-speech system that 
    enables user to synthesise a natural sounding speech from raw 
    transcripts without any additional prosody information. 
    The Tacotron 2 model produces mel spectrograms from input 
    text using encoder-decoder architecture. WaveGlow (also available via torch.hub) 
    is a flow-based model that consumes the mel spectrograms to generate speech.
    This implementation of Tacotron 2 model differs from the model described in the paper. 
    Our implementation uses Dropout instead of Zoneout to regularize the LSTM layers.
    '''
    
    LIST_AUDIO_FILES = ["test.tsv", "train.tsv", "validated.tsv"]

    directory_file_audio_info = args.directory_file_audio_info
    data_directory = args.data_directory
    
    '''
    Aggregation of test, train and validation data file 
    '''
    list_path_audio_files = [os.path.join(directory_file_audio_info,file) for file in LIST_AUDIO_FILES]
    data_info = pd.DataFrame()
    for path_file_audio_info in list_path_audio_files:
        data_read = DataReader(path_file_audio_info).read_data_file()
        data_info = data_info.append(data_read, ignore_index = True)
    
    '''
    Conversion of Mozilla Common Voice audio data information into LSJ format for tacotron2 training
    '''
    data_info_lsj = DataPreprocessor(data_info).convert_data_mcv_to_lsj(user_column="client_id", 
                                                                        path_column="path", 
                                                                        element_column="sentence",
                                                                        data_directory=data_directory,
                                                                        option_column="gender",
                                                                        option="female")
    
    '''
    Train, test, validation splitting
    '''
    
    
    
    
    
    
    

if __name__ == "__main__":
    
    PROJECT_NAME = "Tacotron2_train"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    parser.add_argument("-directory_file_audio_info", help="Directory of the file containing information of user voice", required=True, nargs='?')
    args = parser.parse_args()
    
    main(args)    