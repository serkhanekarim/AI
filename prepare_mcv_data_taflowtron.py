#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scipy.io.wavfile import write
import torch
import pandas as pd
import csv

from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.data import DataPreprocessor
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter

from sklearn.model_selection import train_test_split

from tqdm import tqdm
    
def main(args):
    '''
    Prepare Mozilla common voice data for Tacotron2 and Flowtron
    
    ________________________________________________________________________________________________________________
    The Tacotron 2 and WaveGlow model form a text-to-speech system that 
    enables user to synthesise a natural sounding speech from raw 
    transcripts without any additional prosody information. 
    The Tacotron 2 model produces mel spectrograms from input 
    text using encoder-decoder architecture. WaveGlow (also available via torch.hub) 
    is a flow-based model that consumes the mel spectrograms to generate speech.
    This implementation of Tacotron 2 model differs from the model described in the paper. 
    Our implementation uses Dropout instead of Zoneout to regularize the LSTM layers.
    
    ________________________________________________________________________________________________________________ 
    Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis
    Rafael Valle, Kevin Shih, Ryan Prenger and Bryan Catanzaro

    In our recent paper we propose Flowtron: an autoregressive flow-based generative network for text-to-speech 
    synthesis with control over speech variation and style transfer. Flowtron borrows insights from Autoregressive 
    Flows and revamps Tacotron in order to provide high-quality and expressive mel-spectrogram synthesis. 
    Flowtron is optimized by maximizing the likelihood of the training data, which makes training simple and stable. 
    Flowtron learns an invertible mapping of data to a latent space that can be manipulated to control many aspects 
    of speech synthesis (pitch, tone, speech rate, cadence, accent).

    Our mean opinion scores (MOS) show that Flowtron matches state-of-the-art TTS models in terms of speech quality. 
    In addition, we provide results on control of speech variation, interpolation between samples and style transfer 
    between speakers seen and unseen during training.
    '''
    
    SEED = 42
    LIST_AUDIO_FILES = ["test.tsv", "train.tsv", "validated.tsv"]
    USER_COLUMN = "client_id"
    PATH_COLUMN = "path"
    ELEMENT_COLUMN = "sentence"
    OPTION_COLUMN = 'gender'

    directory_file_audio_info = args.directory_file_audio_info
    data_directory = args.data_directory
    language = args.language
    gender = args.gender
    directory_taflowtron_filelist = args.directory_taflowtron_filelist
    path_hparam_file = args.path_hparam_file
    path_symbols_file = args.path_symbols_file
    batch_size = args.batch_size
    file_lister = args.file_lister
    converter = args.converter
    user_informations = args.user_informations
    
    dir_audio_data_files = os.path.join(data_directory,language,'clips')
    dir_audio_data_files_converted = os.path.join(data_directory,language,'clips_converted')
    
    '''
    Aggregation of test, train and validation data file 
    '''
    list_path_audio_files = [os.path.join(directory_file_audio_info,language,file) for file in LIST_AUDIO_FILES]
    data_info = pd.DataFrame()
    for path_file_audio_info in list_path_audio_files:
        data_read = DataReader(path_file_audio_info).read_data_file()
        data_info = data_info.append(data_read, ignore_index = True)
    
    '''
    Conversion of Mozilla Common Voice audio data information into LSJ format for taflowtron training
    '''
    if file_lister.lower() == 'true' or user_informations.lower() == 'true':
        print("Find the max user...")
        data_info_lsj, data_info_user = DataPreprocessor(data_info).convert_data_mcv_to_lsj(user_column=USER_COLUMN, 
                                                                            path_column=PATH_COLUMN, 
                                                                            element_column=ELEMENT_COLUMN,
                                                                            data_directory=dir_audio_data_files,
                                                                            data_directory_converted=dir_audio_data_files_converted,
                                                                            option_column=OPTION_COLUMN,
                                                                            option=gender)
        
        if user_informations.lower() == "true":
            filename_user_information = "mcv_user_voice_informations_" + language + ".tsv"
            DataWriter(data_info_user, os.path.join(directory_of_results,filename_user_information)).write_data_file()
        
         
    '''
    Convert audio data for taflowtron model
    '''
    if converter.lower() == 'true':
        print("Audio conversion...")
        os.makedirs(dir_audio_data_files_converted,exist_ok=True)
        audio_extension_raw = os.path.splitext(data_info[PATH_COLUMN][0])[-1]
        for element in tqdm(list(data_info_lsj)):
            base = os.path.basename(element.split('|')[0])
            filename = os.path.splitext(base)[0]
            AudioPreprocessor().convert_audio(path_input=os.path.join(dir_audio_data_files,filename + audio_extension_raw), 
                                              path_output=element.split('|')[0], 
                                              sample_rate=22050, 
                                              channel=1, 
                                              bits=16)
        
    
    '''
    Train, test, validation splitting
    '''
    
    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem = train_test_split(data_info_lsj,train_size=0.8, random_state=SEED)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=SEED)
    
    '''
    Write Training, Test, and validation file
    '''
    filename_train = "mcv_audio_text_train_filelist_" + language + "_" + str(gender) + ".txt"
    filename_valid = "mcv_audio_text_valid_filelist_" + language + "_" + str(gender) + ".txt"
    filename_test = "mcv_audio_text_test_filelist_" + language + "_" + str(gender) + ".txt"
    
    path_train_filelist = os.path.join(directory_taflowtron_filelist,filename_train)
    path_valid_filelist = os.path.join(directory_taflowtron_filelist,filename_valid)
    path_test_filelist = os.path.join(directory_taflowtron_filelist,filename_test)    
    
    DataWriter(X_train, path_train_filelist).write_data_file()
    DataWriter(X_valid, path_valid_filelist).write_data_file()
    DataWriter(X_test, path_test_filelist).write_data_file()


    if path_hparam_file is not None:
        '''
        Update hparams with filelist and batch size
        '''
        data_haparams = DataReader(path_hparam_file).read_data_file()
        data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        training_files=', value = "'" + path_train_filelist + "',\n")
        data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        validation_files=', value = "'" + path_valid_filelist + "',\n")
        if language == 'en':
            data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        text_cleaners=', value = "['english_cleaners'],\n")
        else:
            data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        text_cleaners=', value = "['basic_cleaners'],\n")
        
        if batch_size is not None:
            data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        batch_size=', value = batch_size + ",\n")

    if path_symbols_file is not None:
        '''
        Update symbols
        '''
        data_symbols = DataReader(path_symbols_file).read_data_file()
        pad = DataReader(path_symbols_file).read_data_value(key="_pad        = ")[1:-1]
        punctuation = DataReader(path_symbols_file).read_data_value(key="_punctuation = ")[1:-1]
        special = DataReader(path_symbols_file).read_data_value(key="_special = ")[1:-1]
        
        unique_char = set("".join(data_info[ELEMENT_COLUMN]))
        unique_char = "".join([char for char in unique_char if char not in pad + punctuation + special])
        unique_char = "".join(set(unique_char.lower() + unique_char.upper()))
        
        DataWriter(data_symbols, path_symbols_file).write_edit_data(key='_letters = ', value = "'" + unique_char + "'\n")
    


if __name__ == "__main__":
    
# 	'''
#     ./model_taflowtron_train.py -directory_file_audio_info '/home/serkhane/Repositories/marketing-analysis/DATA/cv-corpus-7.0-2021-07-21' -language 'kab' 
#     -gender 'female' -directory_taflowtron_filelist '/home/serkhane/Repositories/tacotron2/filelists' -data_directory 
#     -data_directory '/home/serkhane/Repositories/marketing-analysis/DATA/cv-corpus-7.0-2021-07-21' -converter 'False' -file_lister 'False' 
#     -path_hparam_file '/home/serkhane/Repositories/tacotron2/hparams.py' -path_symbols_file '/home/serkhane/Repositories/tacotron2/text/symbols.py' 
#     -batch_size 8 -user_informations 'True'
# 	'''

    PROJECT_NAME = "mcv_data_taflowtron"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-language", help="Language to use for training the TTS", required=True, nargs='?')
    parser.add_argument("-gender", help="Gender to use for training the TTS", nargs='?')
    parser.add_argument("-directory_file_audio_info", help="Directory of the file containing information of user voice", required=True, nargs='?')
    parser.add_argument("-directory_taflowtron_filelist", help="Directory of the file containing information of user voice splitted for Tacotron or Flowtron training", required=True, nargs='?')
    parser.add_argument("-path_hparam_file", help="Path of the file containing the training paramaters", nargs='?')
    parser.add_argument("-path_symbols_file", help="Path of the file containing the symbols", nargs='?')
    parser.add_argument("-file_lister", help="Boolean to create or not the file lister", default="True", nargs='?')
    parser.add_argument("-converter", help="Boolean to convert or not audio file", default="True", nargs='?')
    parser.add_argument("-batch_size", help="Number of batch size", nargs='?')
    parser.add_argument("-user_informations", help="Boolean to get a summary of audio user information", default="False", nargs='?')
    
    args = parser.parse_args()
    
    main(args)    