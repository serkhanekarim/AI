#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scipy.io.wavfile import write
import torch
import pandas as pd
import csv
import shutil


from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.preprocess_audio import preprocess_audio
from modules.preprocessing.data import DataPreprocessor
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method

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
    USER_CLUSTER = 'ks1'
    DIR_CLUSTER = os.path.join('/home',USER_CLUSTER)
    AUDIO_FORMAT = 'wav' #Required audio format for taflowtron
    LIST_AUDIO_FILES = ["test.tsv", "train.tsv", "validated.tsv"]
    USER_COLUMN = "client_id"
    PATH_COLUMN = "path"
    ELEMENT_COLUMN = "sentence"
    OPTION_COLUMN = 'gender'
    
    tts_model = args.tts_model
    directory_file_audio_info = args.directory_file_audio_info
    data_directory = args.data_directory
    language = args.language.lower()
    gender = args.gender
    path_mcv_cleaner = args.path_mcv_cleaner
    directory_taflowtron_filelist = args.directory_taflowtron_filelist
    converter = args.converter.lower() == "true"
    path_hparam_file = args.path_hparam_file
    path_symbols_file = args.path_symbols_file
    concatenate_vtt = args.concatenate_vtt.lower() == "true"
    silence = args.silence.lower()
    noise = args.noise.lower() == "true"
    audio_normalization = args.audio_normalization.lower() == "true"
    name_train_param_config = args.name_train_param_config
    name_data_config = args.name_data_config
    warmstart_model = args.warmstart_model
    batch_size = args.batch_size
    silence_threshold = args.silence_threshold
    max_limit_duration = args.max_limit_duration
    min_limit_duration = args.min_limit_duration
    
    dir_tts_model = os.path.join('models','tts',tts_model)
    dir_cluster_data = os.path.join(DIR_CLUSTER,DATA_FOLDER_NAME,PROJECT_NAME)
    data_information = pd.DataFrame()
    data_filelist = []
    ITN_symbols = []
    voice_id = 0
    total_set_audio_length = 0
    source = Method().get_filename(path_list_url)
    
    dir_audio_data_files = os.path.join(data_directory,language,'clips')
    dir_audio_data_files_preprocessed = os.path.join(data_directory,language,'clips_preprocessed')
    os.makedirs(dir_audio_data_files_preprocessed,exist_ok=True)
    
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
    print("Find the max user...")
    data_info_mcv, data_info_user, nb_speaker = DataPreprocessor(data_info).convert_data_mcv_to_taflowtron(user_column=USER_COLUMN, 
                                                                                                path_column=PATH_COLUMN, 
                                                                                                element_column=ELEMENT_COLUMN,
                                                                                                data_directory=dir_audio_data_files,
                                                                                                data_directory_converted=dir_audio_data_files_preprocessed,
                                                                                                tts_model=tts_model,
                                                                                                option_column=OPTION_COLUMN,
                                                                                                option=gender)
    list_audio_path = [line.split('|')[0] for line in set(data_info_mcv)]
    
    '''
    Remove Noise
    '''
    if noise:
        print("Revoming noise...")
        [AudioPreprocessor().reduce_audio_noise(path_input=audio_path,
                                                path_output=audio_path) for audio_path in tqdm(list_audio_path)]
        
    '''
    Normalize audio
    '''
    if audio_normalization:
        print("Audio Normalization...")
        [AudioPreprocessor().normalize_audio(path_input=audio_path,
                                             path_output=audio_path) for audio_path in tqdm(list_audio_path)]
    
    '''
    Add and/or Remove leading and trailing silence and/or convert audio
    '''
    if silence == "remove":
        print("Revoming leading/middle/trailing silence and convert audio...")
        for audio_path in tqdm(list_audio_path):
            #Removing middle silence
            filename = Method().get_filename(audio_path)
            path_silence_audio = os.path.join(dir_audio_data_files_preprocessed,filename + "." + AUDIO_FORMAT)
            AudioPreprocessor().trim_silence(path_input=audio_path,
                                                 path_output=path_silence_audio)                    
        shutil.rmtree(dir_audio_data_files)
        os.rename(dir_audio_data_files_preprocessed,dir_audio_data_files)
        
        #REmoving leading and trailing silence
        preprocess_audio(file_list=list_audio_path,silence_audio_size=0)
        
    if silence == "add":
        print("Padding silence...")
        [AudioPreprocessor().add_lead_trail_audio_wav_silence(path_input=audio_path, 
                                                                  path_output=audio_path,
                                                                  silence_duration=silence_threshold,
                                                                  before=True, 
                                                                  after=True) for audio_path in tqdm(list_audio_path)]
        
    '''
    Convert audio data for taflowtron model
    '''
    if (converter or silence == "add") and silence != "remove":
        print("Audio conversion...")
        for audio_path in tqdm(list_audio_path):
            filename = Method().get_filename(audio_path)
            path_converted_audio = os.path.join(dir_audio_data_files_preprocessed,filename + "." + AUDIO_FORMAT)
            AudioPreprocessor().convert_audio(path_input=audio_path, 
                                                  path_output=path_converted_audio,
                                                  sample_rate=22050, 
                                                  channel=1, 
                                                  bits=16)
        shutil.rmtree(dir_audio_data_files)
        os.rename(dir_audio_data_files_preprocessed,dir_audio_data_files)
        
    
    '''
    Train, test, validation splitting
    '''
    
    # In the first step we will split the data in training and remaining dataset
    X_train, X_valid = train_test_split(data_info_mcv,train_size=0.8, random_state=SEED)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    #X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=SEED)
    
    '''
    Write Training, Test, and validation file
    '''
    filename_train = "mcv_audio_text_train_filelist_" + language + "_" + str(gender) + ".txt"
    filename_valid = "mcv_audio_text_valid_filelist_" + language + "_" + str(gender) + ".txt"
    #filename_test = "mcv_audio_text_test_filelist_" + language + "_" + str(gender) + ".txt"
    filename_user_information = "mcv_user_voice_informations_" + language + ".tsv"
    
    path_train_filelist = os.path.join(directory_taflowtron_filelist,filename_train)
    path_valid_filelist = os.path.join(directory_taflowtron_filelist,filename_valid)
    #path_test_filelist = os.path.join(directory_taflowtron_filelist,filename_test)
    path_filename_user_information = os.path.join(directory_of_results,filename_user_information)
    
    DataWriter(X_train, path_train_filelist).write_data_file()
    DataWriter(X_valid, path_valid_filelist).write_data_file()
    #DataWriter(X_test, path_test_filelist).write_data_file()
    DataWriter(data_info_user, path_filename_user_information).write_data_file()
    
    '''
    Update hparams with filelist and batch size
    '''
    if path_hparam_file is not None:
        dir_hparam = os.path.dirname(path_hparam_file)
        new_path_hparam_file = os.path.join(dir_hparam,"config" + "_" + name_train_param_config + ".json")
        path_output_directory = os.path.join(DIR_CLUSTER,dir_tts_model,name_train_param_config,"outdir")
        warmstart_checkpoint_path = os.path.join(DIR_CLUSTER,dir_tts_model,warmstart_model)
        path_cluster_train_filelist = os.path.join(DIR_CLUSTER,'Repositories','AI','modules','tts',tts_model,'filelists','youtube',language,source,name_data_config,filename_train)
        path_cluster_valid_filelist = os.path.join(DIR_CLUSTER,'Repositories','AI','modules','tts',tts_model,'filelists','youtube',language,source,name_data_config,filename_valid)
        
        data_haparams = DataReader(path_hparam_file).read_data_file()
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "output_directory": ', value = '"' + path_output_directory + '",\n')
        if batch_size is not None: data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "batch_size": ', value = ' ' + str(batch_size) + ',\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "warmstart_checkpoint_path": ', value = '"' + warmstart_checkpoint_path + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "training_files": ', value = '"' + path_cluster_train_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "validation_files": ', value = '"' + path_cluster_valid_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "n_speakers": ', value = ' ' + str(voice_id + 1) + ',\n')

    # if path_hparam_file is not None:
    #     '''
    #     Update hparams with filelist and batch size
    #     '''
    #     data_haparams = DataReader(path_hparam_file).read_data_file()
    #     data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        training_files=', value = "'" + path_train_filelist + "',\n")
    #     data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        validation_files=', value = "'" + path_valid_filelist + "',\n")
    #     if language == 'en':
    #         data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        text_cleaners=', value = "['english_cleaners'],\n")
    #     else:
    #         data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        text_cleaners=', value = "['basic_cleaners'],\n")
        
    #     if batch_size is not None:
    #         data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        batch_size=', value = batch_size + ",\n")

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
    DATA_FOLDER_NAME = "DATA"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()    
    parser.add_argument("-tts_model", help="Which model to use to adapt data", required=True, choices=["flowtron", "tacotron2"] ,nargs='?')
    parser.add_argument("-language", help="Language to select for subtitle", required=True, nargs='?')
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-directory_file_audio_info", help="Directory of the file containing information of user voice", required=True, nargs='?')
    parser.add_argument("-path_mcv_cleaner", help="Path of a tsv file containing regex substitution", required=True, nargs='?')
    parser.add_argument("-directory_taflowtron_filelist", help="Directory of the file containing information of user voice splitted for Taflowtron training", required=True, nargs='?')
    parser.add_argument("-path_hparam_file", help="Path of the file containing the training paramaters", required=False, nargs='?')
    parser.add_argument("-path_symbols_file", help="Path of the file containing the symbols", required=False, nargs='?')
    parser.add_argument("-batch_size", help="Number of batch size", required=False, type=int, nargs='?')
    parser.add_argument("-converter", help="Convert or not the audio downliaded from youtube", required=True, nargs='?')
    parser.add_argument("-concatenate_vtt", help="Concatenate subtitles/sentences to avoid small or cut audio subtitles/sentences", required=True, nargs='?')
    parser.add_argument("-silence", help="add or remove silence", required=True, choices=['add', 'remove'],nargs='?')
    parser.add_argument("-silence_threshold", type=int, help="For silence padding, silence threshold in ms and for silence removing, silence threshold in dFBS,lower the value is, les it'll remove the silence", required=False, nargs='?')
    parser.add_argument("-noise", help="Remove noise", required=True,nargs='?')
    parser.add_argument("-audio_normalization", help="Boost quiet audio", required=True,nargs='?')
    parser.add_argument("-max_limit_duration", help="Maximum length authorized of an audion in millisecond", required=False, default=10000, type=int, nargs='?')
    parser.add_argument("-min_limit_duration", help="Minimum length authorized of an audion in millisecond", required=False, default=1000, type=int, nargs='?')
    parser.add_argument("-nb_speaker", help="Number of speaker", required=False, default=0, type=int, nargs='?')
    parser.add_argument("-name_train_param_config", help="Name of the experimentation of the training parameters configuration", required=True, nargs='?')
    parser.add_argument("-name_data_config", help="Name of the experimentation of the training data configuration", required=True, nargs='?')
    parser.add_argument("-warmstart_model", help="Name of the model to use for warmstart", required=True, choices=["flowtron_ljs.pt", "flowtron_libritts2p3k.pt", "tacotron2_statedict.pt"], nargs='?')   
   
    
    args = parser.parse_args()
    
    main(args)    