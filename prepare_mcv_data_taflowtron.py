#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import os
import argparse
import pandas as pd
import csv
import shutil
import re

from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.data import DataPreprocessor
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method

from sklearn.model_selection import train_test_split

from tqdm import tqdm
from multiprocessing import Pool

import librosa

def main(args, project_name):
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
    LIST_AUDIO_FILES = ["validated.tsv"]
    USER_COLUMN = "client_id"
    PATH_COLUMN = "path"
    ELEMENT_COLUMN = "sentence"
    OPTION_COLUMN = 'gender'
    DATA_FOLDER_NAME = "DATA"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    name_train_param_config = args["name_train_param_config"]
    name_data_config = args["name_data_config"]
    language = args["language"].lower()
    gender = args["gender"]
    data_directory = args["data_directory"]
    directory_file_audio_info = args["directory_file_audio_info"]
    path_mcv_cleaner = args["path_mcv_cleaner"]
    directory_taflowtron_filelist = args["directory_taflowtron_filelist"]
    path_hparam_file = args["path_hparam_file"]
    path_symbols_file = args["path_symbols_file"]
    path_speaker_whitelist = args["path_speaker_whitelist"]
    silence_begend = args["silence_begend"]
    more_silence = args["more_silence"]
    add_silence = args["add_silence"]
    silence_threshold = args["silence_threshold"]
    remove_noise = args["remove_noise"]
    audio_normalization = args["audio_normalization"]
    tts_model = args["tts_model"]
    warmstart_model = args["warmstart_model"]
    batch_size = args["batch_size"]
    nb_speaker = args["nb_speaker"]
    
    if data_directory is None: data_directory = directory_of_data
    
    dir_tts_model = os.path.join('models','tts',tts_model)
    dir_cluster_data = os.path.join(DIR_CLUSTER,DATA_FOLDER_NAME)
    #data_information = pd.DataFrame()
    data_filelist = []
    ITN_symbols = []
    #total_set_audio_length = 0
    
    dir_audio_data_files = os.path.join(data_directory,language,'clips')
    dir_audio_data_files_preprocessed = os.path.join(data_directory,language,'clips_preprocessed')
    dir_audio_data_files_preprocessing = os.path.join(data_directory,language,'clips_preprocessing')
    os.makedirs(dir_audio_data_files_preprocessed,exist_ok=True)
    os.makedirs(dir_audio_data_files_preprocessing,exist_ok=True)
    
    '''
    Aggregation of test, train and validation data file
    '''
    list_path_audio_files = [os.path.join(directory_file_audio_info,language,file) for file in LIST_AUDIO_FILES]
    data_info = pd.DataFrame()
    obj = {'na_filter':False, 'quoting':csv.QUOTE_NONE}
    for path_file_audio_info in list_path_audio_files:
        data_read = DataReader(path_file_audio_info).read_data_file(**obj)
        data_info = data_info.append(data_read, ignore_index = True)
    
    '''
    Conversion of Mozilla Common Voice audio data information into taflowtron format
    '''
    print("Collect information from MCV data...")
    obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
    cleaner_mcv = DataReader(path_mcv_cleaner).read_data_file(**obj)
    if path_speaker_whitelist is not None:
        speaker_whitelist = DataReader(path_speaker_whitelist).read_data_file(**obj)
        speaker_whitelist = [re.sub('\\n','',element) for element in speaker_whitelist]
    else:
        speaker_whitelist = None
    list_audio_path, list_subtitle, list_speaker_id, data_info_user, nb_speaker, dir_to_create, list_audio_path_original = DataPreprocessor(data_info).convert_data_mcv_to_taflowtron(user_column=USER_COLUMN,
                                                                                                                                                                                        path_column=PATH_COLUMN, 
                                                                                                                                                                                        element_column=ELEMENT_COLUMN,
                                                                                                                                                                                        data_directory=dir_audio_data_files,
                                                                                                                                                                                        data_directory_preprocessed=dir_audio_data_files_preprocessed,
                                                                                                                                                                                        cleaner=cleaner_mcv,
                                                                                                                                                                                        tts=tts_model,
                                                                                                                                                                                        option_column=OPTION_COLUMN,
                                                                                                                                                                                        option_value=gender,
                                                                                                                                                                                        speaker_whitelist=speaker_whitelist)

    list_audio_path_preprocessing = [os.path.join(dir_audio_data_files_preprocessing,Method().get_filename(audio_path) + "." + AUDIO_FORMAT) for audio_path in list_audio_path_original]

    
    #Fix number of max parallelized process
    nb_max_parallelized_process = min(len(list_audio_path_original), os.cpu_count())
    
    '''
    Convert audio data for taflowtron model
    '''
    print("Audio conversion...")
    list_arg = [(audio_path,
                list_audio_path_preprocessing[index],
                22050,
                1,
                16) for index, audio_path in enumerate(list_audio_path_original)]
    
    with Pool(processes=nb_max_parallelized_process) as pool:
        pool.starmap(AudioPreprocessor().convert_audio, tqdm(list_arg))

    '''
    Remove Noise
    '''
    if remove_noise:
        print("Revoming noise...")
        list_arg = [(audio_path, audio_path) for audio_path in list_audio_path_preprocessing]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().reduce_audio_noise, tqdm(list_arg))   
    
    '''
    Normalize audio
    '''
    if audio_normalization:
        print("Audio Normalization...")
        list_arg = [(audio_path, audio_path) for audio_path in list_audio_path_preprocessing]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().normalize_audio, tqdm(list_arg))
            
    '''
    Add and/or Remove leading and trailing silence and/or convert audio
    '''
    if more_silence:
        print("Revoming leading/middle/trailing silence and convert audio...")
        dir_audio_data_files_trimmed = os.path.join(data_directory,language,'_temp_clips_trimmed')
        os.makedirs(dir_audio_data_files_trimmed,exist_ok=True)
        list_arg = [(audio_path,
                      os.path.join(dir_audio_data_files_trimmed,Method().get_filename(audio_path) + "." + AUDIO_FORMAT),
                      True) for audio_path in list_audio_path_preprocessing]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().trim_silence, tqdm(list_arg))
        shutil.rmtree(dir_audio_data_files_trimmed)
        
        print("Removing empty audio...")
        id_audio_to_keep = [index for index in tqdm(range(len(list_audio_path_preprocessing))) if librosa.get_duration(filename=list_audio_path_preprocessing[index]) != 0]
        list_subtitle = [list_subtitle[index] for index in id_audio_to_keep]
        list_audio_path_preprocessing = [list_audio_path_preprocessing[index] for index in id_audio_to_keep]
    
    if silence_begend:    
        list_arg = [(audio_path, audio_path) for audio_path in list_audio_path_preprocessing]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().trim_lead_trail_silence, tqdm(list_arg))
 
    if add_silence:
        print("Padding silence...")
        list_arg = [(audio_path,audio_path,silence_threshold,True,True) for audio_path in tqdm(list_audio_path_preprocessing)]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().add_lead_trail_audio_wav_silence, tqdm(list_arg))
            
   
    '''
    Copying audio files into ImageNET tree style
    '''
    print("Copying audio files into ImageNET tree style...")
    [os.makedirs(directory,exist_ok=True) for directory in dir_to_create]
    list_arg = [(path_audio, list_audio_path[index]) for index, path_audio in enumerate(list_audio_path_preprocessing)]
    with Pool(processes=nb_max_parallelized_process) as pool:
        pool.starmap(shutil.move, tqdm(list_arg))
    shutil.rmtree(dir_audio_data_files_preprocessing)
        
    '''
    Get ITN symbols from subtitles
    '''
    print("Getting ITN symbols from data...")
    ITN_symbols = DataPreprocessor().get_ITN_data(data_text=list_subtitle, data_option=list_audio_path)
    
    '''
    Update audio path for cluster
    '''
    list_audio_path = [audio_path.replace(data_directory,dir_cluster_data) for audio_path in list_audio_path]

    '''
    Create taflowtron filelist
    '''
    data_filelist = [list_audio_path[index] + "|" + list_subtitle[index] + "|" + list_speaker_id[index] for index in range(len(list_subtitle))]
    
    '''
    Train, test, validation splitting
    '''
    
    # In the first step we will split the data in training and remaining dataset
    X_train, X_valid = train_test_split(data_filelist,train_size=0.8, random_state=SEED)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    #X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=SEED)
    
    '''
    Write Training, Test, and validation file
    '''
    filename_train = "mcv_audio_text_train_filelist_" + language + "_" + str(gender) + ".txt"
    filename_valid = "mcv_audio_text_valid_filelist_" + language + "_" + str(gender) + ".txt"
    #filename_test = "mcv_audio_text_test_filelist_" + language + "_" + str(gender) + ".txt"
    filename_ITN_symbols = "mcv_audio_ITN_symbols_" + language + "_" + name_data_config + "_" + ".txt"
    filename_user_information = "mcv_user_voice_informations_" + language + ".tsv"
    
    new_dir_filelist = os.path.join(directory_taflowtron_filelist,'mcv',language,name_data_config)
    dir_ITN = os.path.join(directory_of_results,'itn',language,"mcv",name_data_config)
    dir_information = os.path.join(directory_of_results,'data_summary',language,"mcv",name_data_config)
    os.makedirs(new_dir_filelist,exist_ok=True)
    os.makedirs(dir_ITN,exist_ok=True)
    os.makedirs(dir_information,exist_ok=True)
    
    path_train_filelist = os.path.join(new_dir_filelist,filename_train)
    path_valid_filelist = os.path.join(new_dir_filelist,filename_valid)
    #path_test_filelist = os.path.join(directory_taflowtron_filelist,filename_test)
    path_filename_user_information = os.path.join(dir_information,filename_user_information)
    path_ITN_symbols = os.path.join(dir_ITN,filename_ITN_symbols)
    
    DataWriter(X_train, path_train_filelist).write_data_file()
    DataWriter(X_valid, path_valid_filelist).write_data_file()
    #DataWriter(X_test, path_test_filelist).write_data_file()
    DataWriter(ITN_symbols, path_ITN_symbols).write_data_file()
    DataWriter(data_info_user, path_filename_user_information).write_data_file()
    
    '''
    Update hparams with filelist and batch size
    '''
    if path_hparam_file is not None:
        dir_hparam = os.path.dirname(path_hparam_file)
        new_path_hparam_file = os.path.join(dir_hparam,"config" + "_" + name_train_param_config + ".json")
        path_output_directory = os.path.join(DIR_CLUSTER,dir_tts_model,name_train_param_config,"outdir")
        warmstart_checkpoint_path = os.path.join(DIR_CLUSTER,dir_tts_model,warmstart_model)
        path_cluster_train_filelist = os.path.join(DIR_CLUSTER,'Repositories','AI','modules','tts',tts_model,'filelists','mcv',language,name_data_config,filename_train)
        path_cluster_valid_filelist = os.path.join(DIR_CLUSTER,'Repositories','AI','modules','tts',tts_model,'filelists','mcv',language,name_data_config,filename_valid)
        
        data_haparams = DataReader(path_hparam_file).read_data_file()
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "output_directory": ', value = '"' + path_output_directory + '",\n')
        if batch_size is not None: data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "batch_size": ', value = ' ' + str(batch_size) + ',\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "warmstart_checkpoint_path": ', value = '"' + warmstart_checkpoint_path + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "language": ', value = '"' + language + '"\n')        
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "training_files": ', value = '"' + path_cluster_train_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "validation_files": ', value = '"' + path_cluster_valid_filelist + '",\n')
        # data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "cmudict_path": ', value = '"' + 'data/data' + '_' + language + '/cmudict_dictionary' + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "n_speakers": ', value = str(nb_speaker) + ',\n')

    if path_symbols_file is not None:
        '''
        Update symbols
        '''
        if tts_model == "tacotron2":
            data_symbols = DataReader(path_symbols_file).read_data_file()
            pad = DataReader(path_symbols_file).read_data_value(key="_pad        = ")[1:-1]
            punctuation = DataReader(path_symbols_file).read_data_value(key="_punctuation = ")[1:-1]
            special = DataReader(path_symbols_file).read_data_value(key="_special = ")[1:-1]
            not_letter = pad + punctuation + special
            
            unique_char = set("".join(data_info[ELEMENT_COLUMN]))
            unique_char = "".join([char for char in unique_char if char not in not_letter])
            unique_char = "".join(set(unique_char.lower() + unique_char.upper()))
            
            DataWriter(data_symbols, path_symbols_file).write_edit_data(key='_letters = ', value = "'" + unique_char + "'\n")
        if tts_model == "flowtron":
            data_symbols = DataReader(path_symbols_file).read_data_file()
            punctuation = DataReader(path_symbols_file).read_data_value(key="_punctuation = ")[1:-1]
            math = DataReader(path_symbols_file).read_data_value(key="_math = ")[1:-1]
            special = DataReader(path_symbols_file).read_data_value(key="_special = ")[1:-1]
            accented = DataReader(path_symbols_file).read_data_value(key="_accented = ")[1:-1]
            numbers = DataReader(path_symbols_file).read_data_value(key="_numbers = ")[1:-1]
            not_letter = punctuation + math + special + accented + numbers
            
            unique_char = set("".join(data_info[ELEMENT_COLUMN]))
            unique_char = "".join([char for char in unique_char if char not in not_letter])
            unique_char = "".join(set(unique_char.lower() + unique_char.upper()))
            
            DataWriter(data_symbols, path_symbols_file).write_edit_data(key='_letters = ', value = "'" + unique_char + "'\n")

if __name__ == "__main__":
    
    PROJECT_NAME = "mcv_data_taflowtron"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0
    
    with open(args.config) as f:
        data = f.read()
    
    args_config = json.loads(data)[PROJECT_NAME]
    args_config = Method().update_params(args_config, args.params)
    
    main(args=args_config, project_name=PROJECT_NAME)