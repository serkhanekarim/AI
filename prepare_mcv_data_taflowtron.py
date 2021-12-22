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
from multiprocessing import Pool
from functools import partial

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
    LIST_AUDIO_FILES = ["validated.tsv"]
    USER_COLUMN = "client_id"
    PATH_COLUMN = "path"
    ELEMENT_COLUMN = "sentence"
    OPTION_COLUMN = 'gender'
    
    tts_model = args.tts_model
    directory_file_audio_info = args.directory_file_audio_info
    data_directory = args.data_directory
    language = args.language.lower()
    gender = args.gender
    nb_speaker = args.nb_speaker
    path_mcv_cleaner = args.path_mcv_cleaner
    directory_taflowtron_filelist = args.directory_taflowtron_filelist
    path_hparam_file = args.path_hparam_file
    path_symbols_file = args.path_symbols_file
    silence = args.silence.lower()
    noise = args.noise.lower() == "true"
    audio_normalization = args.audio_normalization.lower() == "true"
    name_train_param_config = args.name_train_param_config
    name_data_config = args.name_data_config
    warmstart_model = args.warmstart_model
    batch_size = args.batch_size
    silence_threshold = args.silence_threshold
    
    dir_tts_model = os.path.join('models','tts',tts_model)
    dir_cluster_data = os.path.join(DIR_CLUSTER,DATA_FOLDER_NAME,PROJECT_NAME)
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
    for path_file_audio_info in list_path_audio_files:
        obj = {'na_filter':False, 'quoting':csv.QUOTE_NONE}
        data_read = DataReader(path_file_audio_info).read_data_file(**obj)
        data_info = data_info.append(data_read, ignore_index = True)
    
    '''
    Conversion of Mozilla Common Voice audio data information into taflowtron format
    '''
    print("Find the max user...")
    data_info_mcv, data_info_user, nb_speaker, dir_to_create, list_audio_path_original = DataPreprocessor(data_info).convert_data_mcv_to_taflowtron(user_column=USER_COLUMN,
                                                                                                                                                    path_column=PATH_COLUMN, 
                                                                                                                                                    element_column=ELEMENT_COLUMN,
                                                                                                                                                    data_directory=dir_audio_data_files,
                                                                                                                                                    data_directory_preprocessed=dir_audio_data_files_preprocessed,
                                                                                                                                                    path_cleaner=path_mcv_cleaner,
                                                                                                                                                    tts=tts_model,
                                                                                                                                                    option_column=OPTION_COLUMN,
                                                                                                                                                    option=gender)
    list_audio_path = [line.split('|')[0] for line in list(data_info_mcv)]
    list_audio_path_preprocessing = [os.path.join(dir_audio_data_files_preprocessing,Method().get_filename(audio_path) + "." + AUDIO_FORMAT) for audio_path in list_audio_path_original]
    list_subtitle = [line.split('|')[1] for line in list(data_info_mcv)]
    list_speaker_id = [line.split('|')[2] for line in list(data_info_mcv)]
    
    #Fix number of ,ax parallelized process
    nb_max_parallelized_process = min(len(list_audio_path_original), os.cpu_count())
    
    '''
    Convert audio data for taflowtron model
    '''
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
    if noise:
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
    if silence == "remove":
        print("Revoming leading/middle/trailing silence and convert audio...")
        dir_audio_data_files_trimmed = os.path.join(data_directory,language,'clips_trimmed')
        os.makedirs(dir_audio_data_files_trimmed,exist_ok=True)
        list_audio_path_trimmed = [os.path.join(dir_audio_data_files_trimmed,Method().get_filename(audio_path) + "." + AUDIO_FORMAT) for audio_path in list_audio_path_preprocessing]
        list_arg = [(audio_path, list_audio_path_trimmed[index]) for index, audio_path in enumerate(list_audio_path_preprocessing)]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().trim_silence, tqdm(list_arg))
                 
        shutil.rmtree(dir_audio_data_files_preprocessing)
        os.rename(dir_audio_data_files_trimmed,dir_audio_data_files_preprocessing)
        
        list_arg = [(audio_path, audio_path) for audio_path in list_audio_path_preprocessing]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().trim_lead_trail_silence, tqdm(list_arg))
 
    if silence == "add":
        print("Padding silence...")
        list_arg = [(audio_path,audio_path,silence,silence_threshold,True,True) for audio_path in tqdm(list_audio_path_preprocessing)]
        with Pool(processes=nb_max_parallelized_process) as pool:
            pool.starmap(AudioPreprocessor().add_lead_trail_audio_wav_silence, tqdm(list_arg))
            
   
    '''
    Copying audio files tro keep original and preprocess new one
    '''
    print("Copy paste original data into a preprocessed folder to be preprocessed...")
    [os.makedirs(directory,exist_ok=True) for directory in dir_to_create]
    list_arg = [(path_audio, list_audio_path[index]) for index, path_audio in enumerate(list_audio_path_preprocessing)]
    with Pool(processes=nb_max_parallelized_process) as pool:
        pool.starmap(shutil.copy, tqdm(list_arg))
    shutil.rmtree(dir_audio_data_files_preprocessing)
        
    '''
    Get ITN symbols from subtitles
    '''
    print("Getting ITN symbols from data...")
    ITN_symbols += DataPreprocessor().get_ITN_data(data_text=list_subtitle, data_option=list_audio_path)
    
    '''
    Update audio path for cluster
    '''
    list_audio_path = [audio_path.replace(directory_of_data,dir_cluster_data) for audio_path in list_audio_path]

    '''
    Create taflowtron filelist and data information
    '''
    #list_duration = [(time[1]-time[0])/1000 for time in list_time]
    #list_average_duration = [sum(list_duration)/len(list_duration)]*len(list_duration)
    #list_total_video_extraction = [sum(list_duration)/3600]*len(list_duration)
    #total_set_audio_length += sum(list_duration)
    #mem_info = pd.DataFrame({"Audio Path":list_trimmed_audio_path,
    #                          "Text":list_subtitle,
    #                          "Speaker ID":voice_id,
    #                          "Duration (seconds)":list_duration,
    #                          "Average Duration (seconds)":list_average_duration,
    #                          "Total Video Extraction Duration (hours)":list_total_video_extraction},
    #                         columns=["Audio Path","Text","Speaker ID", "Duration (seconds)", "Average Duration (seconds)", "Total Video Extraction Duration (hours)"])
    
    # data_information = data_information.append(mem_info)
    
    data_filelist += [list_audio_path[index] + "|" + list_subtitle[index] + "|" + list_speaker_id[index] for index in range(len(list_subtitle))]
    
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
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "training_files": ', value = '"' + path_cluster_train_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "validation_files": ', value = '"' + path_cluster_valid_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, new_path_hparam_file).write_edit_data(key='        "n_speakers": ', value = ' ' + str(nb_speaker) + ',\n')

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
    parser.add_argument("-gender", help="Specify gender to use", required=False, nargs='?')
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-directory_file_audio_info", help="Directory of the file containing information of user voice", required=True, nargs='?')
    parser.add_argument("-path_mcv_cleaner", help="Path of a tsv file containing regex substitution", required=True, nargs='?')
    parser.add_argument("-directory_taflowtron_filelist", help="Directory of the file containing information of user voice splitted for Taflowtron training", required=True, nargs='?')
    parser.add_argument("-path_hparam_file", help="Path of the file containing the training paramaters", required=False, nargs='?')
    parser.add_argument("-path_symbols_file", help="Path of the file containing the symbols", required=False, nargs='?')
    parser.add_argument("-batch_size", help="Number of batch size", required=False, type=int, nargs='?')
    parser.add_argument("-silence", help="add or remove silence", required=True, choices=['add', 'remove'],nargs='?')
    parser.add_argument("-silence_threshold", type=int, help="For silence padding, silence threshold in ms and for silence removing, silence threshold in dFBS,lower the value is, les it'll remove the silence", required=False, nargs='?')
    parser.add_argument("-noise", help="Remove noise", required=True,nargs='?')
    parser.add_argument("-audio_normalization", help="Boost quiet audio", required=True,nargs='?')
    parser.add_argument("-nb_speaker", help="Number of speaker", required=False, default=0, type=int, nargs='?')
    parser.add_argument("-name_train_param_config", help="Name of the experimentation of the training parameters configuration", required=True, nargs='?')
    parser.add_argument("-name_data_config", help="Name of the experimentation of the training data configuration", required=True, nargs='?')
    parser.add_argument("-warmstart_model", help="Name of the model to use for warmstart", required=True, choices=["flowtron_ljs.pt", "flowtron_libritts2p3k.pt", "tacotron2_statedict.pt"], nargs='?')   
   
    
    args = parser.parse_args()
    
    main(args)    