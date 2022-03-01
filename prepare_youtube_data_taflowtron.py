#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import os
import argparse
import shutil

from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.data import DataPreprocessor
from modules.scraping.media import MediaScraper
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool
import csv
import re
import librosa

def main(args, project_name):
    '''
    Prepare youtube data for Tacotron2 and Flowtron
    
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
    
    USER_CLUSTER = 'ks1'
    DIR_CLUSTER = os.path.join('/home',USER_CLUSTER)
    SEED = 42
    AUDIO_FORMAT = 'wav' #Required audio format for taflowtron
    DATA_FOLDER_NAME = "DATA"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    name_train_param_config = args["name_train_param_config"]
    name_data_config = args["name_data_config"]
    language = args["language"].lower()
    data_directory = args["data_directory"]
    directory_taflowtron_filelist = args["directory_taflowtron_filelist"]
    path_hparam_file = args["path_hparam_file"]
    path_symbols_file = args["path_symbols_file"]
    path_list_url = args["path_list_url"]
    path_youtube_cleaner = args["path_youtube_cleaner"]
    detect_subtitle = args["detect_subtitle"]
    path_local_subtitle_audio = args["path_local_subtitle_audio"]
    converter = args["converter"]
    silence_begend = args["silence_begend"]
    more_silence = args["more_silence"]
    add_silence = args["add_silence"]
    silence_threshold = args["silence_threshold"]
    remove_noise = args["remove_noise"]
    audio_normalization = args["audio_normalization"]
    generated_subtitle = args["generated_subtitle"]
    concatenate_vtt = args["concatenate_vtt"]
    max_limit_duration = args["max_limit_duration"]
    min_limit_duration = args["min_limit_duration"]
    tts_model = args["tts_model"]
    warmstart_model = args["warmstart_model"]
    batch_size = args["batch_size"]
    nb_speaker = args["nb_speaker"]
    
    if data_directory is None: data_directory = directory_of_data
    
    dir_tts_model = os.path.join('models','tts',tts_model)
    dir_cluster_data = os.path.join(DIR_CLUSTER,DATA_FOLDER_NAME,PROJECT_NAME)
    data_information = pd.DataFrame()
    data_filelist = []
    ITN_symbols = []
    voice_id = 0
    total_set_audio_length = 0
    
    obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
    cleaner_youtube = DataReader(path_youtube_cleaner).read_data_file(**obj)
    
    if not detect_subtitle:
        '''
        Get local path
        '''
        source = Method().get_filename(path_local_subtitle_audio)
        obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
        data_path_subtitle_audio = DataReader(path_local_subtitle_audio).read_data_file(**obj)
        list_path_audio = list(data_path_subtitle_audio[data_path_subtitle_audio.columns[0]])
        list_path_subtitle = list(data_path_subtitle_audio[data_path_subtitle_audio.columns[1]])
        list_url = range(len(list_path_subtitle))
    else:
        '''
        Get audio and subtitle from youtube url
        '''
        source = Method().get_filename(path_list_url)
        list_url = DataReader(path_list_url).read_data_file()
        list_url = [line[:-1] for line in list_url]
    
    for url in tqdm(list_url):
        
        '''
        Download audio and subtitle from youtube using youtube-dl
        '''
        if detect_subtitle:
            dir_original_youtube_data = os.path.join(data_directory,language,'original')
            os.makedirs(dir_original_youtube_data,exist_ok=True)
            path_subtitle, path_audio = MediaScraper().get_audio_youtube_data(url=url, 
                                                                              audio_format=AUDIO_FORMAT, 
                                                                              subtitle_language=language, 
                                                                              directory_output=dir_original_youtube_data,
                                                                              generated_subtitle=generated_subtitle)
        else:
            path_subtitle = list_path_subtitle[url]
            path_audio = list_path_audio[url]
            
        if re.search('NO_(MANUAL|GENERATED)_SUBTITLE\.vtt',path_subtitle) is not None:
            continue
        base = os.path.basename(path_audio)
        youtube_code = os.path.splitext(base)[0]
        '''
        Parse subtitles to get trim and text information
        '''
        print("Extracting information from vtt files...")
        data_subtitle = DataReader(path_subtitle).read_data_file()
        #data_subtitle = TextScraper().get_youtube_subtitle(youtube_id=youtube_code, generated_mode=generated_subtitle, language_code=[language])
        list_time, list_subtitle = DataPreprocessor().get_info_from_vtt(data=data_subtitle,
                                                                        cleaner=cleaner_youtube,
                                                                        concatenate=concatenate_vtt,
                                                                        max_limit_duration=max_limit_duration, 
                                                                        min_limit_duration=min_limit_duration,
                                                                        use_youtube_transcript_api=False)
        
        '''
        Trim audio regarding vtt information
        '''
        print("Trimming audio...")
        dir_audio_data_files = os.path.join(data_directory,language,source,name_data_config,youtube_code,'clips')
        os.makedirs(dir_audio_data_files,exist_ok=True)
        path_audio_output = os.path.join(dir_audio_data_files,youtube_code) + "." + AUDIO_FORMAT
        list_trimmed_audio_path = AudioPreprocessor().trim_audio_wav(path_input=path_audio,
                                                                     path_output=path_audio_output,
                                                                     list_time=list_time)
        
        #Fix number of max parallelized process
        nb_max_parallelized_process = min(len(list_trimmed_audio_path), os.cpu_count())
            
        '''
        Remove Noise
        '''
        if remove_noise:
            print("Revoming noise...")
            list_arg = [(audio_path, audio_path) for audio_path in list_trimmed_audio_path]
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().reduce_audio_noise, tqdm(list_arg))
            
        '''
        Normalize audio (Boosting quiet audio)
        '''
        if audio_normalization:
            print("Audio Normalization...")
            list_arg = [(audio_path, audio_path) for audio_path in list_trimmed_audio_path]
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().normalize_audio, tqdm(list_arg))           
            
        '''
        Add and/or Remove leading and trailing silence and/or convert audio
        '''
        if more_silence:
            print("Revoming leading/middle/trailing silence and convert audio...")
            dir_audio_data_files_trimmed = os.path.join(data_directory,language,source,name_data_config,youtube_code,'_temp_clips_trimmed')
            os.makedirs(dir_audio_data_files_trimmed,exist_ok=True)
            list_arg = [(audio_path, 
                         os.path.join(dir_audio_data_files_trimmed,Method().get_filename(audio_path) + "." + AUDIO_FORMAT),
                         True) for audio_path in list_trimmed_audio_path]
            print("Removing all silences found in the middle, at the beginning and at the end...")
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().trim_silence, tqdm(list_arg))
            shutil.rmtree(dir_audio_data_files_trimmed)
            
            print("Removing empty audio...")
            id_audio_to_keep = [index for index in tqdm(range(len(list_trimmed_audio_path))) if librosa.get_duration(filename=list_trimmed_audio_path[index]) != 0]
            list_subtitle = [list_subtitle[index] for index in id_audio_to_keep]
            list_trimmed_audio_path = [list_trimmed_audio_path[index] for index in id_audio_to_keep]
            list_time = [list_time[index] for index in id_audio_to_keep]
            
        if silence_begend:
            list_arg = [(audio_path, audio_path) for audio_path in list_trimmed_audio_path]
            print("Removing all silences found in beginning and at the end...")
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().trim_lead_trail_silence, tqdm(list_arg))
        
        if add_silence:
            print("Padding silence...")
            list_arg = [(audio_path,audio_path,silence_threshold,True,True) for audio_path in tqdm(list_trimmed_audio_path)]
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().add_lead_trail_audio_wav_silence, tqdm(list_arg))
                
        '''
        Convert audio data for taflowtron model
        '''
        if converter:
            print("Audio conversion...")
            dir_audio_data_files_converted = os.path.join(data_directory,language,source,name_data_config,youtube_code,'_temp_clips_converted')
            os.makedirs(dir_audio_data_files_converted,exist_ok=True)
            list_arg = [(audio_path,
                        os.path.join(dir_audio_data_files_converted,Method().get_filename(audio_path) + "." + AUDIO_FORMAT),
                        22050,
                        1,
                        16,
                        True) for audio_path in list_trimmed_audio_path]
            
            with Pool(processes=nb_max_parallelized_process) as pool:
                pool.starmap(AudioPreprocessor().convert_audio, tqdm(list_arg))    
            shutil.rmtree(dir_audio_data_files_converted)
        
        '''
        Get ITN symbols from subtitles
        '''
        print("Getting ITN symbols from data...")
        ITN_symbols += DataPreprocessor().get_ITN_data(data_text=list_subtitle, data_option=list_trimmed_audio_path, language=language)
    
        '''
        Create taflowtron filelist and data information
        '''
        print("Get data information...")
        list_duration = [librosa.get_duration(filename=audio_path) for audio_path in tqdm(list_trimmed_audio_path)]
        
        '''
        Update audio path for cluster
        '''
        list_trimmed_audio_path = [audio_path.replace(data_directory,dir_cluster_data) for audio_path in list_trimmed_audio_path]
        
        #list_duration = [(time[1]-time[0])/1000 for time in list_time]
        list_average_duration = [sum(list_duration)/len(list_duration)]*len(list_duration)
        list_total_video_extraction = [sum(list_duration)/3600]*len(list_duration)
        total_set_audio_length += sum(list_duration)
        mem_info = pd.DataFrame({"Audio Path":list_trimmed_audio_path,
                                 "Text":list_subtitle,
                                 "Speaker ID":voice_id,
                                 "Duration (seconds)":list_duration,
                                 "Average Duration (seconds)":list_average_duration,
                                 "Total Video Extraction Duration (hours)":list_total_video_extraction},
                                columns=["Audio Path","Text","Speaker ID", "Duration (seconds)", "Average Duration (seconds)", "Total Video Extraction Duration (hours)"])
        
        data_information = data_information.append(mem_info)
        
        data_filelist += [list_trimmed_audio_path[index] + "|" + subtitle + "|" + str(voice_id) for index,subtitle in enumerate(list_subtitle)]
        if nb_speaker > 1: voice_id += 1
        
    
    ITN_symbols = set(ITN_symbols)
    data_information["Total Set Extraction Duration (hours)"] = total_set_audio_length/3600
    data_information["Total Set Average Duration (seconds)"] = total_set_audio_length/data_information.shape[0]
        
    '''
    Train, test, validation splitting
    '''
    # In the first step we will split the data in training and remaining dataset
    X_train, X_valid = train_test_split(data_filelist,train_size=0.8, random_state=SEED)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    #X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=SEED)

    '''
    Write Training, Test, and validation file and ITN symbols file and data information
    '''
    filename_train = "youtube_audio_text_train_filelist_" + language + "_" + name_data_config + "_" + source + ".txt"
    filename_valid = "youtube_audio_text_valid_filelist_" + language + "_" + name_data_config + "_" + source + ".txt"
    #filename_test = "youtube_audio_text_test_filelist_" + language + "_" + name_data_config + "_" + source + ".txt"
    filename_ITN_symbols = "youtube_audio_ITN_symbols_" + language + "_" + name_data_config + "_" + source + ".txt"
    filename_data_information = "youtube_audio_data_information_" + language + "_" + name_data_config + "_" + source + ".tsv"

    new_dir_filelist = os.path.join(directory_taflowtron_filelist,'youtube',language,source,name_data_config)
    dir_ITN = os.path.join(directory_of_results,'itn',language,source,name_data_config)
    dir_information = os.path.join(directory_of_results,'data_summary',language,source,name_data_config)
    os.makedirs(new_dir_filelist,exist_ok=True)
    os.makedirs(dir_ITN,exist_ok=True)
    os.makedirs(dir_information,exist_ok=True)
    
    path_train_filelist = os.path.join(new_dir_filelist,filename_train)
    path_valid_filelist = os.path.join(new_dir_filelist,filename_valid)
    #path_test_filelist = os.path.join(new_dir_filelist,filename_test)
    path_ITN_symbols = os.path.join(dir_ITN,filename_ITN_symbols)
    path_data_information = os.path.join(dir_information,filename_data_information)

    DataWriter(X_train, path_train_filelist).write_data_file()
    DataWriter(X_valid, path_valid_filelist).write_data_file()
    #DataWriter(X_test, path_test_filelist).write_data_file()
    DataWriter(ITN_symbols, path_ITN_symbols).write_data_file()
    DataWriter(data_information, path_data_information, header=True).write_data_file()

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

if __name__ == "__main__":
    
    PROJECT_NAME = "youtube_data_taflowtron"
    
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