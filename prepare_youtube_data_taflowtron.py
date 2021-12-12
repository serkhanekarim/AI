#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil

from modules.preprocessing.audio import AudioPreprocessor
from modules.preprocessing.preprocess_audio import preprocess_audio
from modules.preprocessing.data import DataPreprocessor
from modules.scraping.media import MediaScraper
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.preprocessing.time import TimePreprocessor
from modules.Global.method import Method

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main(args):
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
      
    tts_model = args.tts_model
    path_list_url = args.path_list_url
    language = args.language.lower()
    data_directory = args.data_directory
    path_youtube_cleaner = args.path_youtube_cleaner
    directory_taflowtron_filelist = args.directory_taflowtron_filelist
    converter = args.converter.lower() == "true"
    path_hparam_file = args.path_hparam_file
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
    nb_speaker = args.nb_speaker
    
    dir_tts_model = os.path.join('models','tts',tts_model)
    dir_cluster_data = os.path.join(DIR_CLUSTER,DATA_FOLDER_NAME,PROJECT_NAME)
    data_information = pd.DataFrame()
    data_filelist = []
    ITN_symbols = []
    voice_id = 0
    total_set_audio_length = 0
    source = Method().get_filename(path_list_url)
    
    '''
    Get audio and subtitle from youtube url
    '''
    list_url = DataReader(path_list_url).read_data_file()
    list_url = [line[:-1] for line in list_url]
    
    for url in tqdm(list_url):
        
        '''
        Download audio and subtitle from youtube using youtube-dl
        '''
        dir_original_youtube_data = os.path.join(data_directory,language,'original')
        os.makedirs(dir_original_youtube_data,exist_ok=True)
        path_subtitle, path_audio = MediaScraper().get_audio_youtube_data(url=url, 
                                                                          audio_format=AUDIO_FORMAT, 
                                                                          subtitle_language=language, 
                                                                          directory_output=dir_original_youtube_data)
        
        '''
        Parse subtitles to get trim and text information
        '''
        print("Extracting information from vtt files...")
        data_subtitle = DataReader(path_subtitle).read_data_file()
        list_time, list_subtitle = DataPreprocessor().get_info_from_vtt(data=data_subtitle,
                                                                        path_cleaner=path_youtube_cleaner,
                                                                        concatenate=concatenate_vtt,
                                                                        max_limit_duration=max_limit_duration, 
                                                                        min_limit_duration=min_limit_duration)
        list_time = [(TimePreprocessor().convert_time_format(time[0]),TimePreprocessor().convert_time_format(time[1])) for time in list_time]
        
        '''
        Trim audio regarding vtt information
        '''
        print("Trimming audio...")
        base = os.path.basename(path_audio)
        youtube_code = os.path.splitext(base)[0]
        dir_audio_data_files = os.path.join(data_directory,language,source,name_data_config,youtube_code,'clips')
        os.makedirs(dir_audio_data_files,exist_ok=True)
        path_audio_output = os.path.join(dir_audio_data_files,youtube_code) + "." + AUDIO_FORMAT
        list_trimmed_audio_path = AudioPreprocessor().trim_audio_wav(path_input=path_audio,
                                                                     path_output=path_audio_output,
                                                                     list_time=list_time)
        
        '''
        Remove Noise
        '''
        if noise:
            print("Revoming noise...")
            [AudioPreprocessor().reduce_audio_noise(path_input=audio_path,
                                                    path_output=audio_path) for audio_path in tqdm(list_trimmed_audio_path)]
            
        '''
        Normalize audio
        '''
        if audio_normalization:
            print("Audio Normalization...")
            [AudioPreprocessor().normalize_audio(path_input=audio_path,
                                                 path_output=audio_path) for audio_path in tqdm(list_trimmed_audio_path)]
        
        '''
        Add and/or Remove leading and trailing silence and/or convert audio
        '''
        if silence == "remove":
            print("Revoming leading/middle/trailing silence and convert audio...")
            dir_audio_data_files_converted = os.path.join(data_directory,language,source,name_data_config,youtube_code,'clips_trimmed')
            os.makedirs(dir_audio_data_files_converted,exist_ok=True)
            for new_audio_path in tqdm(list_trimmed_audio_path):
                filename = Method().get_filename(new_audio_path)
                path_converted_audio = os.path.join(dir_audio_data_files_converted,filename + "." + AUDIO_FORMAT)
                if not os.path.isfile(path_converted_audio):
                    #Need to update if converted file already exist
                    AudioPreprocessor().trim_silence(path_input=new_audio_path,
                                                     path_output=path_converted_audio)                    
            shutil.rmtree(dir_audio_data_files)
            os.rename(dir_audio_data_files_converted,dir_audio_data_files)
            
            preprocess_audio(file_list=list_trimmed_audio_path,silence_audio_size=0)
        
            # [AudioPreprocessor().remove_lead_trail_audio_wav_silence(path_input=trimmed_audio_path, 
            #                                                          path_output=trimmed_audio_path,
            #                                                          silence_threshold=silence_threshold) for trimmed_audio_path in list_trimmed_audio_path]
        
        if silence == "add":
            print("Padding silence...")
            [AudioPreprocessor().add_lead_trail_audio_wav_silence(path_input=trimmed_audio_path, 
                                                                      path_output=trimmed_audio_path,
                                                                      silence_duration=silence_threshold,
                                                                      before=True, 
                                                                      after=True) for trimmed_audio_path in tqdm(list_trimmed_audio_path)]
        
        '''
        Convert audio data for taflowtron model
        '''
        if (converter or silence == "add") and silence != "remove":
            print("Audio conversion...")
            dir_audio_data_files_converted = os.path.join(data_directory,language,source,name_data_config,youtube_code,'clips_converted')
            os.makedirs(dir_audio_data_files_converted,exist_ok=True)
            for new_audio_path in tqdm(list_trimmed_audio_path):
                filename = Method().get_filename(new_audio_path)
                path_converted_audio = os.path.join(dir_audio_data_files_converted,filename + "." + AUDIO_FORMAT)
                if not os.path.isfile(path_converted_audio):
                    #Need to update if converted file already exist
                    AudioPreprocessor().convert_audio(path_input=new_audio_path, 
                                                      path_output=path_converted_audio,
                                                      sample_rate=22050, 
                                                      channel=1, 
                                                      bits=16)
            shutil.rmtree(dir_audio_data_files)
            os.rename(dir_audio_data_files_converted,dir_audio_data_files)
        
        '''
        Get ITN symbols from subtitles
        '''
        print("Getting ITN symbols from data...")
        ITN_symbols += DataPreprocessor().get_ITN_data(data_text=list_subtitle, data_option=list_trimmed_audio_path)
        
        '''
        Update audio path for cluster
        '''
        list_trimmed_audio_path = [audio_path.replace(directory_of_data,dir_cluster_data) for audio_path in list_trimmed_audio_path]
    
        '''
        Create taflowtron filelist and data information
        '''
        list_duration = [(time[1]-time[0])/1000 for time in list_time]
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
    
#./prepare_youtube_data_taflowtron.py  -path_list_url '/home/serkhane/Repositories/marketing-analysis/modules/scraping/flowtron_youtube_url.txt' -language 'en' -path_youtube_cleaner '/home/serkhane/Repositories/marketing-analysis/modules/preprocessing/cleaners/youtube_subtitle_cleaner_flowtron.tsv' -directory_taflowtron_filelist '/home/serkhane/Repositories/flowtron/filelists' -path_hparam_file '/home/serkhane/Repositories/flowtron/config.json' -converter 'True' -concatenate_vtt 'True' -silence_threshold '-25' -max_limit_duration 10000 -min_limit_duration 1000

    PROJECT_NAME = "youtube_data_taflowtron"
    DATA_FOLDER_NAME = "DATA"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tts_model", help="Which model to use to adapt data", required=True, choices=["flowtron", "tacotron2"] ,nargs='?')
    parser.add_argument("-path_list_url", help="Path of a file containing Youtube urls", required=True, nargs='?')
    parser.add_argument("-language", help="Language to select for subtitle", required=True, nargs='?')
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-path_youtube_cleaner", help="Path of a tsv file containing regex substitution", required=True, nargs='?')
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