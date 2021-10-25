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
from modules.scraping.media import MediaScraper
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.preprocessing.time import TimePreprocessor

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
    
    
    SEED = 42
    AUDIO_FORMAT = 'wav' #Required audio format for taflowtron
       
    path_list_url = args.path_list_url
    language = args.language
    data_directory = args.data_directory
    path_youtube_cleaner = args.path_youtube_cleaner
    directory_taflowtron_filelist = args.directory_taflowtron_filelist
    converter = args.converter
    path_hparam_file = args.path_hparam_file
    concatenate_vtt = args.concatenate_vtt.lower() == "true"
    
    '''
    Get audio and subtitle from youtube url
    '''
    list_url = DataReader(path_list_url).read_data_file()
    list_url = [line[:-1] for line in list_url]    
    
    data_filelist = []
    voice_id = 0
    
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
        data_subtitle = DataReader(path_subtitle).read_data_file()
        list_time, list_subtitle = DataPreprocessor().get_info_from_vtt(data=data_subtitle,path_cleaner=path_youtube_cleaner,concatenate=concatenate_vtt)
        list_time = [(TimePreprocessor().convert_time_format(time[0]),TimePreprocessor().convert_time_format(time[1])) for time in list_time]
        
        '''
        Trim audio regarding vtt information
        '''
        print("Trimming audio...")
        base = os.path.basename(path_audio)
        filename = os.path.splitext(base)[0]
        dir_audio_data_files = os.path.join(data_directory,language,filename,'clips')
        os.makedirs(dir_audio_data_files,exist_ok=True)
        path_audio_output = os.path.join(dir_audio_data_files,filename) + "." + AUDIO_FORMAT
        list_trimmed_audio_path = AudioPreprocessor().trim_audio_wav(path_input=path_audio, 
                                                                 path_output=path_audio_output, 
                                                                 list_time=list_time)
        
        
        '''
        Remove leading and trailing silence
        '''
        print("Revoming leading and trailing silence...")
        [AudioPreprocessor().remove_lead_trail_audio_wav_silence(path_input=trimmed_audio_path, path_output=trimmed_audio_path) for trimmed_audio_path in list_trimmed_audio_path]
        
        '''
        Convert audio data for taflowtron model
        '''
        if converter.lower() == 'true':
            print("Audio conversion...")
            list_total_new_audio_path = []
            dir_audio_data_files_converted = os.path.join(data_directory,language,filename,'clips_converted')
            os.makedirs(dir_audio_data_files_converted,exist_ok=True)
            for new_audio_path in tqdm(list_trimmed_audio_path):
                base = os.path.basename(new_audio_path.split('|')[0])
                filename = os.path.splitext(base)[0]
                path_converted_audio = os.path.join(dir_audio_data_files_converted,filename + "." + AUDIO_FORMAT)
                list_total_new_audio_path.append(path_converted_audio)
                if not os.path.isfile(path_converted_audio):
                    AudioPreprocessor().convert_audio(path_input=new_audio_path, 
                                                      path_output=path_converted_audio,
                                                      sample_rate=22050, 
                                                      channel=1, 
                                                      bits=16)    
        
        else:
            #Get a full list of all path audio and subtitles for taflowtron filelist
            list_total_new_audio_path = list_trimmed_audio_path
        
    
        '''
        Create taflowtron filelist
        '''
        data_filelist += [list_total_new_audio_path[index] + "|" + subtitle + "|" + str(voice_id) for index,subtitle in enumerate(list_subtitle)]
        voice_id += 1
        
        
    '''
    Train, test, validation splitting
    '''
    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem = train_test_split(data_filelist,train_size=0.8, random_state=SEED)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=SEED)
    
    '''
    Write Training, Test, and validation file
    '''
    filename_train = "youtube_audio_text_train_filelist_" + language + ".txt"
    filename_valid = "youtube_audio_text_valid_filelist_" + language + ".txt"
    filename_test = "youtube_audio_text_test_filelist_" + language + ".txt"
    
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
        data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        "training_files": ', value = '"' + path_train_filelist + '",\n')
        data_haparams = DataWriter(data_haparams, path_hparam_file).write_edit_data(key='        "validation_files": ', value = '"' + path_valid_filelist + '",\n')


if __name__ == "__main__":
    
# 	'''
#     ./model_taflowtron_train.py -directory_file_audio_info '/home/serkhane/Repositories/marketing-analysis/DATA/cv-corpus-7.0-2021-07-21' -language 'kab' 
#     -gender 'female' -directory_tacotron_filelist '/home/serkhane/Repositories/tacotron2/filelists' -data_directory 
#     -data_directory '/home/serkhane/Repositories/marketing-analysis/DATA/cv-corpus-7.0-2021-07-21' -converter 'False' -file_lister 'False' 
#     -path_hparam_file '/home/serkhane/Repositories/tacotron2/hparams.py' -path_symbols_file '/home/serkhane/Repositories/tacotron2/text/symbols.py' 
#     -batch_size 8 -user_informations 'True'
# 	'''

    PROJECT_NAME = "youtube_data_taflowtron"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_list_url", help="Path of a file containing Youtube urls", required=True, nargs='?')
    parser.add_argument("-language", help="Language to select for subtitle", required=True, nargs='?')
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-path_youtube_cleaner", help="Path of a tsv file containing regex substitution", nargs='?')
    parser.add_argument("-directory_taflowtron_filelist", help="Directory of the file containing information of user voice splitted for Taflowtron training", required=True, nargs='?')
    parser.add_argument("-path_hparam_file", help="Path of the file containing the training paramaters", nargs='?')
    parser.add_argument("-path_symbols_file", help="Path of the file containing the symbols", nargs='?')
    parser.add_argument("-batch_size", help="Number of batch size", nargs='?')
    parser.add_argument("-converter", help="Convert or not the audio downliaded from youtube", required=True, nargs='?')
    parser.add_argument("-concatenate_vtt", help="Concatenate subtitles/sentences to avoid small or cut audio subtitles/sentences", required=True, nargs='?')
    parser.add_argument("-silence_threshold", help="Silence threshold in dFBS,lower the value is, les it'll remove the silence", required=False, nargs='?')
    parser.add_argument("-max_limit_duration", help="Maximum length authorized of an audion in millisecond", required=False, nargs='?')
    parser.add_argument("-min_limit_duration", help="Minimum length authorized of an audion in millisecond", required=False, nargs='?')
    
    
    args = parser.parse_args()
    
    main(args)    