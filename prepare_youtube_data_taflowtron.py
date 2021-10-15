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
    AUDIO_FORMAT = 'wav'
       
    url = args.url
    language = args.language
    data_directory = args.data_directory
    
    
    '''
    Get audio and subtitle from youtube url
    '''
    path_subtitle, path_audio = MediaScraper().get_audio_youtube_data(url=url, 
                                                                      audio_format=AUDIO_FORMAT, 
                                                                      subtitle_language=language, 
                                                                      directory_output=data_directory)
    
    '''
    Parse subtitles to get trim and text infotmation
    '''
    data_subtitle = DataReader(path_subtitle).read_data_file()
    data_vtt = DataPreprocessor().get_info_from_vtt(data_subtitle)
    
    print(data_vtt)



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
    parser.add_argument("-url", help="Youtube url", required=True, nargs='?')
    parser.add_argument("-language", help="Language to select for subtitle", required=True, nargs='?')
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False, default=directory_of_data, nargs='?')
    parser.add_argument("-directory_taflowtron_filelist", help="Directory of the file containing information of user voice splitted for Taflowtron training", nargs='?')
    parser.add_argument("-path_hparam_file", help="Path of the file containing the training paramaters", nargs='?')
    parser.add_argument("-path_symbols_file", help="Path of the file containing the symbols", nargs='?')
    parser.add_argument("-batch_size", help="Number of batch size", nargs='?')
    
    args = parser.parse_args()
    
    main(args)    