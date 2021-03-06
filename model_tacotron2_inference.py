#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scipy.io.wavfile import write
import torch

    
def main(args):
    '''
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
    
    data_directory = args.data_directory
    audio_filename = args.audio_filename
    sentence = args.sentence
    
    
    '''
    Load the Tacotron2 model pre-trained on LJ Speech dataset and prepare it for inference:
    '''
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2 = tacotron2.to('cuda')
    print(tacotron2.eval())
    
    '''
    Load pretrained WaveGlow model
    '''
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    print(waveglow.eval())
    
    '''
    Now, let’s make the model say:
    '''
    #text = "Hello world, I missed you so much."
    text = sentence
    
    '''
    Format the input using utility methods
    '''
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    sequences, lengths = utils.prepare_input_sequence([text])
    
    with torch.no_grad():
        mel, _, _ = tacotron2.infer(sequences, lengths)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050

    print(len(audio_numpy))
    print(audio_numpy)
    
    
    '''
    You can write it to a file and listen to it
    '''
    write(os.path.join(directory_of_results,audio_filename + ".wav"), rate, audio_numpy)

if __name__ == "__main__":
    
    MODEL_NAME = "speech-recognition-cnn.model"
    AUDIO_FILENAME_DEFAULT = "generated_audio_tacotron"
    SENTENCE_DEFAULT = "Hello world, I missed you so much."
    PROJECT_NAME = "Tacotron2"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_model = os.path.join(directory_of_script,"model")
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_model,exist_ok=True)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    parser.add_argument("-audio_filename", help="Filename of the audio", required=False,  default=AUDIO_FILENAME_DEFAULT, nargs='?')
    parser.add_argument("-sentence", help="Sentence to generate", required=False,  default=SENTENCE_DEFAULT, nargs='?')
    args = parser.parse_args()
    
    main(args)    