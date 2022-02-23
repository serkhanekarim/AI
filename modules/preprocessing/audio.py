#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import tensorflow as tf
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment, effects
from pydub.silence import detect_leading_silence
from scipy.io import wavfile
import noisereduce as nr
import librosa
import shutil


class AudioPreprocessor:
    '''
    Class used to make audio preprocessing
    '''
        
    def _decode_audio(self, audio_binary):
        '''
        Le fichier audio sera initialement lu comme un fichier binaire, que vous voudrez 
        convertir en un tenseur numérique.
        Pour charger un fichier audio, vous utiliserez tf.audio.decode_wav , qui renvoie 
        l'audio encodé WAV en tant que Tensor et la fréquence d'échantillonnage.
        Un fichier WAV contient des données de séries temporelles avec un nombre défini 
        d'échantillons par seconde. Chaque échantillon représente l'amplitude du signal 
        audio à ce moment précis. Dans un système 16 bits, comme les fichiers dans
        mini_speech_commands , les valeurs vont de -32768 à 32767. La fréquence 
        d'échantillonnage pour cet ensemble de données est de 16kHz. Notez que 
        tf.audio.decode_wav normalisera les valeurs dans la plage [-1.0, 1.0].
        
        Parameters
        ----------
        audio_binary : Tensorflow Tensor
            A tensor of dtype "string", with the audio file contents. 
        
        Returns
        -------
        Tensor
            Decoded 16-bit PCM WAV file to a float tensor then, a Tensor. Has the same type as 
            input. Contains the same data as input, but has one or more dimensions of size 1 removed. 
        
        '''

        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)
    
    def _get_label(self, path_audio):
        '''
        L'étiquette de chaque fichier WAV est son répertoire parent.

        Parameters
        ----------
        path_audio : string
            Path of the audio file

        Returns
        -------
        RaggedTensor
            L'étiquette de chaque fichier WAV est son répertoire parent

        '''
        parts = tf.strings.split(path_audio, os.path.sep)
      
        # Note: You'll use indexing here instead of tuple unpacking to enable this 
        # to work in a TensorFlow graph.
        return parts[-2]


    def get_waveform_and_label(self, path_audio):
        '''
        Method used to convert a date into date_format
        
        Parameters
        ----------
        path_audio : string
            Path of the audio file
        
        Returns
        -------
        Tuple
            Tuple of a (Decoded 16-bit PCM WAV file to a float tensor, WAV file label with parent directory) 
        
        '''
        label = self._get_label(path_audio)
        audio_binary = tf.io.read_file(path_audio)
        waveform = self._decode_audio(audio_binary)
        return waveform, label
    
    def get_spectrogram(self, waveform):
        '''
        Spectrogramme
        
        Vous allez convertir la forme d'onde en un spectrogramme, qui montre les changements de 
        fréquence au fil du temps et peut être représenté sous la forme d'une image 2D. 
        Cela peut être fait en appliquant la transformée de Fourier à court terme (STFT) pour 
        convertir l'audio dans le domaine temps-fréquence.
        Une transformée de Fourier ( tf.signal.fft ) convertit un signal en ses fréquences composantes, 
        mais perd toutes les informations temporelles. Le tf.signal.stft ( tf.signal.stft ) divise le 
        signal en fenêtres temporelles et exécute une transformée de Fourier sur chaque fenêtre, 
        préservant certaines informations temporelles et renvoyant un tenseur 2D sur lequel vous pouvez 
        exécuter des convolutions standard.
        STFT produit un tableau de nombres complexes représentant l'amplitude et la phase. 
        Cependant, vous n'aurez besoin que de la magnitude pour ce didacticiel, qui peut être 
        dérivée en appliquant tf.abs sur la sortie de tf.signal.stft .
        Choisissez les paramètres frame_length et frame_step sorte que l'« image » du spectrogramme 
        généré soit presque carrée. Pour plus d'informations sur le choix des paramètres STFT, 
        vous pouvez vous référer à cette vidéo sur le traitement du signal audio.
        Vous souhaitez également que les formes d'onde aient la même longueur, de sorte que lorsque 
        vous les convertissez en image de spectrogramme, les résultats aient des dimensions similaires. 
        Cela peut être fait en supprimant simplement à zéro les clips audio qui sont plus courts qu'une 
        seconde.

        Parameters
        ----------
        waveform : Tensorflow tensor
            Contains the wave form of an audio

        Returns
        -------
        spectrogram : Tensorflow tensor
            Le tf.signal.stft ( tf.signal.stft ) divise le signal en fenêtres temporelles et 
            exécute une transformée de Fourier sur chaque fenêtre, préservant certaines 
            informations temporelles et renvoyant un tenseur 2D sur lequel vous pouvez exécuter 
            des convolutions standard.

        '''
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
      
        # Concatenate audio with padding so that all audio clips will be of the 
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
      
        spectrogram = tf.abs(spectrogram)
      
        return spectrogram
    
    def get_spectrogram_and_label_id(self, audio, label, labels):
        '''
        Transformez maintenant l'ensemble de données de forme d'onde pour avoir des 
        images de spectrogramme et leurs étiquettes correspondantes en tant qu'ID entiers.

        Parameters
        ----------
        audio : Tensorflow tensor
            Contains the wave form of an audio.
        label : string
            Word said in the audio.
        labels : list
            list of all labels

        Returns
        -------
        spectrogram : Tensor
            Spectrogram of the audio.
        label_id : int
            Index of the audio label

        '''
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == labels)
        return spectrogram, label_id
        
    def preprocess_dataset(self, files, labels):
        '''
        Preocess audio data using above method

        Parameters
        ----------
        files : list
            list of files

        Returns
        -------
        spectrogram : Tensor
            Spectrogram of the audio.
        label_id : int
            Index of the audio label

        '''
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        output_ds = output_ds.map(lambda x,y:
            self.get_spectrogram_and_label_id(x,y,labels),  num_parallel_calls=tf.data.AUTOTUNE)
        return output_ds
    
    def convert_audio(self, path_input, path_output, sample_rate, channel, bits, replace=False):
        '''
        Convert audio file

        Parameters
        ----------
        path_input : string
            audio input path
        path_output : string
            audio output path
        sample_rate : int
            sample rate to convert
        channel : int
            number of channel
        bits : int
            number of bits
        replace : boolean
            Replace or not the orginal audio file by the converted one

        Returns
        -------
        None.
            Convert audio

        '''
        if not os.path.isfile(path_output):
            command = 'sox ' + "'" + path_input + "'" + ' -r ' + str(sample_rate) + ' -c ' + str(channel) + ' -b ' + str(bits) + ' ' + "'" + path_output + "'"
            os.system(command)
            if replace: shutil.move(path_output,path_input)
        else:
            print("File already exist: " + path_output)
        
    def get_audio_length(self, path_audio):
        '''
        Get the audio duration/length in seconds

        Parameters
        ----------
        path_audio : string
            path of the audio file

        Returns
        -------
        duration : float
            audio duration in seconds
        

        '''
        
        #Get audio length
        proc = subprocess.check_output("soxi -D " + path_audio, shell=True)
        return float(proc)
    
    def normalize_audio(self, path_input, path_output):
        '''
        Luckily, PyDub's effects module has a function called normalize() which finds the maximum volume 
        of an AudioSegment, then adjusts the rest of the AudioSegment to be in proportion. This means 
        the quiet parts will get a volume boost.

        Parameters
        ----------
        path_input : string
            path of a waw audio to normalize
        path_output : string
            name of the processed audio

        Returns
        -------
        None.

        '''
        
        rawsound = AudioSegment.from_file(path_input, "wav")  
        normalizedsound = effects.normalize(rawsound)  
        normalizedsound.export(path_output, format="wav")
        
    def remove_lead_trail_audio_wav_silence(self, path_input, path_output, silence_threshold=-50):
        '''
        Method to remove lead and trail silence

        Parameters
        ----------
        path_input : string
            path of a waw audio to remove trail/lead silence
        path_output : string
            path of the processed audio
        silence_threshold : int
            The upper bound for how quiet is silent in dFBS

        Returns
        -------
        None.

        '''

        trim_leading_silence: AudioSegment = lambda x: x[detect_leading_silence(x, silence_threshold=silence_threshold) :]
        trim_trailing_silence: AudioSegment = lambda x: trim_leading_silence(x.reverse()).reverse()
        strip_silence: AudioSegment = lambda x: trim_trailing_silence(trim_leading_silence(x))

        audio = AudioSegment.from_wav(path_input)
        newAudio = strip_silence(audio)
        newAudio.export(path_output, format='wav') #Exports to a wav file in the current path.
        
    def trim_lead_trail_silence(self, path_input, path_output):
        '''
        Trim leading and trailing silence from an audio wav file

        Parameters
        ----------
        path_input : string
            Path of an audio wav file
        path_output : string
            Path of the processed audio

        Returns
        -------
        None.
            Trail and Lead silence removed audio file

        '''
        
        sr = 22050
        max_wav_value=32768.0
        trim_fft_size = 1024
        trim_hop_size = 256
        trim_top_db = 23
        silence_mel_padding = 0
        silence_audio_size = trim_hop_size * silence_mel_padding
        
        #print(path_input)
        data, sampling_rate = librosa.core.load(path_input, sr)
        if data.size !=0:
            data_backup = data
            data = data / np.abs(data).max() *0.999
            data_= librosa.effects.trim(data, top_db= trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
            data_ = data_*max_wav_value
            data_ = np.append(data_, [0.]*silence_audio_size)
            data_ = data_.astype(dtype=np.int16)
            if data.size !=0:
                write(path_input, sr, data_)
            else:
                write(path_input, sr, data_backup)
        
    def trim_silence(self, path_input, path_output, replace=False):
        '''
        Shortening long periods of silence and ignoring noise bursts

        Parameters
        ----------
        path_input : string
            path of a waw audio to trim silence
        path_output : string
            path of the new processed audio (should be different of path_input)
        replace : boolean
            Replace or not the orginal audio file by the preprocessed one

        Returns
        -------
        None.
            Create new trimmed audio file

        '''
        
        command = 'sox ' + path_input + ' ' + path_output + ' silence -l 1 0.1 1% -1 0.5 1%'
        os.system(command)
        if replace: shutil.move(path_output,path_input)
        
        
    def add_lead_trail_audio_wav_silence(self, path_input, path_output, silence_duration=250, before=True, after=True):
        '''
        Method to add lead and/or trail silence

        Parameters
        ----------
        path_input : string
            path of a waw audio to remove trail/lead silence
        path_output : string
            name of the processed audio
        silence_duration: int
            Duration to add in milliseconds
        before: boolean
            Boolean for adding silence at the beginning or not
        after: boolean
            Boolean for adding silence at the ending or not

        Returns
        -------
        None.

        '''
        
        # create silence audio segment
        silence_segment = AudioSegment.silent(duration=silence_duration)  #duration in milliseconds
        
        #read wav file to an audio segment
        audio = AudioSegment.from_wav(path_input)
        
        #Add above two audio segments
        if before:
            audio = silence_segment + audio
        if after:
            audio = audio + silence_segment
        
        #Either save modified audio
        audio.export(path_output, format="wav")
        
    def reduce_audio_noise(self, path_input, path_output):
        '''
        Reduce background noise from audio

        Parameters
        ----------
        path_input : string
            path of a waw audio to trim
        path_output : string
            name of the reduced noise audio

        Returns
        -------
        None.

        '''
        
        data, rate = librosa.core.load(path_input)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wavfile.write(path_output, rate, reduced_noise)

    def trim_audio_wav(self, path_input, path_output, list_time):
        '''
        Trim an audio

        Parameters
        ----------
        path_input : string
            path of a waw audio to trim
        path_output : string
            name of the trimmed audio
        list_time : list
            list of list containing start and end time

        Returns
        -------
        list
            Create the trimmed audio into the path_output and return list of new audio path

        '''
        list_new_audio_path = []
        path_without_extension = os.path.splitext(path_output)[0]
        format_audio = os.path.splitext(path_output)[1].split('.')[-1]
        Audio = AudioSegment.from_wav(path_input)
        for index,time in enumerate(list_time):
            newAudio = Audio[time[0]:time[1]]
            new_path = path_without_extension + "_trim_" + str(time[0]) + "_" + str(time[1]) + "." + format_audio
            list_new_audio_path.append(new_path)
            newAudio.export(new_path, format=format_audio) #Exports to a wav file in the current path.
            
        return list_new_audio_path
            
    # def voice_activity_detection(self, ):
    #     import webrtcvad
    #     vad = webrtcvad.Vad()
        
    #     # Run the VAD on 10 ms of silence. The result should be False.
    #     sample_rate = 16000
    #     frame_duration = 10  # ms
    #     frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
    #     print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))
        

        