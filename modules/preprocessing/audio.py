#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import tensorflow as tf

from pydub import AudioSegment

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
    
    def convert_audio(self, path_input, path_output, sample_rate, channel, bits):
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

        Returns
        -------
        None.
            Convert audio

        '''
        
        command = 'sox ' + path_input + ' -r ' + str(sample_rate) + ' -c ' + str(channel) + ' -b ' + str(bits) + ' ' + path_output
        os.system(command)
        
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
        
    def trim_audio(self, path_input, path_output, list_time):
        '''
        Trim an audio

        Parameters
        ----------
        path_input : string
            path of the audio to trim
        path_output : string
            name of the trimmed audio
        list_time : list
            list of list containing start and end time

        Returns
        -------
        None.
            Create the trimmed audio into the path_output

        '''
        path_without_extension = os.path.splitext(path_output)[0]
        format_audio = os.path.splitext(path_output)[1]
        newAudio = AudioSegment.from_wav(path_input)
        for index,time in enumerate(list_time):
            newAudio = newAudio[time[0]:time[1]]
            new_path = path_without_extension + "_part_" + str(index) + "." + format_audio
            newAudio.export(path_without_extension, format=format_audio) #Exports to a wav file in the current path.
        
        
        