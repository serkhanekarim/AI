#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import numpy as np

# from datetime import datetime
# import holidays

# from tqdm import tqdm
# from modules.Global import variable

import os
import tensorflow as tf

class AudioPreprocessor:
    '''
    Class used to make audio preprocessing
    '''
    
    #DATE_FORMAT = variable.Var().DATE_FORMAT
    
    # def __init__(self, path_audio):
    #     self.path_audio = path_audio
        
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
    
    def get_spectrogram_and_label_id(self, audio, label, commands):
        '''
        Transformez maintenant l'ensemble de données de forme d'onde pour avoir des 
        images de spectrogramme et leurs étiquettes correspondantes en tant qu'ID entiers.

        Parameters
        ----------
        audio : Tensorflow tensor
            Contains the wave form of an audio.
        label : string
            Word said in the audio.

        Returns
        -------
        spectrogram : Tensor
            Spectrogram of the audio.
        label_id : int
            Index of the audio label

        '''
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id
        
    # def convert_date_format(self, date, date_format):
    #     '''
    #     Method used to convert a date into date_format

    #     Parameters
    #     ----------
    #     date : string
    #         date
    #     date_format : string
    #         Format of a date

    #     Returns
    #     -------
    #     string
    #         Converted date

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     return str(datetime.strptime(date, date_format).strftime(self.DATE_FORMAT))
        
        
    # def _is_weekend(self, date):
    #     '''
    #     Method used to get if a date is weekend or not

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     bool : False or True for the weekend

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan        
    #     return datetime.strptime(date, self.DATE_FORMAT).weekday() >= 5
        
    # def _is_bank_holiday(self, date, country_code="FR"):
    #     '''
    #     Method used to get if a date is a bank holiday or not

    #     Parameters
    #     ----------
    #     date : string
    #         date in string
    #     country_code : string, optional
    #         Code of the country to use for the bank holiday. The default is 'FR'.

    #     Returns
    #     -------
    #     bool : False or True for the bank holiday

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     return date in set(holidays.CountryHoliday(country_code))
    
    # def _is_leap_year(self, date):
    #     '''
    #     Method used to get if a year is leap year or not

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     bool : False or True for the leap year

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
        
    #     year = int(date.split('-')[0])
        
    #     if (year % 4) == 0:
    #         if (year % 100) == 0:
    #             if (year % 400) == 0:
    #                 return True
    #             else:
    #                 return False
    #         else:
    #              return True
    #     else:
    #         return False
    
        
    # def _get_season(self, date):
    #     '''
    #     Method used to get season from date

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     string : string containing season of a date

    #     '''
        
    #     date = str(date)
    #     season = np.nan
        
    #     if bool(date) and date.lower() != "nan":
    #         month = datetime.strptime(date, self.DATE_FORMAT).strftime('%B').lower()
    #         day = int(date.split('-')[2])
    #     else:
    #         return season
        
    #     if month in {'January', 'February', 'March'}:
    #         season = 'winter'
    #     elif month in {'April', 'May', 'June'}:
    #         season = 'spring'
    #     elif month in {'July', 'August', 'September'}:
    #         season = 'summer'
    #     else:
    #         season = 'autumn'
        
    #     if (month == 'march') and (day > 19):
    #         season = 'spring'
    #     elif (month == 'june') and (day > 20):
    #         season = 'summer'
    #     elif (month == 'september') and (day > 21):
    #         season = 'autumn'
    #     elif (month == 'december') and (day > 20):
    #         season = 'winter'
        
    #     return season
    
    # def _get_week_of_year(self, date):
    #     '''
    #     Method used to get quarter of year

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     int : int containing the week of the year

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     return datetime.strptime(date, self.DATE_FORMAT).isocalendar()[1]
    
    # def _get_quarter_of_year(self, date):
    #     '''
    #     Method used to get week or year

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     int : int containing the week of the year

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     return (datetime.strptime(date, self.DATE_FORMAT).month-1)//3
    
    # def _get_elapsed_day(self, date):
    #     '''
    #     Method used to get time elapsed from date to today

    #     Parameters
    #     ----------
    #     date : string
    #         date in string

    #     Returns
    #     -------
    #     int : int containing the number of day elapsed during the day and today

    #     '''
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     return (datetime.now()-datetime.strptime(date, self.DATE_FORMAT)).days
        
    # def _date_extractor_function(self, date, option):
    #     '''
    #     Method used to exctrat day, month and year from a date

    #     Parameters
    #     ----------
    #     date : string
    #         date in string
    #     option : string
    #         day, month or year

    #     Returns
    #     -------
    #     int : int containing day or month or year

    #     '''
        
    #     date = str(date)
    #     if not date or date.lower() == "nan":
    #         return np.nan
    #     if option.lower() == "day":
    #         return int(date.split('-')[2])
    #     if option.lower() == "month":
    #         return datetime.strptime(date, self.DATE_FORMAT).strftime('%B').lower()
    #     if option.lower() == "year":
    #         return int(date.split('-')[0])
    #     if option.lower() == "weekday":
    #         if not date or date.lower() == "nan":
    #             return np.nan
    #         return datetime.strptime(date, self.DATE_FORMAT).strftime('%A').lower()
        
        
    # def extract_date_information(self, columns_name, date_format='%Y-%m-%d'):
    #     '''
    #     Create new day, month, year and other feature column using date column
        
    #     Parameters
    #     ----------
    #     column_name : string or list of string
    #         Column name of the column containing data data
    #     date_format : string, optional
    #         Format of the date. The default is '%Y-%m-%d'.
                
    #     Returns
    #     -------
    #     DataFrame : Pandas DataFrame with new features (years, months, days, bank holiday...)
                            
    #     Examples
    #     --------
    #         0       2014-01-01       2014   1   1
    #         1       2013-01-01       2013   1   1
    #         2       2000-01-01 ----> 2000   1   1
    #         3       2005-01-01       2005   1   1
    #         4       2015-01-01       2015   1   1
    #     '''
        
    #     print("Feature Engineering on date...")     
    #     if isinstance(columns_name, str): columns_name = [columns_name]
    #     for column_name in tqdm(set(columns_name)):
    #         self.dataframe[column_name] = self.dataframe[column_name].apply(self.convert_date_format, date_format=date_format)
    #         self.dataframe[column_name + "_day"] = self.dataframe[column_name].apply(self._date_extractor_function, option="day")
    #         self.dataframe[column_name + "_month"] = self.dataframe[column_name].apply(self._date_extractor_function, option="month")
    #         self.dataframe[column_name + "_year"] = self.dataframe[column_name].apply(self._date_extractor_function, option="year")
    #         self.dataframe[column_name + "_weekday"] = self.dataframe[column_name].apply(self._date_extractor_function, option="weekday")
    #         self.dataframe[column_name + "_season"] = self.dataframe[column_name].apply(self._get_season)
    #         self.dataframe[column_name + "_bank_holiday"] = self.dataframe[column_name].apply(self._is_bank_holiday)
    #         self.dataframe[column_name + "_weekend"] = self.dataframe[column_name].apply(self._is_weekend)
    #         self.dataframe[column_name + "_week_of_year"] = self.dataframe[column_name].apply(self._get_week_of_year)
    #         self.dataframe[column_name + "_quarter_of_year"] = self.dataframe[column_name].apply(self._get_quarter_of_year)
    #         self.dataframe[column_name + "_leap_year"] = self.dataframe[column_name].apply(self._is_leap_year)
    #         self.dataframe[column_name + "_elapsed_day"] = self.dataframe[column_name].apply(self._get_elapsed_day)
    #         self.dataframe.drop(column_name, axis=1, inplace=True)
    #         print("Feature Engineering on date - DONE")  
            
    #     return self.dataframe   