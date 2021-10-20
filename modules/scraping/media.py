#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re

from modules.preprocessing.cleaner import DataCleaner

#from pytube import YouTube
#from youtube_transcript_api import YouTubeTranscriptApi
#from tube_dl import Youtube
import youtube_dl

class MediaScraper:
    '''
    Class used to scrap media data like video audio...
    '''
    
    # def __init__(self, dataframe):
    #     self.dataframe = dataframe
    
    def get_audio_youtube_data(self, url, audio_format, subtitle_language, directory_output, path_cleaner=None):
        '''
        Download audio youtube video with subtitle

        Parameters
        ----------
        url : string
            Link of the youtube video
        audio_format : string
            format of the audio
        subtitle_language : string
            language of the subtitle
        directory_output : string
            path of the output of the audio and the subtitle
        path_cleaner : string
            path of cleaner to adapt youtube filename audio

        Returns
        -------
        tuple
            Download audio and subtitle
            Return the path of the audio and the subtitle

        '''
        #command_get_title = "youtube-dl " + url + " --get-title" 
        command_get_id = "youtube-dl " + url + " --get-id"
        #youtube_title = subprocess.check_output(command_get_title, text=True, shell=True)[:-1]
        youtube_id = subprocess.check_output(command_get_id, text=True, shell=True)[:-1]
        
        filename_audio = os.path.join(directory_output,youtube_id + "." + audio_format)
        # filename_audio = DataCleaner().clean_text(data=[filename_audio],
        #                                     path_cleaner=path_cleaner)[0]
        
        #filename_subtitle = os.path.join(directory_output,youtube_title + "-" + youtube_id + "." + subtitle_language + ".vtt")
        
        if os.path.isfile(filename_audio):
            print("File: " + filename_audio + " already exists")
            command = "youtube-dl " + url + " --write-sub --sub-lang " + subtitle_language + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(id)s.%(ext)s") + "' --skip-download"
        else:
            command = "youtube-dl " + url + " --write-sub --sub-lang " + subtitle_language + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(id)s.%(ext)s") + "'"          
            
        print("Downloading youtube data...")
        #os.system(command)
        
        proc = ''
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True, text=True) as proc_temp:
            for b in proc_temp.stdout:
                proc += b
                print(b, end='') # b is the byte from stdout
        # proc = subprocess.Popen([command],
        #                         shell=True,
        #                         stdout=subprocess.PIPE)
            #proc = proc_temp.communicate()[0]
        # print(proc)
        # proc = subprocess.check_output(command, text=True, shell=True)
        filename_subtitle = re.findall(r'Writing video subtitles to: (.*\.(srt|ass|vtt|lrc))\n',proc)[0][0]
        #filename_audio = filename_subtitle.split("." + subtitle_language + ".vtt")[0] + "." + audio_format
        # filename_audio = DataCleaner().clean_text(data=[filename_audio],
        #                                     path_cleaner=path_cleaner)[0]
        
        # print(filename_subtitle)
        # if len(filename_subtitle) == 0:
        #     exit("No original subtitle available")
        #filename_audio = re.findall(r'Destination: (.*' + "\." + audio_format + r')\n',proc)[0]
        #filename_audio = re.findall(r'(.*)\.(srt|ass|vtt|lrc)',filename_subtitle)[0][0] + "." + audio_format
        
        path_subtitle = os.path.join(directory_output,filename_subtitle)
        path_audio = os.path.join(directory_output,filename_audio)
        
        return path_subtitle, path_audio
    