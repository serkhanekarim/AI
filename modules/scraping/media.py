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
        command_get_id = "youtube-dl " + url + " --get-id"
        youtube_id = subprocess.check_output(command_get_id, text=True, shell=True)[:-1]
        
        filename_audio = os.path.join(directory_output,youtube_id + "." + audio_format)
        filename_subtitle = os.path.join(directory_output,youtube_id + "." + subtitle_language + "." + "vtt") 
        
        if os.path.isfile(filename_audio) and os.path.isfile(filename_subtitle):
            print("File: " + filename_audio + " already exists")
            print("File: " + filename_subtitle + " already exists")
            path_subtitle = os.path.join(directory_output,filename_subtitle)
            path_audio = os.path.join(directory_output,filename_audio)            
            return path_subtitle, path_audio
            
        if not os.path.isfile(filename_audio) and not os.path.isfile(filename_subtitle):
            command = "youtube-dl " + url + " --write-sub --sub-lang " + subtitle_language + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(id)s.%(ext)s") + "'"
        if not os.path.isfile(filename_audio) and os.path.isfile(filename_subtitle):
            print("File: " + filename_subtitle + " already exists")
            command = "youtube-dl " + url + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(id)s.%(ext)s") + "'"
        if os.path.isfile(filename_audio) and not os.path.isfile(filename_subtitle):
            print("File: " + filename_audio + " already exists")
            command = "youtube-dl " + url + " --write-sub --sub-lang " + subtitle_language + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(id)s.%(ext)s") + "' --skip-download"
            
        print("Downloading youtube data...")
        proc = ''
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True, text=True) as proc_temp:
            for b in proc_temp.stdout:
                proc += b
                print(b, end='') # b is the byte from stdout
                
        filename_subtitle = re.findall(r'Writing video subtitles to: (.*\.(srt|ass|vtt|lrc))\n',proc)
        if len(filename_subtitle) == 0:
            filename_subtitle = "NO_SUBTITLE.vtt"
        else:
            filename_subtitle = filename_subtitle[0][0]
        
        path_subtitle = os.path.join(directory_output,filename_subtitle)
        path_audio = os.path.join(directory_output,filename_audio)
        
        return path_subtitle, path_audio
    