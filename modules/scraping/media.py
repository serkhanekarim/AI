#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re

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
    
    def get_audio_youtube_data(self, url, audio_format, subtitle_language, directory_output):
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

        Returns
        -------
        tuple
            Download audio and subtitle
            Return the path of the audio and the subtitle

        '''
        
        command = "youtube-dl " + url + " --write-sub --sub-lang " + subtitle_language + " --extract-audio --audio-format " + audio_format + " --audio-quality 0 --output " + "'" + os.path.join(directory_output,"%(title)s-%(id)s.%(ext)s") + "'"
        
        print("Downloading youtube data...")
        proc = subprocess.check_output(command, text=True, shell=True)
        filename_subtitle = re.findall(r'Writing video subtitles to: (.*\.(srt|ass|vtt|lrc))\n',proc)[0][0]
        if len(filename_subtitle) == 0:
            exit("No original subtitle available")
        filename_audio = re.findall(r'Destination: (.*' + "\." + audio_format + r')\n',proc)[0]
        #filename_audio = re.findall(r'(.*)\.(srt|ass|vtt|lrc)',filename_subtitle)[0][0] + "." + audio_format
        
        path_subtitle = os.path.join(directory_output,filename_subtitle)
        path_audio = os.path.join(directory_output,filename_audio)
        
        return (path_subtitle, path_audio)
    