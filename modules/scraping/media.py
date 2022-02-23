#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re

from modules.Global.method import Method

class MediaScraper:
    '''
    Class used to scrap media data like video, audio...
    '''
    
    # def __init__(self, dataframe):
    #     self.dataframe = dataframe
    
    def get_audio_youtube_data(self, url, audio_format, subtitle_language, directory_output, generated_subtitle=False):
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
        generated_subtitle : boolean
            If it should download the automatic generated subtitle or not

        Returns
        -------
        tuple
            Download audio and subtitle
            Return the path of the audio and the subtitle

        '''
        #if subtitle_language == "fr": subtitle_language = "fr-FR"
        command_get_id = "youtube-dl " + url + " --get-id"
        youtube_id = subprocess.check_output(command_get_id, text=True, shell=True)[:-1]
        if generated_subtitle: 
            generated_subtitle_string = "_generated_subtitle"
        else: 
            generated_subtitle_string = '_manual_subtitle'
        
        filename_audio = os.path.join(directory_output,youtube_id + "." + audio_format)
        filename_subtitle_downloaded = os.path.join(directory_output,youtube_id + "." + subtitle_language + "." + "vtt") 
        filename_subtitle = os.path.join(directory_output,youtube_id + "." + subtitle_language + generated_subtitle_string + "." + "vtt") 
        
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
        if generated_subtitle:
            command = command.replace("--write-sub","--write-auto-sub")
            
        print("Downloading youtube data...")
        proc = ''
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True, text=True) as proc_temp:
            for b in proc_temp.stdout:
                proc += b
                print(b, end='') # b is the byte from stdout
                
        filename_subtitle_output = re.findall(r'Writing video subtitles to: (.*\.(srt|ass|vtt|lrc))\n',proc)
        if len(filename_subtitle_output) == 0:
            no_subtitle_name = "NO" + generated_subtitle_string.upper() + ".vtt"
            filename_subtitle_output = os.path.join(directory_output,no_subtitle_name)
            with open(filename_subtitle_output, 'w') as fp:
                pass
        else:
            filename_subtitle_output = filename_subtitle_output[0][0]
            filename_subtitle_output = os.path.splitext(filename_subtitle_output)[0] + generated_subtitle_string + os.path.splitext(filename_subtitle_output)[-1]
            os.rename(filename_subtitle_downloaded,filename_subtitle_output)
        path_audio = os.path.join(directory_output,filename_audio)
        
        return filename_subtitle_output, path_audio
    