#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from modules.preprocessing.date import DatePreprocessor
from modules.Global import variable

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
    
    def youtube_data(self, url):
        '''
        Download youtube video

        Parameters
        ----------
        url : string
            Link of the youtube video
        date_format : string, optional
            Format of the date in the data frame. The default is '%Y-%m-%d'.
        product : string, optional
            Product code to get the stock data. The default is "GLD".

        Returns
        -------
        DataFrame
            Updated dataframe with stock data.

        '''
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
                }],
            'logger': MyLogger(),
            'progress_hooks': [my_hook],
            }
         
        date_preprocessor = DatePreprocessor()
        df_date = self.dataframe[column_name].apply(lambda date : str(date_preprocessor.convert_date_format(date,date_preprocessor.DATE_FORMAT)))
        
        print("Concatenation of stock data...")
        stock_data = yf.Ticker(product).history(period="max")
        index = df_date.apply(lambda date : stock_data.index.get_loc(date,method='nearest'))
        self.dataframe = self.dataframe.join(stock_data.iloc[index].set_index(self.dataframe.index))
        print("Concatenation of stock data - DONE")
        
        return self.dataframe