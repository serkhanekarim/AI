#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
from youtube_transcript_api import YouTubeTranscriptApi

from modules.preprocessing.date import DatePreprocessor
from modules.Global import variable

from tqdm import tqdm

class TextScraper:
    '''
    Class used to scrap data
    '''
    
    DATE_FORMAT = variable.Var().DATE_FORMAT
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def get_youtube_subtitle(self, youtube_id, generated_mode, language_code):
        '''
        Get youtube subtitle using youtube_transcript_api

        Parameters
        ----------
        youtube_id : string
            youtube video ID
        generated_mode : boolean
            Generated or manual subtitle (True of False)
        language_code : list
            list of string containing subtitle language code ['en','fr','ar']

        Returns
        -------
        list
            list containing youtube subtitle information [{text, start, duration}, ...]

        '''
        transcript_list = YouTubeTranscriptApi.list_transcripts(youtube_id)
        if generated_mode:
            return transcript_list.find_generated_transcript(language_code).fetch()
        else:
            return transcript_list.find_manually_created_transcript(language_code).fetch()
            
    
    def add_scraped_stock_data(self,
                         column_name,
                         date_format='%Y-%m-%d', 
                         product="GLD"):
        '''
        Add columns containing stock data from Yahoo Finance

        Parameters
        ----------
        column_name : string
            Name of the date column.
        date_format : string, optional
            Format of the date in the data frame. The default is '%Y-%m-%d'.
        product : string, optional
            Product code to get the stock data. The default is "GLD".

        Returns
        -------
        DataFrame
            Updated dataframe with stock data.

        '''
         
        date_preprocessor = DatePreprocessor()
        df_date = self.dataframe[column_name].apply(lambda date : str(date_preprocessor.convert_date_format(date,date_preprocessor.DATE_FORMAT)))
        
        print("Concatenation of stock data...")
        stock_data = yf.Ticker(product).history(period="max")
        index = df_date.apply(lambda date : stock_data.index.get_loc(date,method='nearest'))
        self.dataframe = self.dataframe.join(stock_data.iloc[index].set_index(self.dataframe.index))
        print("Concatenation of stock data - DONE")
        
        return self.dataframe

        
import requests
from bs4 import BeautifulSoup as bs

class BBCScraper:
    def __init__(self, url:str):
        article = requests.get(url)
        self.soup = bs(article.content, "html.parser")
        self.body = self.get_body()
        self.title = self.get_title()
        
    def get_body(self) -> list:
        body = self.soup.find(property="body")
        print(body)
        return [p.text for p in body.find_all("h3")]
    
    def get_title(self) -> str:
        return self.soup.find(class_="story-body__h1").text        
        
        
        
        
        
        
        
        
        