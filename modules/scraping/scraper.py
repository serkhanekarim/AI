#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
from modules.feature_engineering.date import DatePreprocessor
from modules.Global import variable

class DataScraper:
    '''
    Class used to scrap data
    '''
    
    DATE_FORMAT = variable.Var().DATE_FORMAT
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def add_scraped_data(self,
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