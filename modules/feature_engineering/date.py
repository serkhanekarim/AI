#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from datetime import datetime
import holidays

class DatePreprocessor:
    '''
    Class used to make feature engineering on date data
    '''
    
    DATE_FORMAT = '%Y-%m-%d'
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def convert_date_format(self, date, date_format):
        '''
        Method used to convert a date into date_format

        Parameters
        ----------
        date : string
            date
        date_format : string
            Format of a date

        Returns
        -------
        string
            Converted date

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        return str(datetime.strptime(date, date_format).strftime(self.DATE_FORMAT))
        
        
    def _is_weekend(self, date):
        '''
        Method used to get if a date is weekend or not

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        bool : False or True for the weekend

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan        
        return datetime.strptime(date, self.DATE_FORMAT).weekday() >= 5
        
    def _is_bank_holiday(self, date, country_code="FR"):
        '''
        Method used to get if a date is a bank holiday or not

        Parameters
        ----------
        date : string
            date in string
        country_code : string, optional
            Code of the country to use for the bank holiday. The default is 'FR'.

        Returns
        -------
        bool : False or True for the bank holiday

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        return date in holidays.CountryHoliday(country_code)
    
    def _is_leap_year(self, date):
        '''
        Method used to get if a year is leap year or not

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        bool : False or True for the leap year

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        
        year = int(date.split('-')[0])
        
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    return True
                else:
                    return False
            else:
                 return True
        else:
            return False
    
        
    def _get_season(self, date):
        '''
        Method used to get season from date

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        string : string containing season of a date

        '''
        
        date = str(date)
        season = np.nan
        
        if bool(date) and date.lower() != "nan":
            month = datetime.strptime(date, self.DATE_FORMAT).strftime('%B').lower()
            day = int(date.split('-')[2])
        else:
            return season
        
        if month in ('January', 'February', 'March'):
            season = 'winter'
        elif month in ('April', 'May', 'June'):
            season = 'spring'
        elif month in ('July', 'August', 'September'):
            season = 'summer'
        else:
            season = 'autumn'
        
        if (month == 'march') and (day > 19):
            season = 'spring'
        elif (month == 'june') and (day > 20):
            season = 'summer'
        elif (month == 'september') and (day > 21):
            season = 'autumn'
        elif (month == 'december') and (day > 20):
            season = 'winter'
        
        return season
    
    def _get_week_of_year(self, date):
        '''
        Method used to get quarter of year

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        int : int containing the week of the year

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        return datetime.strptime(date, self.DATE_FORMAT).isocalendar()[1]
    
    def _get_quarter_of_year(self, date):
        '''
        Method used to get week or year

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        int : int containing the week of the year

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        return (datetime.strptime(date, self.DATE_FORMAT).month-1)//3
    
    def _get_elapsed_day(self, date):
        '''
        Method used to get time elapsed from date to today

        Parameters
        ----------
        date : string
            date in string

        Returns
        -------
        int : int containing the number of day elapsed during the day and today

        '''
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        return (datetime.now()-datetime.strptime(date, self.DATE_FORMAT)).days
        
    def _date_extractor_function(self, date, option):
        '''
        Method used to exctrat day, month and year from a date

        Parameters
        ----------
        date : string
            date in string
        option : string
            day, month or year

        Returns
        -------
        int : int containing day or month or year

        '''
        
        date = str(date)
        if not date or date.lower() == "nan":
            return np.nan
        if option.lower() == "day":
            return int(date.split('-')[2])
        if option.lower() == "month":
            return datetime.strptime(date, self.DATE_FORMAT).strftime('%B').lower()
        if option.lower() == "year":
            return int(date.split('-')[0])
        if option.lower() == "weekday":
            if not date or date.lower() == "nan":
                return np.nan
            return datetime.strptime(date, self.DATE_FORMAT).strftime('%A').lower()
        
        
    def extract_date_information(self, column_name, date_format='%Y-%m-%d'):
        '''
        Create new day, month, year and other feature column using date column
        
        Parameters
        ----------
        column_name : string or list of string
            Column name of the column containing data data
        date_format : string, optional
            Format of the date. The default is '%Y-%m-%d'.
                
        Returns
        -------
        DataFrame : Pandas DataFrame with new features (years, months, days, bank holiday...)
                            
        Examples
        --------
            0       2014-01-01       2014   1   1
            1       2013-01-01       2013   1   1
            2       2000-01-01 ----> 2000   1   1
            3       2005-01-01       2005   1   1
            4       2015-01-01       2015   1   1
        '''
        
        print("Feature Engineering on date...")     
        if isinstance(column_name, str): columns_name = set(column_name)
        for column_name in columns_name:
            self.dataframe[column_name] = self.dataframe[column_name].apply(self.convert_date_format, date_format=date_format)
            self.dataframe[column_name + "_day"] = self.dataframe[column_name].apply(self._date_extractor_function, option="day")
            self.dataframe[column_name + "_month"] = self.dataframe[column_name].apply(self._date_extractor_function, option="month")
            self.dataframe[column_name + "_year"] = self.dataframe[column_name].apply(self._date_extractor_function, option="year")
            self.dataframe[column_name + "_weekday"] = self.dataframe[column_name].apply(self._date_extractor_function, option="weekday")
            self.dataframe[column_name + "_season"] = self.dataframe[column_name].apply(self._get_season)
            self.dataframe[column_name + "_bank_holiday"] = self.dataframe[column_name].apply(self._is_bank_holiday)
            self.dataframe[column_name + "_weekend"] = self.dataframe[column_name].apply(self._is_weekend)
            self.dataframe[column_name + "_week_of_year"] = self.dataframe[column_name].apply(self._get_week_of_year)
            self.dataframe[column_name + "_quarter_of_year"] = self.dataframe[column_name].apply(self._get_quarter_of_year)
            self.dataframe[column_name + "_leap_year"] = self.dataframe[column_name].apply(self._is_leap_year)
            self.dataframe[column_name + "_elapsed_day"] = self.dataframe[column_name].apply(self._get_elapsed_day)
            self.dataframe.drop(name, axis=1, inplace=True)
            print("Feature Engineering on date - DONE")  
            
        return self.dataframe
    