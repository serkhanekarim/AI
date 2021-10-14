#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from modules.Global import variable

class TimePreprocessor:
    '''
    Class used to convert time data
    '''
        
    def convert_time_format(self, time, time_format, unit="milliseconds"):
        '''
        Method used to convert a time from a time format into a required unit

        Parameters
        ----------
        time : string
            time
        time_format : string
            Format of a date
        unit : string

        Returns
        -------
        float
            Converted time into required unit

        '''
        
        switcher = Var().SWITCHER_SECOND_TIME_CONVERSION        
        factor_conversion = switcher.get(unit)
        
        time_0 = datetime(1900, 1, 1)
        time = datetime.strptime(time, time_format)
        time = (time - begining_date) * factor_conversion
        time = time.total_seconds()
        
        return time
        