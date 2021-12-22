#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
# import sys
# dir_cleaners = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","tts","flowtron","text")
# print(dir_cleaners)
# sys.path.append(dir_cleaners)
# from cleaners import basic_cleaners
# from cleaners import transliteration_cleaners
# from cleaners import flowtron_cleaners
# from cleaners import english_cleaners


class Var:
    '''
    Class used to store global variable
    '''
    
    SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    NB_LIMIT_FILE_CLUSTER = 10000
    DATE_FORMAT = '%Y-%m-%d'
    TIME_FORMAT = "%H:%M:%S.%f"
    MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = 20
    MAX_NUMBER_OF_FEATURE_IMPORTANCE = 20
    LABEL_OBJECTIVE_REGRESSION = ['reg:squarederror','reg:logistic']
    LABEL_OBJECTIVE_CLASSIFICATION = ['multi:softprob','binary:logistic']
    METRIC_REGRESSION = ['rmse']
    METRIC_CLASSIFICATION = ['logloss','mlogloss']
    PKMN_TYPE_COLORS = ['#78C850',  # Grass
                        '#F08030',  # Fire
                        '#6890F0',  # Water
                        '#A8B820',  # Bug
                        '#A8A878',  # Normal
                        '#A040A0',  # Poison
                        '#F8D030',  # Electric
                        '#E0C068',  # Ground
                        '#EE99AC',  # Fairy
                        '#C03028',  # Fighting
                        '#F85888',  # Psychic
                        '#B8A038',  # Rock
                        '#705898',  # Ghost
                        '#98D8D8',  # Ice
                        '#7038F8',  # Dragon
                       ]
    SWITCHER_EXTENSION_SEPARATOR = {
        "csv": ",",
        "tsv": "\t"
    }
    
    SWITCHER_EXTENSION_FILETYPE = {
        "csv": "sv",
        "tsv": "sv",
        "xlsx": "excel",
        "txt": "text",
        "vtt": "text",
        "py": "python"
    }

    SWITCHER_SECOND_TIME_CONVERSION = {
        "millisecond": 1000,
        "microsecond": 100000,
        "nanosecond": 1000000000
    }
    
    END_CHARS = [".","?","!"]
    
    # TEXT_NORMALIZATION_CLEANERS_FUNCTION_MAPPINGS = {
    #     'basic_cleaners': basic_cleaners,
    #     'transliteration_cleaners': transliteration_cleaners,
    #     'flowtron_cleaners': flowtron_cleaners,
    #     'english_cleaners': english_cleaners
    # }
