#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

class Var:
    '''
    Class used to store global variable
    '''
    
    SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    DATE_FORMAT = '%Y-%m-%d'
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
