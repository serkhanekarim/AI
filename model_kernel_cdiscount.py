#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path

from math import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from modules.preprocessing.date import DatePreprocessor
from modules.preprocessing.feature_encoder import DataEncoder
from modules.scraping.scraper import DataScraper
from modules.reader.reader import DataReader
from modules.visualization.visualization import DataVisualizator
from modules.modeling.machine_learning import Modeler
from modules.preprocessing.missing_value import DataImputation
from modules.preprocessing.feature_scaling import DataScaler
from modules.preprocessing.feature_selection import DataSelector
from modules.preprocessing.feature_generator import DataGenerator
from modules.preprocessing.date import DatePreprocessor
from modules.Global.method import DataMethod

from datetime import datetime

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from scipy.interpolate import lagrange

from modules.Global.variable import Var

from modules.visualization.visualization import DataVisualizator

from datetime import datetime


def main(args):
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    data_directory = args.data_directory
    
    #data_directory = "/home/serkhane/repo/test-cdiscount/data/"    
    FILENAME_TRAIN = "purchases.csv"    
    LABEL = "amount"    
    path_train = os.path.join(data_directory,FILENAME_TRAIN)
    
    # Read data from the file
    data = DataReader(path_train).read_data_file()

    day_0 = datetime.strptime(data['time'][0], '%Y-%m-%d %H:%M:%S')
    X_1 = [(datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - day_0).total_seconds()/3600 for date in data['time']]
    X_2 = list(data[LABEL])
    
    data_kde = pd.DataFrame({'Elapsed Time':X_1, 'Amount':X_2},columns=['Elapsed Time','Amount'])
    
    
    print("Waiting for the Kernel computaion of data...")
    plt.figure()
    plt.title("Estimation par noyau Gaussien")
    sns.kdeplot(
        data=data_kde, x="Elapsed Time", y="Amount",
        fill=True, thresh=0, levels=100, cmap="mako",
    )
    
    plt.show()


if __name__ == "__main__":
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_data = os.path.join(directory_of_script,"data")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    args = parser.parse_args()
    
    main(args)    