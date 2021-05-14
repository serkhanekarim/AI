#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from modules.feature_engineering.date import DatePreprocessor
from modules.feature_engineering.encoder import DataEncoder
from modules.scraping.scraper import DataScraper
from modules.reader.reader import DataReader
from modules.visualization.visualization import DataVisualizator
from modules.modeling.machine_learning import Modeler


    
def main(args):
    
    #data_directory = "/home/serkhane/repo/test-quantmetry/data/"
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    data_directory = args.data_directory
    path_model = args.path_model
    mode = args.mode
    NaN_imputation_feature_scaling_PCA_usage = args.NaN_imputation_feature_scaling_PCA_usage
    max_depth = args.max_depth
    eta = args.eta
    num_round = args.num_round   
    
    FILENAME_TRAIN = "data_v1.0 (3) (1) (1) (1) (2).csv"    
    FEATURES_TO_NLP = ['Instrument(s) joué(s)','Niveau dans ces instruments','Styles musicaux','Villes','Vos disponibilités:','Groupe']
    LABEL = "embauche"
    
    path_train = os.path.join(data_directory,FILENAME_TRAIN)
    

    # Read data from the file
    df_train = DataReader(path_train).read_data_file()
    
    # Data Visualization
    data_visualizator = DataVisualizator(df_train)
    # Print information on variables
    data_visualizator.data_inforamtion()    
    
    # Remove two first column which reffers to index, non useful to create model
    df_train.drop(columns=['Unnamed: 0', 'index'], inplace=True)
    
    # Data Visualization
    data_visualizator = DataVisualizator(df_train)    
    data_visualizator.missing_value_plotting()
    data_visualizator.correlation_matrix()
    data_visualizator.pie_chart(columns_name=df_train.columns)
    data_visualizator.bar_chart(columns_name=df_train.columns, label=LABEL)
    data_visualizator.histogram(columns_name=df_train.columns)
    data_visualizator.pair_plot(label=LABEL)
    data_visualizator.box_plot(columns_name=df_train.columns,label=LABEL)
    data_visualizator.violin_plot(columns_name=df_train.columns,label=LABEL)
    
    # Add external data (gold financial data)
    df_train = DataScraper(df_train).add_scraped_data("date")
    # Feature Engineering on date
    df_train = DatePreprocessor(df_train).extract_date_information(columns_name="date")
    # Convert categorical variable into dummy/indicator variables 
    df_train = DataEncoder(df_train).dummy_variable()
    
    #/home/serkhane/repo/test-quantmetry/results
    Modeler(df_train).create_XGB_model(label=LABEL,
                                       model_path="/home/serkhane/repo/test-quantmetry/results/test.model",
                                       num_round=5000,
                                       threshold_class=1/2,
                                       max_depth=3,
                                       eta=0.005)
    
if __name__ == "__main__":
    
    MODEL_NAME = "XGBoost_marketing.model"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_model = os.path.join(directory_of_script,"model")
    directory_of_results = os.path.join(directory_of_script,"results")
    directory_of_data = os.path.join(directory_of_script,"data")
    os.makedirs(directory_of_model,exist_ok=True)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    parser.add_argument("-path_model", help="Path of an existing model to use for prediction", required=False, default=os.path.join(directory_of_model,MODEL_NAME), nargs='?')
    parser.add_argument("-NaN_imputation_feature_scaling_PCA_usage", help="Apply or not NaN imputation, Feature Scaling and PCA", required=False, choices=["False","True"], default="False", nargs='?')
    parser.add_argument("-max_depth", help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist)", required=False, default=6, type=int, nargs='?')
    parser.add_argument("-eta", help="Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. range: [0,1]", required=False, default=0.3, type=float, nargs='?')
    parser.add_argument("-num_round", help="The number of rounds for boosting", required=False, default=100, type=int, nargs='?')
    args = parser.parse_args()
    
    main(args)    