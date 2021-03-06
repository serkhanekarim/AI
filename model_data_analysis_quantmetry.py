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
from modules.Global.method import DataMethod


    
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
    data = DataReader(path_train).read_data_file()

    # Print information on variables
    DataVisualizator(data).data_inforamtion()    
    
    # Remove two first column which reffers to index, non useful to create model
    data.drop(columns=['Unnamed: 0', 'index'], inplace=True)
    
    
    '''
    Feature Engineering/Extraction/Vectorization
    '''
    # Add external data (gold financial data)
    data = DataScraper(data).add_scraped_data("date")
    # Feature Engineering on date
    data = DatePreprocessor(data).extract_date_information(columns_name="date")
    # Convert categorical variable into dummy/indicator variables 
    data = DataEncoder(data).dummy_variable()
    
    # Print missing value
    DataVisualizator(data).missing_value_plotting()
    
    #Split data into train and test set to evaluate the model
    df_train, df_test, df_train_label, df_test_label = train_test_split(data.drop(LABEL,axis=1), data[LABEL], test_size=0.2)
    
    # Missing data imputation using K-NN
    imputer = DataImputation(df_train).imputer()
    df_train = imputer[0]
    # Missing data imputation on test data using training data imputer model
    df_test = imputer[1].transform(df_test)
    df_test = pd.DataFrame(df_test, columns = df_train.columns)
    
    # Feature Selection before applying scaling data
    DataVisualizator(df_train).correlation_matrix() # Correlation Matrix
    selector = DataSelector(df_train, df_train_label).features_selection() # Trees
    DataVisualizator(df_train).features_importance(estimator=selector[0],
                                                   estimator_type='tree',
                                                   max_num_features=20)
    
    
    
    # Feature generation using most important features
    number_of_generated_feature = 7
    selector_train = DataSelector(df_train, df_train_label).features_selection(number_of_generated_feature)
    generator_train = DataGenerator(df_train,selector_train[1]).polynomial_features(3)    
    df_train = generator_train[0]
    
    # Feature generation on test data using training data generator model
    selector_test = df_test[selector_train[1].columns]
    generator_test = DataGenerator(df_test,selector_test).polynomial_features(3)
    df_test = generator_test[0]
    
    # Correlation Matrix
    DataVisualizator(df_train).correlation_matrix()
    
    # Pie Chart
    DataVisualizator(df_train).pie_chart(columns_name=df_train.columns)
    # Bar Chart
    DataVisualizator(df_train).bar_chart(columns_name=df_train.columns, label=LABEL)
    # Histogram
    DataVisualizator(df_train).histogram(columns_name=df_train.columns)
    # Pair plot, plotting feature vs label
    DataVisualizator(df_train).pair_plot(label=LABEL)
    # Box plot
    DataVisualizator(df_train).box_plot(columns_name=df_train.columns,label=LABEL)
    # Violin Plot
    DataVisualizator(df_train).violin_plot(columns_name=df_train.columns,label=LABEL)
    
    
    # Scaling data
    scaler = DataScaler(df_train).scaler(method='yeo-johnson')
    df_train = pd.DataFrame(scaler[0])
    # Scaling data on test data using training data scaler model
    df_test = pd.DataFrame(scaler[1].transform(df_test))
    
    #/home/serkhane/repo/test-quantmetry/results
    Modeler(df_train, LABEL).XGBoost_model(df_train=df_train,
                                       df_train_label=df_train_label,
                                       df_test=df_test,
                                       df_test_label=df_test_label,
                                       model_path="/home/serkhane/repo/test-quantmetry/results/test.model",
                                       num_round=250,
                                       threshold_class=0.3,
                                       max_depth=2,
                                       eta=0.1)
    
    Modeler(df_train, LABEL).Ensemble_model(df_train=df_train,
                                           df_train_label=df_train_label,
                                           df_test=df_test,
                                           df_test_label=df_test_label,
                                           model_path="/home/serkhane/repo/test-quantmetry/results/test.model",
                                           n_estimators=10000,
                                           max_depth=None,
                                           threshold_class=0.25,
                                           method='RandomForest')


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