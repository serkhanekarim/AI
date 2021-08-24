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


import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer        
from sklearn.feature_extraction.text import TfidfTransformer

    
def main(args):
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    data_directory = args.data_directory
    
    #data_directory = "/home/serkhane/repo/test-cdiscount/data/"    
    FILENAME_TRAIN = "train.csv"
    FILENAME_TEST = "test.csv"
    FILENAME_PREDICTION = 'predictions.csv'
    LABEL = "category_id"    
    path_train = os.path.join(data_directory,FILENAME_TRAIN)
    path_test = os.path.join(data_directory,FILENAME_TEST)
    path_prediction = os.path.join(directory_of_script,'results','experiments',FILENAME_PREDICTION)
    

    # Read data from the file
    data = DataReader(path_train).read_data_file()
    data_test = DataReader(path_test).read_data_file()
    
    column_to_vectorize = ['description', 'title']
    
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('french'))
    tfidfconverter = TfidfTransformer()
    
    for column in column_to_vectorize:
    
        print("Train Vectorization of : " + column)
        
        X = data[column]
        
        documents = []
        
        stemmer = WordNetLemmatizer()
    
        for sen in range(0, len(X)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))
            
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            
            # Converting to Lowercase
            document = document.lower()
            
            # Lemmatization
            document = document.split()
        
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            
            documents.append(document)

        X = vectorizer.fit_transform(documents).toarray()
        X = tfidfconverter.fit_transform(X).toarray()
        
        data = pd.concat([data,pd.DataFrame(X).add_suffix('_'+column)], axis=1)
    
    category = data['category']
    data.drop(columns=['category','description','title'], inplace=True)
    
    #Split data into train and test set to evaluate the model
    df_train, df_test, df_train_label, df_test_label = train_test_split(data.drop(LABEL,axis=1), data[LABEL], test_size=0.2)


    bst = Modeler(df_train, LABEL).XGBoost_model(df_train=df_train.values,
                                                   df_train_label=df_train_label.values,
                                                   df_test=df_test.values,
                                                   df_test_label=df_test_label.values,
                                                   num_class=len(set(df_train_label)),
                                                   num_round=75,
                                                   max_depth=2,
                                                   eta=0.3)

    plt.show(block=True)      
    
    for column in column_to_vectorize:
        
        print("Test Vectorization of : " + column)
        
        X_test = data_test[column]
        
        documents = []
            
        stemmer = WordNetLemmatizer()
    
        for sen in range(0, len(X_test)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X_test[sen]))
            
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            
            # Converting to Lowercase
            document = document.lower()
            
            # Lemmatization
            document = document.split()
        
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            
            documents.append(document)
        
        X_test = vectorizer.fit_transform(documents).toarray()
        
        X_test = tfidfconverter.fit_transform(X_test).toarray()
        
        data_test = pd.concat([data_test,pd.DataFrame(X_test).add_suffix('_'+column)], axis=1)
     
    ID = data_test['id']
    data_test.drop(columns=['description','title','id'], inplace=True)
   
    prediction = bst.predict(xgb.DMatrix(data_test.values))
    prediction = [element.argmax() for element in prediction]
    
    prediction_table = pd.DataFrame({'id':ID, 'predicted_category_id':prediction},columns=['id', 'predicted_category_id'])
    
    prediction_table.to_csv(path_prediction, index=False)
    
    print("Prediction is available here: " + path_prediction)
    


if __name__ == "__main__":
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_data = os.path.join(directory_of_script,"data")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    args = parser.parse_args()
    
    main(args)    