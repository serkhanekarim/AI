#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocessing.preprocessing import preprocess_whole_pipeline

from flask import Flask, request


parser = argparse.ArgumentParser()
parser.add_argument("-path_model", help="Path of an existing model to use for prediction", required=False, default=os.path.join(directory_of_model,MODEL_NAME), nargs='?')
parser.add_argument("-path_of_data_to_predict", help="Path of data file to make prediction", required=False)
parser.add_argument("-path_of_output_prediction", help="Path of output file containing the prediction of data to predict - Use print for a print of results", required=False, default=os.path.join(directory_of_results,"submission.csv"), nargs='?')
args = parser.parse_args()

FEATURES_TO_NLP = ['Instrument(s) joué(s)','Niveau dans ces instruments','Styles musicaux','Villes','Vos disponibilités:','Groupe']
FEATURES_TO_EMBED = 
FEATURES_TO_DUMMY = 


model = None
app = Flask(__name__)

def load_model(path_model):
    global model
    model = xgb.Booster({'nthread': 8})  # init model
    model.load_model(path_model)  # load data
#    return model

@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def get_prediction(ID, 
                   label, 
                   output_path):
    
    if request.method == 'POST':
        df = request.get_json()        
        if mode == "prediction" : NaN_imputation_feature_scaling_PCA_usage = "True" in path_model        
        df, df_test, df_label = preprocess_whole_pipeline(df,
                                                        features_to_NLP=FEATURES_TO_NLP,
                                                        features_to_embed=FEATURES_TO_EMBED,
                                                        features_to_dummy=FEATURES_TO_DUMMY,
                                                        NaN_imputation_feature_scaling_PCA_boolean=NaN_imputation_feature_scaling_PCA_boolean,
                                                        column_to_remove,
                                                        label)                
        df = xgb.DMatrix(df)
        ypred = [max(0,round(value,2)) for value in list(model.predict(df))]        
        res = pd.DataFrame({"drug_id":ID, label:ypred},columns=["drug_id",label])
        if output_path != "print":
            res.to_csv(path_or_buf=output_path, index=False, float_format='%.2f')
            print("Prediction DONE, results available here: " + output_path)
        else:
            print(res)
    
if __name__ == "__main__":
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
     