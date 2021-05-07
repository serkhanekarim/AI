#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from modules.feature_engineering.date import DatePreprocessor
from modules.scraping.scraper import DataScraper
from modules.reader.reader import DataReader
from modules.visualization.visualization import DataVisualizator



def create_model(data_train,
                 label,
                 max_depth,
                 eta,
                 num_round,
                 path_model,
                 NaN_imputation_feature_scaling_PCA_boolean,
                 directory_of_script):
    '''
    Creation of the model using XGBoost
    '''
    print("Training model using: XGBoost")
    df_train, df_test = train_test_split(data_train, test_size=0.2)    
    dtrain = xgb.DMatrix(df_train.drop(label,axis=1), label=df_train[label])
    dtest = xgb.DMatrix(df_test.drop(label,axis=1), label=df_test[label])
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    evals_result = {}    
    param = {'max_depth': max_depth, 'eta': eta, 'objective': 'reg:squarederror'}
    param['nthread'] = 8
    param['eval_metric'] = 'rmse'
    
    bst = xgb.train(param, 
                    dtrain, 
                    num_round, 
                    evallist, 
                    early_stopping_rounds=10, 
                    evals_result=evals_result)
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    param_string = 'max_depth_' + str(param['max_depth']) + "_eta_" + str(param['eta']) + "_num_round_" + str(num_round) + "_NaN_imputation_feature_scaling_PCA_usage_" + str(NaN_imputation_feature_scaling_PCA_boolean)
    model_name = param_string + "_" + dt_string
    bst.save_model(path_model + "_" + model_name)
    print("Model is available here: " + path_model + "_" + model_name)
    
    '''
    Get the XGBoost model results and information
    '''       
    print("Plotting validation curve")
    x_axis = range(len(evals_result['train']['rmse']))    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_axis, evals_result['train']['rmse'], label='Train')
    ax.plot(x_axis, evals_result['eval']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Number of Rounds')
    plt.title('XGBoost RMSE')
    plt.savefig(os.path.join(directory_of_script,"results","Validation Curve" + "_" + model_name + ".png"))
    print("Learning Curve is available here: " + os.path.join(directory_of_script,"results","Validation Curve" + "_" + model_name + ".png"))       
    
    ypred = bst.predict(dtest)    
    RMSE = mean_squared_error(df_test[label], ypred, squared=False)
    print("RMSE: %.4f" % RMSE)
            
    print("Check importance of features\n")
    fig, ax = plt.subplots(figsize=(100, 100))
    ax = xgb.plot_importance(bst,ax=ax)
    ax.figure.savefig(os.path.join(directory_of_script,"results","Feature Importance" + "_" + model_name + ".png"))
    print("Features Importance is available here: " + os.path.join(directory_of_script,"results","Feature Importance" + "_" + model_name + ".png"))
    print("Training DONE")
    
    
    
    
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
    LABEL = "price"
    
    path_train = os.path.join(data_directory,FILENAME_TRAIN)
    

    # Read data from the file
    df_train = DataReader(path_train).read_data_file()
    
    # Data Visualization
    data_visualizator = DataVisualizator(df_train)
    # Print information on variables
    data_visualizator.data_inforamtion()
    data_visualizator.missing_value_plotting(length=100,width=100)
    data_visualizator.correlation_matrix(length=100,width=100)
    data_visualizator.pie_chart(columns_name=df_train.columns)
    data_visualizator.bar_chart(columns_name=df_train.columns)
    
    # Add external data (gold financial data)
    df_train=DataScraper(df_train).add_scraped_data("date")
    # Feature Engineering on date
    df_train = DatePreprocessor(df_train).extract_date_information("date")
    # Remove two first column which reffers to index, non useful to create model
    df_train.drop(columns=['Unnamed: 0', 'index'], inplace=True)
    #Get 
    df_train = pd.get_dummies(df_train,prefix_sep="_")

        
    # Data Visualization
    data_visualizator = DataVisualizator(df_train)
    data_visualizator.correlation_matrix(length=100,width=100)

    customers, customers_normalized = Preprocessor.preprocess_RFM(data=df_train,
                                                                    customer_ID_column='CustomerID',
                                                                    date_column="VENTE_DATE",
                                                                    invoice_ID_column='InvoiceNo',
                                                                    totalprice_column='VENTE_MONTANT',
                                                                    log_transformation=False,
                                                                    root_transformation=False,
                                                                    box_cox_transformation=True,
                                                                    cubic_root_transformation=True,
                                                                    bool_apply_transformation=True,
                                                                    bool_convert_date=False)
    
    sse = {}
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(customers_normalized)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroidplt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()
    
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(customers_normalized)
    model.labels_.shape
    
    customers["Cluster"] = model.labels_
    customers.groupby('Cluster').agg({'Recency':'mean',
                                    'Frequency':'mean',
                                    'MonetaryValue':['mean', 'count']}).round(2)
    
    # Create the dataframe
    df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
    df_normalized['ID'] = customers.index
    df_normalized['Cluster'] = model.labels_
    
    # Melt The Data
    df_nor_melt = pd.melt(df_normalized.reset_index(),
                          id_vars=['ID', 'Cluster'],
                          value_vars=['Recency','Frequency','MonetaryValue'],
                          var_name='Attribute',
                          value_name='Value')
    df_nor_melt.head()
    
    # Visualize it
    sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)

  
    '''
    Preprocessing variables from train and test data, feature engineering 
    and active ingredients data
    '''
    
    if mode == "prediction" : NaN_imputation_feature_scaling_PCA_usage = "True" in path_model
    
    df_train, df_test, df_label = preprocess_whole_pipeline(df_train,
                                                            features_to_NLP,
                                                            feature_to_embed,
                                                            feature_to_dummy,
                                                            NaN_imputation_feature_scaling_PCA_boolean,
                                                            column_to_remove,
                                                            label)
    
    '''
    Creation of the model using XGBoost
    '''
    create_model(data_train=df_train,
                 label=LABEL,
                 max_depth=max_depth,
                 eta=eta,
                 num_round=num_round,
                 path_model=path_model,
                 NaN_imputation_feature_scaling_PCA_boolean=NaN_imputation_feature_scaling_PCA_usage,
                 directory_of_script=directory_of_script)
    
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