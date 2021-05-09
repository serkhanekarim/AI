#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt

import os.path
from modules.Global import variable

class Modeler:
    '''
    Class used to create Machine Learning with data
    '''
    
    SCRIPT_DIRECTORY = variable.Var().SCRIPT_DIRECTORY
    MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = variable.Var().MAX_NUMBER_OF_CATEGORICAL_OCCURENCES
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def _detect_objective(self, label):
        '''
        Detect which objective to choose for XGBoost model regarding values in label

        Parameters
        ----------
        label : string
            Name of the column used as label on data

        Returns
        -------
        str
            Name of the objective to use for the XGBoost model

        '''
        
        length_unique_value = len(set(self.dataframe[label]))
        minimum = min(self.dataframe[label])
        maximum = max(self.dataframe[label])
        
        if 3 <= length_unique_value <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
            print('multi:softprob')
            return ('multi:softprob''logloss')
        if length_unique_value <= 2:
            print('binary:logistic')
            return ('binary:logistic','logloss')
        if 3 <= length_unique_value and 0 <= minimum and maximum <= 1:
            print('reg:logistic')
            return ('reg:logistic','rmse')
        if self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES < length_unique_value:
            print('reg:squarederror')
            return ('reg:squarederror','rmse')
            
    
    def create_XGB_model(self,
                     label,
                     model_path,
                     num_round,
                     max_depth=6,
                     eta=0.3,
                     objective=None):
        '''
        Create XGB model

        Parameters
        ----------
        label : string
            string containing of the data to train the model.
        path_model : string
            string containing the path of the model which will be saved.
        num_round : int
            The number of rounds for boosting.
        max_depth : int, optional
            Maximum depth of a tree. Increasing this value will make the model 
            more complex and more likely to overfit. 0 is only accepted in 
            lossguided growing policy when tree_method is set as hist or gpu_hist 
            and it indicates no limit on depth. Beware that XGBoost aggressively 
            consumes memory when training a deep tree.
            range: [0,∞] (0 is only accepted in lossguided growing policy when 
                          tree_method is set as hist or gpu_hist). The default is 6.
        eta : float
            Step size shrinkage used in update to prevents overfitting. After each 
            boosting step, we can directly get the weights of new features, and eta 
            shrinks the feature weights to make the boosting process more 
            conservative. The default is 0.3.
            range: [0,1]

        Returns
        -------
        Save model and related plot

        '''
        
        print("Training model using: XGBoost")
        detected_objective = self._detect_objective(label)
        eval_metric = detected_objective[1]
        objective = objective or detected_objective[0]
        df_train, df_test = train_test_split(self.dataframe, test_size=0.2)    
        dtrain = xgb.DMatrix(df_train.drop(label,axis=1), label=df_train[label])
        dtest = xgb.DMatrix(df_test.drop(label,axis=1), label=df_test[label])
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        evals_result = {}    
        param = {'max_depth': max_depth, 'eta': eta, 'objective': objective}
        param['nthread'] = 8
        
        bst = xgb.train(param, 
                        dtrain, 
                        num_round, 
                        evallist, 
                        early_stopping_rounds=10, 
                        evals_result=evals_result)
        
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        param_string = 'max_depth_' + str(param['max_depth']) + "_eta_" + str(param['eta']) + "_num_round_" + str(num_round)
        model_name = param_string + "_" + dt_string
        #bst.save_model(model_path + "_" + model_name)
        print("Model is available here: " + model_path + "_" + model_name)
        
        '''
        Get the XGBoost model results and information
        '''       
        print("Plotting validation curve")
        x_axis = range(len(evals_result['train'][eval_metric]))    
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x_axis, evals_result['train'][eval_metric], label='Train')
        ax.plot(x_axis, evals_result['eval'][eval_metric], label='Test')
        ax.legend()
        plt.ylabel(eval_metric.upper())
        plt.xlabel('Number of Rounds')
        plt.title("XGBoost - " + eval_metric.upper())
        #plt.savefig(os.path.join(self.SCRIPT_DIRECTORY,"results","Validation Curve" + "_" + model_name + ".png"))
        print("Learning Curve is available here: " + os.path.join(self.SCRIPT_DIRECTORY,"results","Validation Curve" + "_" + model_name + ".png"))       
        
        ypred = bst.predict(dtest)    
        RMSE = mean_squared_error(df_test[label], ypred, squared=False)
        print("RMSE: %.4f" % RMSE)
                
        print("Check importance of features\n")
        fig, ax = plt.subplots(figsize=(100, 100))
        ax = xgb.plot_importance(bst,ax=ax)
        #ax.figure.savefig(os.path.join(self.SCRIPT_DIRECTORY,"results","Feature Importance" + "_" + model_name + ".png"))
        print("Features Importance is available here: " + os.path.join(self.SCRIPT_DIRECTORY,"results","Feature Importance" + "_" + model_name + ".png"))
        print("Training DONE")