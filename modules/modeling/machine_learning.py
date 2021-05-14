#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from modules.Global import variable

from modules.visualization.visualization import DataVisualizator



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
                     objective=None,
                     threshold_class=None):
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
        Save model and related plot and display them

        '''
        
        print("Training model using: XGBoost")
        detected_objective = self._detect_objective(label)
        eval_metric = detected_objective[1]
        objective = objective or detected_objective[0]
        df_train, df_test = train_test_split(self.dataframe, test_size=0.2)
        df_train_label = df_train[label]
        df_train = df_train.drop(label,axis=1)
        df_test_label = df_test[label]
        df_test = df_test.drop(label,axis=1)
        dtrain = xgb.DMatrix(df_train, label=df_train_label)
        dtest = xgb.DMatrix(df_test, label=df_test_label)
        
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        evals_result = {}    
        param = {'max_depth': max_depth, 
                 'eta': eta, 
                 'objective': objective, 
                 'nthread' : -1}
        
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
        x_axis = range(len(evals_result['train'][eval_metric]))
        y_axis = [evals_result['train'][eval_metric], 
                  evals_result['eval'][eval_metric]]
        
        DataVisualizator.curve(x=x_axis, 
                               y=y_axis,
                               xlabel='Number of Rounds',
                               ylabel=eval_metric.upper(),
                               title="XGBoost - " + eval_metric.upper(),
                               label=["train","eval"])
        
        ypred = bst.predict(dtest)        
        if eval_metric == "rmse":
            RMSE = mean_squared_error(df_test[label], ypred, squared=False)
            print("RMSE: %.4f" % RMSE)
        if eval_metric == "logloss":
            ypred_rounded = ypred.round()
            if len(set(self.dataframe[label])) == 2:
                ypred_rounded = np.where(ypred >= threshold_class,1,0)            
            cf_matrix = confusion_matrix(df_test_label,ypred_rounded)
            DataVisualizator.confusion_matrix(cf_matrix)
            DataVisualizator.roc_auc(np.array(df_test_label), ypred)
            
        DataVisualizator.features_importance(bst)
        
        print("Training DONE")
        
        
    def hyperParameterTuningXGB(self, X_train, y_train, objective):
        '''
        Grid search best parameters to fit the model

        Parameters
        ----------
        X_train : xgb.DMatrix
            Array containing training data.
        y_train : xgb.DMatrix
            Array containing test data.

        Returns
        -------
        dict
            Best paramaters.

        '''
        
        param_tuning = {
            'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.1, 0.3, 0.4, 0.5 , 0.7, 1],
            'n_estimators' : [100, 200, 500],
            'gamma' : [0.0, 0.1, 0.2 , 0.3, 0.4],
            'objective': [objective]
        }
    
        xgb_model = xgb.XGBRegressor()
    
        gsearch = GridSearchCV(estimator = xgb_model,
                               param_grid = param_tuning,                        
                               #scoring = 'neg_mean_absolute_error', #MAE
                               #scoring = 'neg_mean_squared_error',  #MSE
                               cv = 5,
                               n_jobs = -1,
                               verbose = 1)
    
        gsearch.fit(X_train,y_train)
    
        return gsearch.best_params_