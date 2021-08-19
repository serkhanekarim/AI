#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from datetime import datetime

# import numpy as np

# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import ShuffleSplit
# from sklearn.metrics import confusion_matrix
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from scipy.optimize import curve_fit
# from scipy.interpolate import lagrange

# from modules.Global.variable import Var

# from modules.visualization.visualization import DataVisualizator

import tensorflow as tf


class basic_CNN:
    '''
    Pour le modèle, vous utiliserez un simple réseau de neurones convolutifs (CNN), puisque 
    vous avez transformé les fichiers audio en images de spectrogramme. Le modèle comporte 
    également les couches de prétraitement supplémentaires suivantes :
        
        Une couche de Resizing pour sous-échantillonner l'entrée afin de permettre au modèle de 
        s'entraîner plus rapidement.
        Une Normalization couche de normaliser chaque pixel de l'image en fonction de son écart 
        moyen et standard.
        
    Pour la couche de Normalization , sa méthode d' adapt devrait d'abord être appelée sur les données 
    d'apprentissage afin de calculer les statistiques agrégées (c'est-à-dire la moyenne et l'écart type).
    '''


# class Modeler:
#     '''
#     Class used to create Machine Learning with data
#     '''
    
#     SCRIPT_DIRECTORY = Var().SCRIPT_DIRECTORY
#     MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = Var().MAX_NUMBER_OF_CATEGORICAL_OCCURENCES
#     LABEL_OBJECTIVE_REGRESSION = Var().LABEL_OBJECTIVE_REGRESSION
#     LABEL_OBJECTIVE_CLASSIFICATION = Var().LABEL_OBJECTIVE_CLASSIFICATION
#     METRIC_REGRESSION = Var().METRIC_REGRESSION
#     METRIC_CLASSIFICATION = Var().METRIC_CLASSIFICATION 
    
#     def __init__(self, dataframe, label):
#         self.dataframe = dataframe
#         self.label = label
        
#     def _detect_objective(self, data_label):
#         '''
#         Detect which objective to choose for XGBoost model regarding values in label

#         Parameters
#         ----------
#         data_label : dataframe
            

#         Returns
#         -------
#         str
#             Name of the objective to use for the XGBoost model

#         '''
        
#         length_unique_value = len(set(data_label))
#         minimum = min(data_label)
#         maximum = max(data_label)
        
#         if 3 <= length_unique_value <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
#             # Multi classification
#             print(self.LABEL_OBJECTIVE_CLASSIFICATION[0])
#             return (self.LABEL_OBJECTIVE_CLASSIFICATION[0],self.METRIC_CLASSIFICATION[1])
#         if length_unique_value <= 2:
#             # Binary classification
#             print(self.LABEL_OBJECTIVE_CLASSIFICATION[1])
#             return (self.LABEL_OBJECTIVE_CLASSIFICATION[1],self.METRIC_CLASSIFICATION[0])
#         if 3 <= length_unique_value and 0 <= minimum and maximum <= 1:
#             # Logistic regression
#             print(self.LABEL_OBJECTIVE_REGRESSION[1])
#             return (self.LABEL_OBJECTIVE_REGRESSION[1],self.METRIC_REGRESSION[0])
#         if self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES < length_unique_value:
#             # Regression
#             print(self.LABEL_OBJECTIVE_REGRESSION[0])
#             return (self.LABEL_OBJECTIVE_REGRESSION[0],self.METRIC_REGRESSION[0])
        
#     def linear_regressor(self, x):
#         '''
#         Create linear regression

#         Parameters
#         ----------
#         x : string
#             column name of x
#         y : string
#             column name of y

#         Returns
#         -------
#         str, plot
#             Display the linear regression function

#         '''
#         X = self.dataframe[x].to_numpy().reshape(-1, 1)
#         y = self.dataframe[self.label].to_numpy()
#         reg = LinearRegression().fit(X, y)
#         reg.score(X, y)
#         print(reg.coef_)
#         a = reg.coef_[0]
#         b = reg.coef_[1]
#         print("Linear regression: y = " + str(a) + "*X+" + str(b))

#     def Ensemble_model(self,
#                         df_train,
#                         df_train_label,
#                         df_test,
#                         df_test_label,
#                         model_path,
#                         n_estimators=100,
#                         max_depth=6,
#                         objective=None,
#                         threshold_class=None,
#                         method='RandomForest'):
#         '''
#         Create Random Forest model

#         Parameters
#         ----------
#         df_train : dataframe
#             dataframe containing training data
#         df_train_label : dataframe
#             dataframe containing ttraining label data
#         df_test : dataframe
#             dataframe containing test data
#         df_test_label : dataframe
#             dataframe containing test label data
#         path_model : string
#             string containing the path of the model which will be saved.
#         n_estimators int, default=100
#             The number of trees in the forest.
#         max_depth : int, optional
#             Maximum depth of a tree. Increasing this value will make the model 
#             more complex and more likely to overfit. 0 is only accepted in 
#             lossguided growing policy when tree_method is set as hist or gpu_hist 
#             and it indicates no limit on depth. Beware that XGBoost aggressively 
#             consumes memory when training a deep tree.
#             range: [0,∞] (0 is only accepted in lossguided growing policy when 
#                           tree_method is set as hist or gpu_hist). The default is 6.

#         Returns
#         -------
#         Save model and related plot and display them

#         '''
        
#         print("Training model using: Random Forest")
#         detected_objective = self._detect_objective(df_train_label)
#         eval_metric = detected_objective[1]
#         objective = objective or detected_objective[0]
#         # X = df_train.append(df_test,ignore_index=True).to_numpy()
#         # y = df_train_label.append(df_test_label,ignore_index=True).to_numpy()
#         X_train = df_train.to_numpy()
#         # X_test = df_test.to_numpy()
#         y_train = df_train_label.to_numpy()
#         # y_test = df_test_label.to_numpy()
#         if method == 'RandomForest':
#             param = {'n_estimators': n_estimators,
#                      'max_depth': max_depth, 
#                      'n_jobs' : -1,
#                      'verbose' : 1}
#             if objective in set(self.LABEL_OBJECTIVE_REGRESSION):
#                 estimator = RandomForestRegressor()
#             if objective in set(self.LABEL_OBJECTIVE_CLASSIFICATION):
#                 estimator = RandomForestClassifier()
#         estimator.set_params(**param)
#         estimator.fit(X_train, y_train)
#         #scores = cross_val_score(estimator, X, y, cv=5)
        
#         now = datetime.now()
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         param_string = "_".join([key +"_" + str(param[key]) for key in param.keys()])
#         model_name = param_string + "_" + dt_string
#         #bst.save_model(model_path + "_" + model_name)
#         print("Model is available here: " + model_path + "_" + model_name)
        
        
#         '''
#         Get the RandomForest model results and information
#         '''
        
#         # Cross validation with 100 iterations to get smoother mean test and train
#         # score curves, each time with 20% data randomly selected as a validation set.
        
#         # DataVisualizator.plot_learning_curve(estimator,
#         #                     title="Learning Curves: " + method,
#         #                     X=X,
#         #                     y=y,
#         #                     cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
#         #                     n_jobs=-1)
               
#         if eval_metric == self.METRIC_REGRESSION[0]:
#             ypred = estimator.predict(df_test)
#             RMSE = mean_squared_error(df_test_label, ypred, squared=False)
#             print(eval_metric.upper() + ": %.4f" % RMSE)
#         if eval_metric == self.METRIC_CLASSIFICATION[0]:
#             ypred = estimator.predict_proba(df_test)[:,1]
#             ypred_rounded = ypred.round()
#             if len(set(df_train_label)) == 2:
#                 ypred_rounded = np.where(ypred >= threshold_class,1,0)            
#             cf_matrix = confusion_matrix(df_test_label,ypred_rounded)
#             DataVisualizator.confusion_matrix(cf_matrix)
#             DataVisualizator.roc_auc(np.array(df_test_label), ypred)
            
#         DataVisualizator(self.dataframe).features_importance(estimator=estimator, estimator_type='scikit-learn')
        
#         print("Training DONE")        
    
#     def XGBoost_model(self,
#                          df_train,
#                          df_train_label,
#                          df_test,
#                          df_test_label,
#                          num_round,
#                          max_depth=6,
#                          eta=0.3,
#                          num_class=None,
#                          objective=None,
#                          threshold_class=None,
#                          model_path=None):
#         '''
#         Create XGBoost model

#         Parameters
#         ----------
#         df_train : dataframe
#             dataframe containing training data
#         df_train_label : dataframe
#             dataframe containing ttraining label data
#         df_test : dataframe
#             dataframe containing test data
#         df_test_label : dataframe
#             dataframe containing test label data
#         path_model : string
#             string containing the path of the model which will be saved.
#         num_round : int
#             The number of rounds for boosting.
#         max_depth : int, optional
#             Maximum depth of a tree. Increasing this value will make the model 
#             more complex and more likely to overfit. 0 is only accepted in 
#             lossguided growing policy when tree_method is set as hist or gpu_hist 
#             and it indicates no limit on depth. Beware that XGBoost aggressively 
#             consumes memory when training a deep tree.
#             range: [0,∞] (0 is only accepted in lossguided growing policy when 
#                           tree_method is set as hist or gpu_hist). The default is 6.
#         eta : float
#             Step size shrinkage used in update to prevents overfitting. After each 
#             boosting step, we can directly get the weights of new features, and eta 
#             shrinks the feature weights to make the boosting process more 
#             conservative. The default is 0.3.
#             range: [0,1]

#         Returns
#         -------
#         Save model and related plot and display them

#         '''
        
#         print("Training model using: XGBoost")
#         detected_objective = self._detect_objective(df_train_label)
#         eval_metric = detected_objective[1]
#         objective = objective or detected_objective[0]
#         dtrain = xgb.DMatrix(df_train, label=df_train_label)
#         dtest = xgb.DMatrix(df_test, label=df_test_label)
        
#         evallist = [(dtest, 'eval'), (dtrain, 'train')]
#         evals_result = {}    
#         param = {'max_depth': max_depth, 
#                  'eta': eta, 
#                  'objective': objective, 
#                  'nthread' : -1,
#                  'num_class' : num_class}
        
#         bst = xgb.train(param, 
#                         dtrain, 
#                         num_round, 
#                         evallist, 
#                         early_stopping_rounds=10, 
#                         evals_result=evals_result)
        
#         now = datetime.now()
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         param_string = "_".join([key +"_" + str(param[key]) for key in param.keys()])
#         model_name = param_string + "_" + dt_string
#         #bst.save_model(model_path + "_" + model_name)
#         #print("Model is available here: " + model_path + "_" + model_name)
        
        
#         '''
#         Get the XGBoost model results and information
#         '''
#         x_axis = range(len(evals_result['train'][eval_metric]))
#         y_axis = [evals_result['train'][eval_metric], 
#                   evals_result['eval'][eval_metric]]
        
#         DataVisualizator.curve(x=x_axis, 
#                                y=y_axis,
#                                xlabel='Number of Rounds',
#                                ylabel=eval_metric.upper(),
#                                title="XGBoost - " + eval_metric.upper(),
#                                label=["train","eval"])
        
#         ypred = bst.predict(dtest)        
#         if eval_metric == self.METRIC_REGRESSION[0]:
#             RMSE = mean_squared_error(df_test_label, ypred, squared=False)
#             print(eval_metric.upper() + ": %.4f" % RMSE)
#         if eval_metric == self.METRIC_CLASSIFICATION[0]:
#             ypred_rounded = ypred.round()
#             if len(set(df_train_label)) == 2:
#                 ypred_rounded = np.where(ypred >= threshold_class,1,0)            
#             cf_matrix = confusion_matrix(df_test_label,ypred_rounded)
#             DataVisualizator.confusion_matrix(cf_matrix)
#             DataVisualizator.roc_auc(np.array(df_test_label), ypred)
            
#         DataVisualizator(self.dataframe).features_importance(estimator=bst, estimator_type='xgboost')
#         print("Training DONE")
#         return bst
        
        
#     def hyperParameterTuningXGB(self, X_train, y_train, objective):
#         '''
#         Grid search best parameters to fit the model

#         Parameters
#         ----------
#         X_train : xgb.DMatrix
#             Array containing training data.
#         y_train : xgb.DMatrix
#             Array containing test data.

#         Returns
#         -------
#         dict
#             Best paramaters.

#         '''
        
#         param_tuning = {
#             'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
#             'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
#             'min_child_weight': [1, 3, 5, 7],
#             'subsample': [0.5, 0.7, 1],
#             'colsample_bytree': [0.1, 0.3, 0.4, 0.5 , 0.7, 1],
#             'n_estimators' : [100, 200, 500],
#             'gamma' : [0.0, 0.1, 0.2 , 0.3, 0.4],
#             'objective': [objective]
#         }
    
#         xgb_model = xgb.XGBRegressor()
    
#         gsearch = GridSearchCV(estimator = xgb_model,
#                                param_grid = param_tuning,                        
#                                #scoring = 'neg_mean_absolute_error', #MAE
#                                #scoring = 'neg_mean_squared_error',  #MSE
#                                cv = 5,
#                                n_jobs = -1,
#                                verbose = 1)
    
#         gsearch.fit(X_train,y_train)
    
#         return gsearch.best_params_