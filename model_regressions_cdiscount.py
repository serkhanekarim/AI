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


def func(x, a, b, c, d):
    return a * np.sin(b*x+c) + d 
    
def main(args):
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    data_directory = args.data_directory
    
    #data_directory = "/home/serkhane/repo/test-cdiscount/data/"    
    FILENAME_TRAIN = "regression_data.csv"    
    LABEL = " y"    
    path_train = os.path.join(data_directory,FILENAME_TRAIN)
    

    # Read data from the file
    data = DataReader(path_train).read_data_file()
    
    #Split data into train and test set to evaluate the model
    df_train, df_test, df_train_label, df_test_label = train_test_split(data.drop(LABEL,axis=1), data[LABEL], test_size=0.2)

    # Print information on variables
    DataVisualizator(data).data_inforamtion()
    
        
    X_full = data['x'].to_numpy()
    y_full = data[' y'].to_numpy()
    X_full_reshape = data['x'].to_numpy().reshape(1, -1).reshape((len(X_full),1))
    y_full_reshape = data[' y'].to_numpy().reshape(1, -1).reshape((len(y_full),1))
    X_manual = [0,18,22,30,35,40,45,55,60,65,70,80,100]
    y_manual = [y_full[X_full==i][0] for i in X_manual]
    X = np.array(X_manual).reshape(1, -1).reshape((len(X_manual),1))
    y = np.array(y_manual)              
    
    poly = PolynomialFeatures(degree = 4)
    X_poly = poly.fit_transform(X_full_reshape)          
    poly.fit(X_poly, y_full)
    lin = LinearRegression()
    lin.fit(X_poly, y_full)
    
    poly_lagrange = lagrange(X_manual, y_manual)
    print("Polynome de Lagrange :")
    print(poly_lagrange)
    
    popt, pcov = curve_fit(func, X_full, y_full)
    print("Curve Fit : a * np.sin(b*x+c) + d ")
    print(popt)
    
    reg, stats = np.polynomial.polynomial.polyfit(X_full, y_full, 1, full=True)
    model = np.poly1d(reg)
    print("Polyfit Numpy :")
    print(model)
    
    xgb_model = Modeler(data, LABEL).XGBoost_model(df_train=df_train,
                                                   df_train_label=df_train_label,
                                                   df_test=df_test,
                                                   df_test_label=df_test_label,
                                                   num_round=100,
                                                   max_depth=3,
                                                   eta=0.1)
    
    
    x_xgboost = pd.concat([df_train,df_test], axis=0).sort_index()
    y_xgboost = pd.concat([df_train_label,df_test_label], axis=0).sort_index()
    xgb_predictions = xgb_model.predict(xgb.DMatrix(pd.concat([df_train,df_test], axis=0).sort_index(), label=y_xgboost))    
    
    RMSE_poly_fit = mean_squared_error(y_full, model(y_full))
    RMSE_curve_fit_optimizer = mean_squared_error(y_full, func(y_full, *popt))
    RMSE_regression_polynomial = mean_squared_error(y_full, lin.predict(poly.fit_transform(y_full_reshape)))
    RMSE_lagrange = mean_squared_error(y_full, poly_lagrange(y_full))
    RMSE_XGBoost = mean_squared_error(y_full, xgb_model.predict(xgb.DMatrix(pd.concat([df_train,df_test], axis=0).sort_index(), label=pd.concat([df_train_label,df_test_label], axis=0).sort_index())))
        
    print("RMSE_poly_fit: " + str(RMSE_poly_fit))
    print("RMSE_curve_fit_optimizer: " + str(RMSE_curve_fit_optimizer))
    print("RMSE_regression_polynomial: " + str(RMSE_regression_polynomial))
    print("RMSE_lagrange: " + str(RMSE_lagrange))
    print("RMSE_XGBoost: " + str(RMSE_XGBoost))
    
    plt.figure()
    ylim=[min(y), max(y)]
    xlim=[min(X), max(X)]
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.scatter(X_full, y_full, color = 'blue')
    
    x_linsp = np.linspace(0,100,1000)
  
    plt.plot(x_linsp, model(x_linsp), 'yellow', label='Poly Fit')
    plt.plot(x_linsp, func(x_linsp, *popt), 'orange', label='Curve Fit Optimizer')
    plt.plot(X, lin.predict(poly.fit_transform(X)), color = 'red', label='Regression Polynomial')
    plt.plot(x_linsp, poly_lagrange(x_linsp), 'green', label='Interpolation de Lagrange')
    plt.plot(x_xgboost, xgb_predictions, 'black', label='XGBoost')
    plt.title('Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show(block=True)
    


if __name__ == "__main__":
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_data = os.path.join(directory_of_script,"data")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    args = parser.parse_args()
    
    main(args)    