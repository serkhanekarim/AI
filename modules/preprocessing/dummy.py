#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from datetime import datetime
from datetime import date as dt
import holidays

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt


class DummyPreprocessor:
    '''
    Class used to preprocess data
    '''
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def preprocess_feature_embedding(self,features,more_data=None,separator=","):
        
        '''
        Make dummy variable on a feature from training and test dataframe containing word separated by comma
        
        Parameters
        ----------
        data : Pandas Dataframe
        more_data : Pandas DataFrame, in any case there is value in test data not 
                    present in training data on feature
        feature : feature name (str) to make the dummy variable
                
        Returns
        -------
        Dataframe : Pandas dataframe with expanded categorical variable (dummy variable) 
                    from feature
        
        Examples
        --------
        >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
        >>> df_drugs_test = pd.read_csv(path_drugs_test, sep=",")  
        >>> preprocess_feature_embedding(data = df_drugs_train,more_data = df_drugs_test,feature="route_of_administration")
                      drug_id  ...  route_of_administration_orale
            0          0_test  ...                             1
            1         0_train  ...                             1
            2       1000_test  ...                             1
            3      1000_train  ...                             1
            4       1001_test  ...                             1  
        '''
        
        for feature in features:
            if more_data is not None:
                list_feature = set([word for value in set(pd.concat([self.dataframedata[feature], more_data[feature]], ignore_index=True)) for word in value.split(separator)])
            else:
                list_feature = set([word for value in set(self.dataframedata[feature]) for word in value.split(separator)])
            
            for value in tqdm(list_feature):
                self.dataframedata[feature + "_" + value] = 0
                for index in range(self.dataframe.shape[0]):
                    if value in self.dataframe[feature][index]:
                        self.dataframedata[feature + "_" + value][index] = 1
                        
        return self.dataframe
                
                
    def preprocess_NLP(self, data,features):
        
        '''
        Cleaning text, remove additional space, lowercase text
        
        Parameters
        ----------
        data : Pandas Dataframe
        features : list of feature name (str) where the text cleaning occurs
                
        Returns
        -------
        Dataframe : Pandas dataframe with the cleaned feature
        
        Examples
        --------
        >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
        >>> df_drugs_test = pd.read_csv(path_drugs_test, sep=",")  
        >>> preprocess_NLP(data = df_drugs_train,feature="description")
                      drug_id  ...  description
            0         0_train  ...  5 flacon(s) en verre de 0,5 ml
            1         1_train  ...  plaquette(s) pvc pvdc aluminium de 28 comprimé(s)
            2         2_train  ...  plaquette(s) pvc pvdc aluminium de 28 comprimé(s)
        '''   
        
        for feature in tqdm(features) :
            data[feature] = data[feature].apply(lambda x : str(x).lower())
            data[feature] = data[feature].apply(lambda x : re.sub(r'\s+', ' ',x)) #Remove additional spaces  
            data[feature] = data[feature].apply(lambda x : re.sub(r'\\n|^ | $', '',x)) #Remove line break and space at the beginning and the end
            data[feature] = data[feature].apply(lambda x : re.sub(r' *([,/]) *', '\1',x)) #Remove space between comma or slash        
    
    def NaN_imputation_FeatureScaling_PCA(data,imputation_strategy):
        '''
        Missing value imputation, StandardScaler and PCA transformation
        '''        
        print("Missing value imputation...")
        imputer = SimpleImputer(missing_values=np.nan,strategy=imputation_strategy)
        imputer = imputer.fit(data)
        data = imputer.transform(data)
        
        print("Feature Scaling...")
        data = StandardScaler().fit_transform(data)
        
        print("PCA transformation...")
        pca = PCA()
        principalComponents = pca.fit_transform(data)
        df_PCA = pd.DataFrame(principalComponents)
        
        print("Data preprocessing DONE")
        return df_PCA


def preprocess_whole_pipeline(df_train,
                              features_to_NLP,
                              features_to_embed,
                              features_to_dummy,
                              NaN_imputation_feature_scaling_PCA_boolean,
                              label,                         
                              column_to_remove=None,
                              df_test=None):
    '''
    Preprocessing variables from train and test data, feature engineering 
    and active ingredients data using preprocessor functions
    '''
    
    preprocessor = preprocessing()
    
    print("Preprocessing variables from train, test data, feature engineering and active ingredients data: \n")
    #Clean text from dataframe feature (lowercase and remove additional spaces)
    preprocessor.preprocess_NLP(data = df_train, features=features_to_NLP)   
    #Make dummy variable with feature_to_embed
    preprocessor.preprocess_feature_embedding(data = df_train,more_data=df_test,feature=features_to_embed)
    if df_test is not None:
        preprocessor.preprocess_NLP(data = df_test, features=features_to_NLP)
        preprocessor.preprocess_feature_embedding(data = df_test,more_data=df_train,feature=features_to_embed)    
    
    #Store label separately from train data for ML method
    df_label = df_train[label]
    df_train.drop(label, axis=1, inplace=True)
        
    #Make dummy variable with string/objcect features by concatening train and test data to avoid missing dummy variable that
    #are present in train or test data and not present in test or train data
    data = pd.get_dummies(pd.concat([df_train, df_test], ignore_index=True),columns=features_to_dummy,prefix_sep="__")
    
    #Revove special chars from column name for the ML method
    data.rename(columns=lambda x: re.sub(r'[\s+()\',\[\]<>]',"_",str(x)),inplace=True)
    
    #Remove useless feature
    if column_to_remove is not None:
        data.drop(column_to_remove, axis=1, inplace=True)
    
    if NaN_imputation_feature_scaling_PCA_boolean == "True":
        '''
        Missing value imputation, StandardScaler and PCA transformation
        '''
        data = preprocessor.NaN_imputation_FeatureScaling_PCA(data=data,
                                                   imputation_strategy="median")        
        
        #Get again train and test data using preprocessed data (data = data_train + data_test)
        df_train = data[0:df_train.shape[0]]
        df_test = data[df_train.shape[0]:data.shape[0]]
    
    
    return df_train,df_test,df_label
    