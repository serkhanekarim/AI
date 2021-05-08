#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pandasgui import show

class DataVisualizator:
    '''
    Class used to make vizulisation of data
    '''
    
    MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = 20
    PKMN_TYPE_COLORS = ['#78C850',  # Grass
                        '#F08030',  # Fire
                        '#6890F0',  # Water
                        '#A8B820',  # Bug
                        '#A8A878',  # Normal
                        '#A040A0',  # Poison
                        '#F8D030',  # Electric
                        '#E0C068',  # Ground
                        '#EE99AC',  # Fairy
                        '#C03028',  # Fighting
                        '#F85888',  # Psychic
                        '#B8A038',  # Rock
                        '#705898',  # Ghost
                        '#98D8D8',  # Ice
                        '#7038F8',  # Dragon
                       ]
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def data_inforamtion(self, option=None):
        '''
        Display the dataframe

        Parameters
        ----------
        option : string, optional
            string containing option to display or not dataframe on Notebook, or using PandasGUI. The default is None.

        Returns
        -------
        Display information of the dataframe

        '''
        print("Plotting DataFrame information...")
        # Option to display all columns
        pd.set_option("display.max.columns", None)
        # Display Dataframe on Notebook
        if option == "notebook": display(HTML(self.dataframe.to_html()))
        # Display Dataframe on PandasGUI
        if option == "pandasgui": show(self.dataframe)
        # Display Dataframe information on Console
        #print(self.dataframe.head().style.bar(color='red'))
        print(self.dataframe.head())
        print(self.dataframe.info())
        
    def missing_value_plotting(self, length, width):
        '''
        Display plot for the missing value of the dataframe

        Parameters
        ----------
        length : int
            length of the figure.
        width : int
            width of the figure.

        Returns
        -------
        Display plot for the missing value of the dataframe and save them.

        '''
        
        print("Plotting Missing Values...")
        
        '''
        The sparkline at right summarizes the general shape of the data completeness 
        and points out the rows with the maximum and minimum nullity in the dataset.
        '''
        msno.bar(self.dataframe)
        plt.title("Matrice des valeurs manquantes des données\n", fontsize=18)
        
        msno.matrix(self.dataframe)
        plt.title("Diagramme à barres des valeurs manquantes des données\n", fontsize=18)
        
        '''
        A value near -1 means if one variable appears then the other variable is very likely to be missing.
        A value near 0 means there is no dependence between the occurrence of missing values of two variables.
        A value near 1 means if one variable appears then the other variable is very likely to be present.
        '''
        msno.heatmap(self.dataframe)
        plt.title("Diagramme à barres des valeurs manquantes des données\n", fontsize=18)

        msno.dendrogram(self.dataframe)
        
    
    def violin_plot(self, columns_name, label):
        '''
        Display the violin plot for all features from the dataframe

        Parameters
        ----------
        None.

        Returns
        -------
        Display the violin plot for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = set(columns_name)       
        for column_name_continuous in set(columns_name):
            if self.dataframe[column_name_continuous].dtypes != "O" and column_name_continuous != label:
                for column_name_categorical in set(columns_name):
                    if self.dataframe[column_name_categorical].dtypes == "O" and column_name_categorical!= label and len(set(self.dataframe[column_name_categorical])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                        print("Plotting violin plot of " + column_name_categorical.upper() + " on " + column_name_continuous.upper() + "...")
                        plt.figure()
                        with sns.axes_style(style=None):
                            sns.violinplot(x=column_name_categorical, y=column_name_continuous, hue=label, data=self.dataframe, split=True, palette=self.PKMN_TYPE_COLORS);
                            plt.title("Diagramme en violon de " + column_name_categorical.upper() + " sur " + column_name_continuous.upper() + "\n", fontsize=18)

    def box_plot(self, columns_name, label):
        '''
        Display the box plot for all features from the dataframe

        Parameters
        ----------
        None.

        Returns
        -------
        Display the box plot for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = set(columns_name)       
        for column_name_continuous in set(columns_name):
            if self.dataframe[column_name_continuous].dtypes != "O" and column_name_continuous != label:
                for column_name_categorical in set(columns_name):
                    if self.dataframe[column_name_categorical].dtypes == "O" and column_name_categorical != label and len(set(self.dataframe[column_name_categorical])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                        print("Plotting box plot of " + column_name_categorical.upper() + " on " + column_name_continuous.upper() + "...")
                        plt.figure()
                        sns.boxplot(x=column_name_categorical, y=column_name_continuous, hue=label, data=self.dataframe)
                        plt.title("Boîte à moustaches de " + column_name_categorical.upper() + " sur " + column_name_continuous.upper() + "\n", fontsize=18)
    
    def correlation_matrix(self):
        '''
        Display the correlation matrix between all features from the dataframe

        Parameters
        ----------
        None.

        Returns
        -------
        Display correlation matrix between all feature and save it

        '''
        print("Plotting Correlation between all features...")
        sns.heatmap(self.dataframe.corr(), cmap='Reds', annot=True, linewidths=1)
        plt.title("Matrice de corrélation entre les différentes caractéristiques\n", fontsize=18, color='#c0392b')

    def pair_plot(self, label, height=2):
        '''
        Plot pairwise relationships in a dataset.

        Parameters
        ----------
        label : string
            string containing the column names to highlight in the plotting

        Returns
        -------
        Display plot pairwise relationships in a dataset and save them.

        '''
        print("Plotting pairwise relationships between all features...")
        g = sns.pairplot(self.dataframe, hue=label, height=height, corner=False)
        g.fig.suptitle("Graphique des relations entre les variables et " + label.upper() + "\n", fontsize=18)

    def histogram(self, columns_name):
        '''
        Display histogram of column name from the DataFrame.

        Parameters
        ----------
        columns_name : string or list of string
            string containing the column names for the display of the histogram.

        Returns
        -------
        Display the histogram and save them.

        '''
        if isinstance(columns_name,str): columns_name = set(columns_name)       
        for column_name in set(columns_name):
            if self.dataframe[column_name].dtypes != "O":
                print("Ploting DataFrame histogram for " + column_name.upper() + "...")
                plt.figure()
                # Distribution Plot (a.k.a. Histogram)
                sns.distplot(self.dataframe[column_name])
                plt.ylabel('Frequency')
                plt.xlabel(column_name)
                plt.title("Histogramme de " + column_name.upper() + "\n", fontsize=18)

    def bar_chart(self, columns_name):
        '''
        Display Bar Chart of column name from the DataFrame.

        Parameters
        ----------
        columns_name : string or list of string
            string containing the column names for the display of the bar chart.

        Returns
        -------
        Display the Pie Chart and save them.

        '''
        
        if isinstance(columns_name,str): columns_name = set(columns_name)       
        for column_name in set(columns_name):
            if len(set(self.dataframe[column_name])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                print("Ploting DataFrame Bar Chart for " + column_name.upper() + "...")
                plt.figure()
                # Count Plot (a.k.a. Bar Plot)
                sns.countplot(x=column_name, data=self.dataframe, palette=self.PKMN_TYPE_COLORS)
                # Rotate x-labels
                plt.xticks(rotation=-45)
                plt.title("Diagramme à barres de " + column_name.upper() + "\n", fontsize=18)
                
    def _autopct_format(self, values):
        '''
        Convert percentage into number of values /occurences for pie chart

        Parameters
        ----------
        values : DataFrame
            Dataframe containing number of occuerences.

        Returns
        -------
        int
            Return the number of values.

        '''
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_format
    
    def pie_chart(self, columns_name, autopct='%1.1f%%'):
        '''
        Display Pie Chart of column name from the DataFrame

        Parameters
        ----------
        columns_name : string or list of string
            string containing the column names for the display of the pie chart.
        autopct : string, optional
            string containing mode for the display or not of number of occurences in Pie Chart instead of percentage. The default is '%1.1f%%'.

        Returns
        -------
            Display the Pie Chart and save them.

        '''
        
        if isinstance(columns_name,str): columns_name = set(columns_name)       
        for column_name in set(columns_name):
            sizes = self.dataframe[column_name].value_counts(dropna=False)
            if len(set(self.dataframe[column_name])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                print("Ploting DataFrame Pie Chart for " + column_name.upper() + "...")
                explode = (sizes == max(sizes)) * 0.01
                labels = sizes.index
                if autopct == "number" : autopct = self._autopct_format(sizes)                
                plt.figure()
                plt.pie(sizes, explode=explode, labels=labels, autopct=autopct, shadow=True, startangle=90)
                plt.legend(loc='best', bbox_to_anchor=(-0.1, 1),fancybox=True, shadow=True)
                plt.title("Diagramme Circulaire de " + column_name.upper() + "\n", fontsize=18)