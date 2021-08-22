#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import learning_curve

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pandasgui import show

from modules.Global.variable import Var

class DataVisualizator:
    '''
    Class used to make vizulisation of data
    '''
    
    MAX_NUMBER_OF_CATEGORICAL_OCCURENCES = Var().MAX_NUMBER_OF_CATEGORICAL_OCCURENCES
    MAX_NUMBER_OF_FEATURE_IMPORTANCE = Var().MAX_NUMBER_OF_FEATURE_IMPORTANCE
    PKMN_TYPE_COLORS = Var().PKMN_TYPE_COLORS
    
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
        
    def missing_value_plotting(self):
        '''
        Display plot for the missing value of the dataframe

        Parameters
        ----------
        None.

        Returns
        -------
        Display plot for the missing value of the dataframe and save them.

        '''
        
        print("Plotting Missing Values...")
        
        '''
        The sparkline at right summarizes the general shape of the data completeness 
        and points out the rows with the maximum and minimum nullity in the dataset.
        '''
        plt.figure()
        msno.bar(self.dataframe)
        plt.title("Matrice des valeurs manquantes des données\n", fontsize=18)
        
        plt.figure()
        msno.matrix(self.dataframe)
        plt.title("Diagramme à barres des valeurs manquantes des données\n", fontsize=18)
        
        '''
        A value near -1 means if one variable appears then the other variable is very likely to be missing.
        A value near 0 means there is no dependence between the occurrence of missing values of two variables.
        A value near 1 means if one variable appears then the other variable is very likely to be present.
        '''
        plt.figure()
        msno.heatmap(self.dataframe)
        plt.title("Diagramme à barres des valeurs manquantes des données\n", fontsize=18)

        plt.figure()
        msno.dendrogram(self.dataframe)
        
    def linear_regression(self, columns_name, label):
        '''
        Display Regression models for all features from the dataframe

        Parameters
        ----------
        columns_name : string or list of string
            containing the features column names for the display of the regression.
        label : string
            string containing the column name of the output.

        Returns
        -------
        Display regression models for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = [columns_name]       
        for column_name in set(columns_name):
            plt.figure()
            sns.regplot(x=column_name, y=label, data=self.dataframe);
            plt.title("Regression model " + str(label).upper() + " sur " + str(column_name).upper() + "\n", fontsize=18)


    def polynomial_regression(self, columns_name, label, model):
        '''
        Display Regression models for all features from the dataframe

        Parameters
        ----------
        columns_name : string or list of string
            containing the features column names for the display of the regression.
        label : string
            string containing the column name of the output.
        model : class
            A convenience class, used to encapsulate “natural” operations on polynomials 
            so that said operations may take on their customary form in code.

        Returns
        -------
        Display regression models for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = [columns_name]       
        for column_name in set(columns_name):
            plt.scatter(column_name, label)
            mini_x = min(self.dataframe[column_name])
            maxi_x = max(self.dataframe[column_name])
            mini_y = min(self.dataframe[label])
            maxi_y = max(self.dataframe[label])
            line = np.linspace(mini_x, maxi_x, 100)        
            DataVisualizator.curve(x=line, 
                              y=model(line), 
                              xlabel=column_name, 
                              ylabel=label, 
                              title="Regression model " + str(label).upper() + " sur " + str(column_name).upper() + "\n", 
                              label=None,
                              xlim=None,
                              ylim=[mini_y, maxi_y],
                              legend_loc=None)
            plt.scatter(self.dataframe[column_name], self.dataframe[label])
            # plt.scatter(self.dataframe[column_name], self.dataframe[label])
            # plt.plot(line, model(line))
            # plt.title("Regression model " + str(label).upper() + " sur " + str(column_name).upper() + "\n", fontsize=18)


        
    
    def violin_plot(self, columns_name, label):
        '''
        Display the violin plot for all features from the dataframe

        Parameters
        ----------
        columns_name : string or list of string
            string containing the column names for the display of the violin plot.
        label : string
            string containing the column names to highlight in the plotting.

        Returns
        -------
        Display the violin plot for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = [columns_name]       
        for column_name_continuous in set(columns_name):
            if self.dataframe[column_name_continuous].dtypes != "O" and column_name_continuous != label:
                for column_name_categorical in set(columns_name):
                    if self.dataframe[column_name_categorical].dtypes == "O" and column_name_categorical!= label and len(set(self.dataframe[column_name_categorical])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                        print("Plotting violin plot of " + str(column_name_categorical).upper() + " on " + str(column_name_continuous).upper() + "...")
                        plt.figure()
                        with sns.axes_style(style=None):
                            sns.violinplot(x=column_name_categorical, y=column_name_continuous, hue=label, data=self.dataframe, split=True, palette=self.PKMN_TYPE_COLORS);
                            plt.title("Diagramme en violon de " + str(column_name_categorical).upper() + " sur " + str(column_name_continuous).upper() + "\n", fontsize=18)

    def box_plot(self, columns_name, label):
        '''
        Display the box plot for all features from the dataframe

        Parameters
        ----------
        columns_name : string or list of string
            string containing the column names for the display of the box plot.
        label : string
            string containing the column names to highlight in the plotting.

        Returns
        -------
        Display the box plot for all all feature and save it

        '''
        
        if isinstance(columns_name,str): columns_name = [columns_name]       
        for column_name_continuous in set(columns_name):
            if self.dataframe[column_name_continuous].dtypes != "O" and column_name_continuous != label:
                for column_name_categorical in set(columns_name):
                    if self.dataframe[column_name_categorical].dtypes == "O" and column_name_categorical != label and len(set(self.dataframe[column_name_categorical])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                        print("Plotting box plot of " + str(column_name_categorical).upper() + " on " + str(column_name_continuous).upper() + "...")
                        plt.figure()
                        sns.boxplot(x=column_name_categorical, y=column_name_continuous, hue=label, data=self.dataframe)
                        plt.title("Boîte à moustaches de " + str(column_name_categorical).upper() + " sur " + str(column_name_continuous).upper() + "\n", fontsize=18)
    
    @staticmethod
    def confusion_matrix(cf,
                        group_names=None,
                        categories='auto',
                        count=True,
                        percent=True,
                        cbar=True,
                        xyticks=True,
                        xyplotlabels=True,
                        sum_stats=True,
                        figsize=None,
                        cmap='Blues',
                        title=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html
                       
        title:         Title for the heatmap. Default is None.
        '''


        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]
    
        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks
    
        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks
    
        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks
    
        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    
    
        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))
    
            #if it is a binary confusion matrix, show some more stats
            if len(cf)==2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""
    
    
        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')
    
        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False
    
    
        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    
        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)
        
        if title:
            plt.title(title)
    
    def correlation_matrix(self):
        '''
        Display the correlation matrix between all features from the dataframe

        Parameters
        ----------
        None

        Returns
        -------
        Display correlation matrix between all feature and save it

        '''
        print("Plotting Correlation between all features and the label...")
        plt.figure(figsize=(2*self.dataframe.shape[1],2*self.dataframe.shape[1]))
        sns.heatmap(self.dataframe.corr(), cmap='Reds', annot=True, linewidths=1)
        plt.title("Matrice de corrélation entre les différentes caractéristiques\n", fontsize=18, color='#c0392b')

    # def correlation_label(self, label):
    #     '''
    #     Display the correlation matrix between all features from the dataframe and a label

    #     Parameters
    #     ----------
    #     label : string
    #         string containing the column names to highlight in the plotting

    #     Returns
    #     -------
    #     Display correlation matrix between all feature and save it

    #     '''
    #     print("Plotting Correlation between all features...")
    #     plt.figure()
    #     corr = self.dataframe.corr()[label].unstack().sort_values().drop_duplicates().iloc
    #     sns.heatmap(self.dataframe.corr(), cmap='Reds', annot=True, linewidths=1)
    #     plt.title("Matrice de corrélation entre les différentes caractéristiques\n", fontsize=18, color='#c0392b')


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
        plt.figure()
        g = sns.pairplot(self.dataframe, hue=label, height=height, corner=False)
        g.fig.suptitle("Graphique des relations entre les variables et " + str(label).upper() + "\n", fontsize=18)

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
        if isinstance(columns_name,str): columns_name = [columns_name]     
        for column_name in set(columns_name):
            if self.dataframe[column_name].dtypes != "O":
                print("Ploting DataFrame histogram for " + str(column_name).upper() + "...")
                plt.figure()
                # Distribution Plot (a.k.a. Histogram)
                sns.distplot(self.dataframe[column_name])
                plt.ylabel('Frequency')
                plt.xlabel(column_name)
                plt.title("Histogramme de " + str(column_name).upper() + "\n", fontsize=18)

    def bar_chart(self, columns_name, label=None):
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
        
        if isinstance(columns_name,str): columns_name = [columns_name]      
        for column_name in set(columns_name):
            if len(set(self.dataframe[column_name])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                print("Ploting DataFrame Bar Chart for " + str(column_name).upper() + "...")
                plt.figure()
                # Count Plot (a.k.a. Bar Plot)
                if column_name == label:
                    sns.countplot(x=column_name, data=self.dataframe, palette=self.PKMN_TYPE_COLORS)
                else:
                    sns.countplot(x=column_name, data=self.dataframe, palette=self.PKMN_TYPE_COLORS, hue=label)
                # Rotate x-labels
                plt.xticks(rotation=-45)
                plt.title("Diagramme à barres de " + str(column_name).upper() + "\n", fontsize=18)
                
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
        
        if isinstance(columns_name,str): columns_name = [columns_name]       
        for column_name in set(columns_name):
            sizes = self.dataframe[column_name].value_counts(dropna=False)
            if len(set(self.dataframe[column_name])) <= self.MAX_NUMBER_OF_CATEGORICAL_OCCURENCES:
                print("Ploting DataFrame Pie Chart for " + str(column_name).upper() + "...")
                explode = (sizes == max(sizes)) * 0.01
                labels = sizes.index
                if autopct == "number" : autopct = self._autopct_format(sizes)                
                plt.figure()
                plt.pie(sizes, explode=explode, labels=labels, autopct=autopct, shadow=True, startangle=90)
                plt.legend(loc='best', bbox_to_anchor=(-0.1, 1),fancybox=True, shadow=True)
                plt.title("Diagramme Circulaire de " + str(column_name).upper() + "\n", fontsize=18)
     
    def features_importance(self, estimator, estimator_type, max_num_features=MAX_NUMBER_OF_FEATURE_IMPORTANCE):
        '''
        Display the importance of features on label from a model based on tree

        Parameters
        ----------
        estimator : estimator object
            Estimator (XGBoost or RandomForest...).
        estimator_type : string
            String which indicate the type of the esrtimator to plot the importance of features

        Returns
        -------
        Display the importance of features on label and save them.

        '''
        
        print("Check importance of features on label\n")
        plt.figure()
        if estimator_type == 'xgboost':
            xgb.plot_importance(estimator,max_num_features=max_num_features)
        else:
            feat_importances = pd.Series(estimator.feature_importances_, index=self.dataframe.columns)
            feat_importances.nlargest(max_num_features).sort_values(ascending=True).plot(kind='barh')
            plt.ylabel('Features')
            plt.xlabel('F score')
            plt.title("Feature importance" + "\n", fontsize=18)

            
        
    @staticmethod    
    def curve(x, 
              y, 
              xlabel, 
              ylabel, 
              title, 
              label=None,
              xlim=None,
              ylim=None,
              legend_loc=None):
        '''
        Display the curve of x,y

        Parameters
        ----------
        x : array
            x coordinate.
        y : array or array of array
            y coordinate.
        label : array, optional
            Array of string ontaining label of curve. The default is None.
        xlabel : string
            name of the label of x
        ylabel : string
            name of the label of y
        title : string
            name of the plotting title

        Returns
        -------
        Display the curve of x,y and save it.

        '''
        
        bool_x_number = isinstance(x[0],int) or isinstance(x[0],float)
        bool_y_number = isinstance(y[0],int) or isinstance(y[0],float)
        
        plt.figure()
        if bool_x_number and bool_y_number:
            plt.plot(x,y,label=label)
        if bool_x_number and not bool_y_number:
            for i in range(len(y)):
                if label is not None:
                    plt.plot(x,y[i],label=label[i])
                else:
                    plt.plot(x,y[i])
        if not bool_x_number and bool_y_number:
            for i in range(len(y)):
                if label is not None:
                    plt.plot(x[i],y,label=label[i])
                else:
                    plt.plot(x[i],y)       
        if not bool_x_number and not bool_y_number:
            for i in range(len(y)):
                if label is not None:
                    plt.plot(x[i],y[i],label=label[i])
                else:
                    plt.plot(x,y)            
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(title)
        plt.legend(loc=legend_loc)
        
    @staticmethod  
    def roc_auc(y_test, y_score, n_classes=1):
        '''
        Display the ROC AUC curve

        Parameters
        ----------
        y_test : array
            Aray of real values
        y_score : array
            Aray of scored/predicted values
        n_classes : int, optional
            Number of predicted classes. The default is 1.

        Returns
        -------
        Display ROC AUC curve and save it

        '''
        # Compute ROC curve and ROC area for each class
        fpr = []
        tpr = []
        roc_auc = []
        if n_classes > 1:
            for i in range(n_classes):
                fpr.append(roc_curve(y_test[:, i], y_score[:, i])[0])
                tpr.append(roc_curve(y_test[:, i], y_score[:, i])[1])
                roc_auc.append(auc(fpr[i], tpr[i]))
        else:
            fpr.append(roc_curve(y_test, y_score)[0])
            tpr.append(roc_curve(y_test, y_score)[1])
            roc_auc.append(auc(fpr[0], tpr[0]))

        # Compute micro-average ROC curve and ROC area
        fpr.append(roc_curve(y_test.ravel(), y_score.ravel())[0])
        tpr.append(roc_curve(y_test.ravel(), y_score.ravel())[1])
        roc_auc.append(auc(fpr[-1], tpr[-1]))
        label = ['ROC curve (area = %0.2f)' % value for value in roc_auc]
        
        DataVisualizator.curve(x=fpr,
                                y=tpr, 
                                xlabel=[0, 1], 
                                ylabel=[0, 1], 
                                title="ROC", 
                                label=label,
                                xlim=[0.0, 1.0],
                                ylim=[0.0, 1.05],
                                legend_loc="lower right")
        
    @staticmethod    
    def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.
    
        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.
    
        title : str
            Title for the chart.
    
        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
    
        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.
    
        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.
    
        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
    
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
    
              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.
    
            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
    
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.
    
        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
    
        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
    
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    
        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
    
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")
    
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")
    
        return plt   

    @staticmethod
    def plot_audio_wave(rows, cols, data_set):
        '''
        Display multiple audio wave plot in a table or rows x cols
        
        Parameters
        ----------
        rows : int
            number of rows to display
        cols : int
            number of columns to display
        data_set : tf.data.Dataset.from_tensor_slices
            Creates a Dataset whose elements are slices of the given tensors.
            The given tensors are sliced along their first dimension. This operation 
            preserves the structure of the input tensors, removing the first dimension 
            of each tensor and using it as the dataset dimension. All input tensors must 
            have the same size in their first dimensions.
        
        Returns
        -------
        Display multiple audio wave plot in a table or rows x cols
        
        '''
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        for i, (audio, label) in enumerate(data_set.take(n)):
          r = i // cols
          c = i % cols
          ax = axes[r][c]
          ax.plot(audio.numpy())
          ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
          label = label.numpy().decode('utf-8')
          ax.set_title(label)
        
        plt.show()
        
    @staticmethod
    def plot_spectrogram(spectrogram, ax):
        '''
        Plot spectogram

        Parameters
        ----------
        spectrogram : Tensorflow tensor
            Contains the the spectogram tensor
        ax : axe
            Axe of the plot

        Returns
        -------
        Display spectrogram from waveform

        '''
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns).
        log_spec = np.log(spectrogram.T)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

        
     
    @staticmethod    
    def plot_waveform_spectrogram(waveform, spectrogram):
        '''
        Plot waveform and spectrogram

        Parameters
        ----------
        waveform : Tensorflow tensor
            Contains the wave form of an audio
        spectrogram : Tensorflow tensor
            Le tf.signal.stft ( tf.signal.stft ) divise le signal en fenêtres temporelles et 
            exécute une transformée de Fourier sur chaque fenêtre, préservant certaines 
            informations temporelles et renvoyant un tenseur 2D sur lequel vous pouvez exécuter 
            des convolutions standard.

        Returns
        -------
        Display spectrogram and waveform

        '''
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns).
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(waveform.shape[0])
        axes[0].plot(timescale, waveform.numpy())
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])

        DataVisualizator.plot_spectrogram(spectrogram.numpy(), axes[1])
    
        axes[1].set_title('Spectrogram')
        plt.show()

    @staticmethod
    def plot_audio_spectrogram_table(rows, cols, data_set, labels):
        '''
        Display multiple audio spectrogram plot in a table or rows x cols
        
        Parameters
        ----------
        rows : int
            number of rows to display
        cols : int
            number of columns to display
        data_set : tf.data.Dataset.from_tensor_slices
            Creates a Dataset whose elements are slices of the given tensors.
            The given tensors are sliced along their first dimension. This operation 
            preserves the structure of the input tensors, removing the first dimension 
            of each tensor and using it as the dataset dimension. All input tensors must 
            have the same size in their first dimensions.
        
        Returns
        -------
        Display multiple audio wave plot in a table or rows x cols
        
        '''
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for i, (spectrogram, label_id) in enumerate(data_set.take(n)):
          r = i // cols
          c = i % cols
          ax = axes[r][c]
          DataVisualizator.plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
          ax.set_title(labels[label_id.numpy()])
          ax.axis('off')
        
        plt.show()

