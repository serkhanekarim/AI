#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import re
import string

class NLPPreprocessor:
    '''
    Class used to preprocess NLP data
    '''
    
    def custom_standardization(self, input_data):
        '''
        Remove non usable char on text

        Parameters
        ----------
        input_data : Tensor
            A Tensor of type string. The input to be lower-cased.

        Returns
        -------
        Tensor
            Cleaned data

        '''
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                      '[%s]' % re.escape(string.punctuation),
                                      '')

    def vectorize_text(self, vectorize_layer, text, label):
        '''
        Expand dimension of text data

        Parameters
        ----------
        vectorize_layer : Keras Layer
            Text Vectorization function
        text : Tensor
            Containing text data
        label : int
            containing index of the label

        Returns
        -------
        vectorize_layer : Keras Layer
            Text Vectorization function with additional dim in input
        label : int
            index of the label

        '''
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    def vectorize_text_display(self, vectorize_layer, data_set):
        '''
        Créons une fonction pour voir le résultat de l'utilisation de cette couche pour prétraiter 
        certaines données.

        Parameters
        ----------
        vectorize_layer : Keras Layer
            Text Vectorization function
        data_set : tf.data.Dataset

        Returns
        -------
        Print information of data used after the vectorize_layer

        '''
        # retrieve a batch (of 32 reviews and labels) from the dataset
        text_batch, label_batch = next(iter(data_set))
        first_review, first_label = text_batch[0], label_batch[0]
        print("Review", first_review)
        print("Label", data_set.class_names[first_label])
        print("Vectorized review", self.vectorize_text(vectorize_layer, first_review, first_label))
    
    
    
    