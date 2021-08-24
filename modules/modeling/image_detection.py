#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


class Models:
    '''
    Class containing model used for speech recognition
    '''

    def basic_NN(self, input_shape, num_labels):
        '''
        La première couche de ce réseau, tf.keras.layers.Flatten , transforme le format des 
        images d'un tableau bidimensionnel (de 28 par 28 pixels) en un tableau 
        unidimensionnel (de 28 * 28 = 784 pixels). Considérez ce calque comme un dépilage 
        de rangées de pixels dans l'image et leur alignement. Cette couche n'a pas de paramètres 
        à apprendre ; il reformate seulement les données.
        
        Une fois les pixels aplatis, le réseau se compose d'une séquence de deux 
        couches tf.keras.layers.Dense . Ce sont des couches neuronales densément c
        onnectées ou entièrement connectées. La première couche Dense a 128 nœuds 
        (ou neurones). La deuxième (et dernière) couche renvoie un tableau logits 
        d'une longueur de 10. Chaque nœud contient un score qui indique que l'image 
        actuelle appartient à l'une des 10 classes.
        
        Parameters
        ----------
        input_shape : Tensor
            Tensor indicating the shape of the input
        num_labels : int
            number of different labels

        Returns
        -------
        Keras Sequential Model

        '''
        
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_labels)
        ])
        
        model.summary()
        
        return model
    