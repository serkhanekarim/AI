#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


class TextModels:
    '''
    Class containing model used for text classification
    '''

    def basic_NN(self, num_features, embedding_dim):
        '''
        Les couches sont empilées séquentiellement pour construire le classificateur :

       La première couche est une Embedding couche. Cette couche prend les révisions codées 
       en nombres entiers et recherche un vecteur d'intégration pour chaque index de mot. 
       Ces vecteurs sont appris comme les trains miniatures. Les vecteurs ajoutent une 
       dimension au tableau de sortie. Les dimensions résultantes sont les suivantes : 
           (batch, sequence, embedding)
       Ensuite, une GlobalAveragePooling1D couche renvoie un vecteur de sortie de longueur 
       fixe pour chaque exemple en faisant la moyenne sur la dimension de la séquence. 
       Cela permet au modèle de gérer l'entrée de longueur variable, de la manière la plus 
       simple possible.
       Ce vecteur de sortie de longueur fixe est canalisé à travers un 
       (entièrement connecté Dense couche) avec 16 unités cachées.
       La dernière couche est densément connectée à un seul nœud de sortie.
        
        Parameters
        ----------
        num_features: int
            Size of the vocabulary, i.e. maximum integer index + 1.
        embedding_dim : int
            Dimension of the dense embedding.

        Returns
        -------
        Keras Sequential Model

        '''
        
        model = tf.keras.Sequential([
          layers.Embedding(num_features + 1, embedding_dim),
          layers.Dropout(0.2),
          layers.GlobalAveragePooling1D(),
          layers.Dropout(0.2),
          layers.Dense(1)])
        
        model.summary()
        
        return model
    
    