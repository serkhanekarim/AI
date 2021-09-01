#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


class WordEmbedding:
    '''
    Class containing model used for word embedding
    '''

    def word2vec(self, num_features, embedding_dim):
        '''
        Model and Training
        The Word2Vec model can be implemented as a classifier to distinguish between 
        true context words from skip-grams and false context words obtained through 
        negative sampling. You can perform a dot product between the embeddings of 
        target and context words to obtain predictions for labels and compute loss 
        against true labels in the dataset.
        
        Subclassed Word2Vec Model

        Use the Keras Subclassing API to define your Word2Vec model with the following layers:

        target_embedding: A tf.keras.layers.Embedding layer which looks up the embedding of a 
        word when it appears as a target word. The number of parameters in this 
        layer are (vocab_size * embedding_dim).
        
        context_embedding: Another tf.keras.layers.Embedding layer which looks 
        up the embedding of a word when it appears as a context word. 
        The number of parameters in this layer are the same as those in 
        target_embedding, i.e. (vocab_size * embedding_dim).
        
        dots: A tf.keras.layers.Dot layer that computes the dot product of 
        target and context embeddings from a training pair.
        
        flatten: A tf.keras.layers.Flatten layer to flatten the results 
        of dots layer into logits.

        With the subclassed model, you can define the call() function that accepts 
        (target, context) pairs which can then be passed into their corresponding 
        embedding layer. Reshape the context_embedding to perform a dot product 
        with target_embedding and return the flattened result.
        
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
    
    