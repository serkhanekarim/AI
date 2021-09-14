#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

        
class Word2Vec(tf.keras.Model):
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
    '''
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        
        self.context_embedding = layers.Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns+1)
    
    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
          target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots
    
    def custom_loss(x_logit, y_true):
        '''
        Define loss function and compile model
        For simplicity, you can use tf.keras.losses.CategoricalCrossEntropy as an 
        alternative to the negative sampling loss. If you would like to write 
        your own custom loss function, you can also do so as follows:

        Parameters
        ----------
        x_logit : TYPE
            DESCRIPTION.
        y_true : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

    
    