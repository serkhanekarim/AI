#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pathlib

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from modules.reader.reader import DataReader
from modules.preprocessing.nlp import NLPPreprocessor
from modules.modeling.word_embedding import Word2Vec


SEED = 42
NUM_NS = 4
AUTOTUNE = tf.data.AUTOTUNE

    
def main(args):
    '''
    Download text corpus
    You will use a text file of Shakespeare's writing for this tutorial. 
    Change the following line to run this code on your own data.
    '''
    
    data_directory = args.data_directory
    size_data_directory = os.path.getsize(data_directory)
    path_to_file = os.path.join(data_directory,'shakespeare.txt')
    if size_data_directory <= 4096:
        tf.keras.utils.get_file(
            fname=path_to_file,
            origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
            extract=False,
            cache_subdir='.',
            cache_dir=data_directory)

    lines = DataReader(path_to_file).read_data_file()
    text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    
    # Define the vocabulary size and number of words in a sequence.
    vocab_size = 4096
    sequence_length = 10

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Set output_sequence_length length to pad all samples to same length.
    vectorize_layer = layers.experimental.preprocessing.TextVectorization(
        standardize=NLPPreprocessor().custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
        )
    
    '''
    Call adapt on the text dataset to create vocabulary.
    '''
    vectorize_layer.adapt(text_ds.batch(1024))
    
    '''
    Once the state of the layer has been adapted to represent the text corpus, 
    the vocabulary can be accessed with get_vocabulary(). This function returns 
    a list of all vocabulary tokens sorted (descending) by their frequency. 
    '''
    # Save the created vocabulary for reference.
    inverse_vocab = vectorize_layer.get_vocabulary()
    
    '''
    The vectorize_layer can now be used to generate vectors for each element in the text_ds.
    '''
    # Vectorize the data in text_ds.
    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

    '''
    Obtain sequences from the dataset
    You now have a tf.data.Dataset of integer encoded sentences. 
    To prepare the dataset for training a Word2Vec model, flatten the dataset into 
    a list of sentence vector sequences. This step is required as you would iterate over 
    each sentence in the dataset to produce positive and negative examples. 
    '''
    sequences = list(text_vector_ds.as_numpy_iterator())
    
    '''
    Generate training examples from sequences
    sequences is now a list of int encoded sentences. 
    Just call the generate_training_data() function defined earlier to generate training 
    examples for the Word2Vec model. To recap, the function iterates over each word from 
    each sequence to collect positive and negative context words. Length of target, 
    contexts and labels should be same, representing the total number of training examples.
    '''
    targets, contexts, labels = NLPPreprocessor().generate_training_data(sequences=sequences,
                                                                         window_size=2,
                                                                         num_ns=NUM_NS,
                                                                         vocab_size=vocab_size,
                                                                         seed=SEED)

    targets = np.array(targets)
    contexts = np.array(contexts)[:,:,0]
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")
    
    '''
    Configure the dataset for performance
    To perform efficient batching for the potentially large number of training examples, 
    use the tf.data.Dataset API. After this step, you would have a tf.data.Dataset object 
    of (target_word, context_word), (label) elements to train your Word2Vec model!
    '''
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    '''
    Add cache() and prefetch() to improve performance.
    '''
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)

    '''
    It's time to build your model! Instantiate your Word2Vec class with an 
    embedding dimension of 128 (you could experiment with different values). 
    Compile the model with the tf.keras.optimizers.Adam optimizer. 
    '''
    embedding_dim = 128
    word2vec = Word2Vec(vocab_size, embedding_dim, NUM_NS)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    '''
    Also define a callback to log training statistics for tensorboard.
    '''
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    
    '''
    Train the model with dataset prepared above for some number of epochs.
    '''
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])


if __name__ == "__main__":
    
    MODEL_NAME = "speech-recognition-cnn.model"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_model = os.path.join(directory_of_script,"model")
    directory_of_results = os.path.join(directory_of_script,"results")
    directory_of_data = os.path.join(directory_of_script,"DATA","Shakespeare")
    os.makedirs(directory_of_model,exist_ok=True)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    # parser.add_argument("-path_model", help="Path of an existing model to use for prediction", required=False, default=os.path.join(directory_of_model,MODEL_NAME), nargs='?')
    # parser.add_argument("-NaN_imputation_feature_scaling_PCA_usage", help="Apply or not NaN imputation, Feature Scaling and PCA", required=False, choices=["False","True"], default="False", nargs='?')
    # parser.add_argument("-max_depth", help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist)", required=False, default=6, type=int, nargs='?')
    # parser.add_argument("-eta", help="Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. range: [0,1]", required=False, default=0.3, type=float, nargs='?')
    # parser.add_argument("-num_round", help="The number of rounds for boosting", required=False, default=100, type=int, nargs='?')
    args = parser.parse_args()
    
    main(args)    