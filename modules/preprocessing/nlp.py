#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import re
import string
import tqdm

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
        
    
    def generate_training_data(self, sequences, window_size, num_ns, vocab_size, seed):
        '''
        Generates skip-gram pairs with negative sampling for a list of sequences
        (int-encoded sentences) based on window size, number of negative samples
        and vocabulary size.
        
        https://www.tensorflow.org/tutorials/text/word2vec

        Parameters
        ----------
        sequences : list
            sentences in list
        window_size : TYPE
            The window size determines the span of words on either side of a 
            target_word that can be considered context word
        num_ns : int
            number of negative samples for a given target word in a window.
        vocab_size : int
            Size of all unique word from all the sentences
        seed : int
            seed for reproducibility

        Returns
        -------
        targets : list
            List of all targets word
        contexts : list
            Concat of positive context word with negative sampled words.
        labels : list
            Label first context word as 1 (positive) followed by num_ns 0s (negative).

        '''
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []
      
        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
      
        # Iterate over all sequences (sentences) in dataset.
        for sequence in tqdm.tqdm(sequences):
      
          # Generate positive skip-gram pairs for a sequence (sentence).
          positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)
      
          # Iterate over each positive skip-gram pair to produce training examples
          # with positive context word and negative samples.
          for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,  # class that should be sampled as 'positive'
                num_true=1,  # each positive skip-gram has 1 positive context class
                num_sampled=num_ns,  # number of negative context words to sample
                unique=True,  # all the negative samples should be unique
                range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
                seed=seed,  # seed for reproducibility
                name="negative_sampling"  # name of this operation
                )

      
            # Build context and label vectors (for one target word)
            # Add a dimension so you can use concatenation (on the next step).
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
      
            # Concat positive context word with negative sampled words.
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            # Label first context word as 1 (positive) followed by num_ns 0s (negative).
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
      
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
      
        return targets, contexts, labels
    
    
    