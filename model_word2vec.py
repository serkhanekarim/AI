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


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

    
def main(args):
    
    # sentence = "The wide road shimmered in the hot sun"
    # tokens = list(sentence.lower().split())
    # print(len(tokens))
    
    # vocab, index = {}, 1  # start indexing from 1
    # vocab['<pad>'] = 0  # add a padding token
    # for token in tokens:
    #   if token not in vocab:
    #     vocab[token] = index
    #     index += 1
    # vocab_size = len(vocab)
    # print(vocab)

    # inverse_vocab = {index: token for token, index in vocab.items()}
    # print(inverse_vocab)

    # example_sequence = [vocab[word] for word in tokens]
    # print(example_sequence)
    
    # window_size = 2
    # positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    #   example_sequence,
    #   vocabulary_size=vocab_size,
    #   window_size=window_size,
    #   negative_samples=0)
    # print(len(positive_skip_grams))
    # print(positive_skip_grams)
    
    # for target, context in positive_skip_grams[:5]:
    #     print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
        
    # # Get target and context words for one positive skip-gram.
    # target_word, context_word = positive_skip_grams[0]
    # print("--------")
    # print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])
    # print(positive_skip_grams[0])
    
    # # Set the number of negative samples per positive context.
    # num_ns = 4
    
    # context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
    # negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    #     true_classes=context_class,  # class that should be sampled as 'positive'
    #     num_true=1,  # each positive skip-gram has 1 positive context class
    #     num_sampled=num_ns,  # number of negative context words to sample
    #     unique=True,  # all the negative samples should be unique
    #     range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
    #     seed=SEED,  # seed for reproducibility
    #     name="negative_sampling"  # name of this operation
    # )
    # print(negative_sampling_candidates)
    # print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

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
    print(lines)
    
    quit()

    
    '''
    Classification de texte de base
    Ce didacticiel illustre la classification de texte à partir de fichiers de texte 
    brut stockés sur disque. Vous entraînerez un classificateur binaire pour effectuer 
    une analyse des sentiments sur un ensemble de données IMDB. À la fin du bloc-notes, 
    vous pourrez essayer un exercice dans lequel vous apprendrez à un classificateur 
    multiclasse à prédire la balise d'une question de programmation sur Stack Overflow.
    '''   
    
    '''
    Analyse des sentiments

    Ce portable forme un modèle d'analyse des sentiments pour classer les critiques de 
    films comme positifs ou négatifs, sur la base du texte de l'examen. Ceci est un 
    exemple de binaire -ou classification à deux classes, une espèce importante et 
    largement applicable problème d' apprentissage de la machine.

    Vous utiliserez le Grand Film Review Dataset qui contient le texte de 50.000 
    critiques de films de la Internet Movie Database . Ceux-ci sont divisés en 
    25 000 avis pour la formation et 25 000 avis pour les tests. Les ensembles 
    de formation et d' essai sont équilibrés, ce qui signifie qu'ils contiennent 
    un nombre égal de commentaires positifs et négatifs.
    Téléchargez et explorez l'ensemble de données IMDB
    Téléchargeons et extrayons l'ensemble de données, puis explorons la structure du répertoire.
    '''
    data_directory = args.data_directory
    size_data_directory = os.path.getsize(data_directory)
    if size_data_directory <= 4096:
        tf.keras.utils.get_file(
            fname=os.path.join(data_directory,'aclImdb_v1'),
            origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            extract=True,
            cache_subdir='.',
            cache_dir=data_directory)
    
    dataset_dir = os.path.join(data_directory,'aclImdb')
    
    '''
    Les aclImdb/train/pos et aclImdb/train/neg répertoires contiennent de nombreux fichiers 
    texte, dont chacun est une seule critique de film. Jetons un coup d'œil à l'un d'eux.
    '''
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())

    '''
    Charger le jeu de données
    Ensuite, vous allez charger les données sur le disque et les préparer 
    dans un format adapté à l'entraînement. Pour ce faire, vous utiliserez 
    l'utile text_dataset_from_directory utilitaire, qui attend une structure 
    de répertoire comme suit:
        main_directory/
        ...class_a/
        ......a_text_1.txt
        ......a_text_2.txt
        ...class_b/
        ......b_text_1.txt
        ......b_text_2.txt
    Pour préparer un ensemble de données pour la classification binaire, 
    vous aurez besoin de deux dossiers sur le disque correspondant à class_a et class_b . 
    Ceux - ci seront les critiques de films positifs et négatifs, qui peuvent être trouvés d
    ans aclImdb/train/pos et aclImdb/train/neg . Comme l'ensemble de données IMDB contient 
    des dossiers supplémentaires, vous les supprimerez avant d'utiliser cet utilitaire.
    '''
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    
    '''
    Ensuite, vous utiliserez l' text_dataset_from_directory utilitaire pour créer une 
    étiquette tf.data.Dataset . tf.data est une collection d'outils puissants pour 
    travailler avec des données.
    
    Lors de l' exécution d' une expérience d'apprentissage de la machine, il est 
    recommandé de diviser votre ensemble de données en trois divisions: le train , 
    la validation et essai .
    
    L'ensemble de données IMDB a déjà été divisé en train et test, mais il manque un 
    ensemble de validation. Créons un ensemble de validation à l' aide d' une 80:20 division 
    des données de formation en utilisant le validation_split arguments ci - dessous.
    '''
    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(directory=train_dir, 
                                                                      batch_size=batch_size, 
                                                                      validation_split=0.2, 
                                                                      subset='training', 
                                                                      seed=seed)
   
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(directory=train_dir, 
                                                                     batch_size=batch_size, 
                                                                     validation_split=0.2, 
                                                                     subset='validation', 
                                                                     seed=seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(directory=test_dir, 
                                                                     batch_size=batch_size)

    '''
    Préparer l'ensemble de données pour la formation
    Ensuite, vous normalisera, tokenize et vectoriser les données en utilisant 
    l'utile preprocessing.TextVectorization couche.
    La normalisation fait référence au prétraitement du texte, généralement pour 
    supprimer la ponctuation ou les éléments HTML afin de simplifier l'ensemble de 
    données. La tokenisation fait référence au fractionnement de chaînes en jetons 
    (par exemple, fractionnement d'une phrase en mots individuels, en fractionnant sur 
     des espaces). La vectorisation fait référence à la conversion de jetons en 
    nombres afin qu'ils puissent être introduits dans un réseau de neurones. 
    Toutes ces tâches peuvent être accomplies avec cette couche.
    
    Comme vous l' avez vu ci - dessus, les commentaires contiennent différentes 
    balises HTML comme <br /> . Ces balises ne seront pas supprimés par le 
    normalisateur par défaut dans la TextVectorization couche 
    (qui convertit le texte en minuscules et des bandes de ponctuation par défaut, mais ne 
    supprime pas HTML). Vous allez écrire une fonction de normalisation personnalisée 
    pour supprimer le code HTML.
    
    Ensuite, vous allez créer une TextVectorization couche. Vous utiliserez cette couche 
    pour standardiser, tokeniser et vectoriser nos données. Vous définissez le output_mode à 
    int pour créer des indices entiers uniques pour chaque jeton.

    Notez que vous utilisez la fonction de division par défaut et la fonction de 
    normalisation personnalisée que vous avez définie ci-dessus. 
    Vous aurez également définir des constantes pour le modèle, comme un maximum 
    de explicite sequence_length , ce qui entraînera la couche à des séquences de 
    pad ou tronquer exactement sequence_length valeurs.
    '''
    
    max_features = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(standardize=NLPPreprocessor().custom_standardization,
                                        max_tokens=max_features,
                                        output_mode='int',
                                        output_sequence_length=sequence_length)
    '''
    Ensuite, vous appellerez adapt en fonction de l'état de la couche pré - traitement à 
    l'ensemble de données. Cela amènera le modèle à construire un index de chaînes en nombres entiers.
    '''   
    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    '''
    Créons une fonction pour voir le résultat de l'utilisation de cette couche pour prétraiter 
    certaines données.
    '''
    NLPPreprocessor().vectorize_text_display(vectorize_layer=vectorize_layer, 
                                             data_set=raw_train_ds)
    
    '''
    Comme vous pouvez le voir ci-dessus, chaque jeton a été remplacé par un entier. 
    Vous pouvez rechercher le jeton (chaîne) que chaque entier correspond à 
    l' appel par .get_vocabulary() sur la couche.
    '''

    print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
    print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

    '''
    Vous êtes presque prêt à entraîner votre modèle. Comme dernière étape de prétraitement, 
    vous appliquerez la couche TextVectorization que vous avez créée précédemment à 
    l'ensemble de données d'apprentissage, de validation et de test.
    '''
    train_ds = raw_train_ds.map(lambda x,y : NLPPreprocessor().vectorize_text(vectorize_layer,x,y))
    val_ds = raw_val_ds.map(lambda x,y : NLPPreprocessor().vectorize_text(vectorize_layer,x,y))
    test_ds = raw_test_ds.map(lambda x,y : NLPPreprocessor().vectorize_text(vectorize_layer,x,y))

    '''
    Configurer l'ensemble de données pour les performances
    Ce sont deux méthodes importantes que vous devez utiliser lors du chargement des données 
    pour vous assurer que les E/S ne deviennent pas bloquantes.
    .cache() conserve les données en mémoire après son chargement de disque. Cela garantira 
    que l'ensemble de données ne deviendra pas un goulot d'étranglement lors de l'entraînement 
    de votre modèle. Si votre ensemble de données est trop volumineux pour tenir dans la mémoire, 
    vous pouvez également utiliser cette méthode pour créer un cache sur disque performant, 
    qui est plus efficace à lire que de nombreux petits fichiers.
    .prefetch() chevauche les données de prétraitement et l' exécution du modèle pendant 
    l' entraînement.
    Vous pouvez en apprendre davantage sur les méthodes, ainsi que la façon dont les données 
    du cache sur le disque dans le guide de performance des données .
    '''
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    '''
    Fonction de perte et optimiseur
    Un modèle a besoin d'une fonction de perte et d'un optimiseur pour l'entraînement. 
    Comme il s'agit d' un problème de classification binaire et le modèle délivre une 
    probabilité (une seule couche unité avec une activation sigmoïde), vous utiliserez 
    losses.BinaryCrossentropy fonction de perte.
    Maintenant, configurez le modèle pour utiliser un optimiseur et une fonction de perte :
    '''

    model = TextModels().basic_NN(num_features=max_features, embedding_dim=16)
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    '''
    Former le modèle
    Vous formerez le modèle en passant l' dataset de dataset objet à la méthode d' ajustement.
    '''
    history = model.fit(train_ds,validation_data=val_ds,epochs=1)

    '''
    Évaluer le modèle
    Voyons comment le modèle fonctionne. Deux valeurs seront renvoyées. 
    Perte (un nombre qui représente notre erreur, les valeurs inférieures sont meilleures) et précision.
    '''
    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    
    '''
    Vérifions les courbes de perte d'entraînement et de validation pour voir comment 
    votre modèle s'est amélioré pendant l'entraînement.
    '''
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    
    export_model = tf.keras.Sequential([
      vectorize_layer,
      model,
      layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)



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