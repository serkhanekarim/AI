#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from modules.modeling.speech_recognition import Models
from modules.preprocessing.audio import AudioPreprocessor
from modules.visualization.visualization import DataVisualizator


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

    
def main(args):
    '''
    Ce didacticiel vous montrera comment créer un réseau de reconnaissance vocale de base qui 
    reconnaît dix mots différents. Il est important de savoir que les vrais systèmes de 
    reconnaissance vocale et audio sont beaucoup plus complexes, mais comme le MNIST pour les 
    images, il devrait vous donner une compréhension de base des techniques impliquées. Une fois que 
    vous aurez terminé ce tutoriel, vous aurez un modèle qui essaie de classer un clip audio d'une 
    seconde comme "bas", "aller", "gauche", "non", "droit", "stop", "haut " et oui".
    '''
    
    
    '''
    Vous allez rédiger un script pour télécharger une partie de l' ensemble de données 
    Speech Commands . L'ensemble de données original se compose de plus de 105 000 fichiers 
    audio WAV de personnes disant trente mots différents. Ces données ont été collectées par 
    Google et publiées sous licence CC BY.
    Vous utiliserez une partie de l'ensemble de données pour gagner du temps lors du 
    chargement des données. Extrayez le mini_speech_commands.zip et chargez-le à l'aide de 
    l'API tf.data .
    '''
    data_directory = args.data_directory
    size_data_directory = os.path.getsize(data_directory)
    if size_data_directory <= 4096:
        tf.keras.utils.get_file(
            fname=os.path.join(data_directory,'mini_speech_commands.zip'),
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_subdir='.',
            cache_dir=data_directory)
    
    '''
    Vérifiez les statistiques de base sur l'ensemble de données.
    '''
    commands_directory = os.path.join(data_directory,'mini_speech_commands')
    commands = np.array(tf.io.gfile.listdir(str(commands_directory)))
    commands = commands[commands != 'README.md']
    print('Commands:', commands)
    
    '''
    Extrayez les fichiers audio dans une liste et mélangez-la.
    '''
    filenames = tf.io.gfile.glob(str(commands_directory) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    [print(path,len(os.listdir(os.path.join(commands_directory,path)))) for path in os.listdir(commands_directory) if path != 'README.md']
    print('Example file tensor:', filenames[0])

    '''
    Divisez les fichiers en ensembles d'entraînement, de validation et de 
    test en utilisant un ratio de 80:10:10, respectivement.
    '''
    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]  
    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))
    
    '''
    Vous allez maintenant appliquer process_path pour créer votre ensemble 
    d'apprentissage afin d'extraire les paires d'étiquettes audio et vérifier 
    les résultats. Vous construirez les ensembles de validation et de test en utilisant 
    une procédure similaire plus tard.
    '''
    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(AudioPreprocessor().get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    '''
    Examinons quelques formes d'onde audio avec leurs étiquettes correspondantes.
    '''
    DataVisualizator.plot_audio_wave(rows=3, 
                                     cols=3, 
                                     data_set=waveform_ds)
    
    '''
    Spectrogramme
    
    Vous allez convertir la forme d'onde en un spectrogramme, qui montre les changements de 
    fréquence au fil du temps et peut être représenté sous la forme d'une image 2D. 
    Cela peut être fait en appliquant la transformée de Fourier à court terme (STFT) pour 
    convertir l'audio dans le domaine temps-fréquence.
    '''
    for waveform, label in waveform_ds.take(1):    
        label = label.numpy().decode('utf-8')
        spectrogram = AudioPreprocessor().get_spectrogram(waveform)
    
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))
    
    DataVisualizator.plot_waveform_spectrogram(waveform, spectrogram)
    
    '''
    Transformez maintenant l'ensemble de données de forme d'onde pour avoir des 
    images de spectrogramme et leurs étiquettes correspondantes en tant qu'ID entiers.
    '''
    spectrogram_ds = waveform_ds.map(lambda x,y: 
    AudioPreprocessor().get_spectrogram_and_label_id(x, y, commands), num_parallel_calls=AUTOTUNE)

    '''
    Examinez les "images" du spectrogramme pour différents échantillons de l'ensemble de données.
    '''
    DataVisualizator.plot_audio_spectrogram_table(rows=3, 
                                                    cols=3, 
                                                    data_set=spectrogram_ds,
                                                    labels=commands)
    
    
    '''
    Construire et entraîner le modèle
    
    Vous pouvez maintenant créer et entraîner votre modèle. Mais avant cela, vous devrez 
    répéter le prétraitement de l'ensemble d'apprentissage sur les ensembles de validation et de test.
    '''
    train_ds = spectrogram_ds
    val_ds = AudioPreprocessor().preprocess_dataset(val_files, commands)
    test_ds = AudioPreprocessor().preprocess_dataset(test_files, commands)
    
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    '''
    Ajoutez des opérations cache() et prefetch() ensemble de données pour réduire la 
    latence de lecture lors de l'entraînement du modèle.
    '''
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    '''
    Get shape of the spectrogram data and number of laber (commands)
    '''
    for spectrogram, _ in spectrogram_ds.take(1):
      input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)
    
    
    '''
    Create and compile the model
    '''
    model = Models().basic_CNN(spectrogram_ds, input_shape, num_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
                        )
    '''
    Vérifions les courbes de perte d'entraînement et de validation pour voir comment 
    votre modèle s'est amélioré pendant l'entraînement.
    '''
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    
    '''
    Évaluer les performances de l'ensemble de test
    Exécutons le modèle sur l'ensemble de test et vérifions les performances.
    '''
    test_audio = []
    test_labels = []
    
    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())
    
    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    
    '''
    Afficher une matrice de confusion
    Une matrice de confusion est utile pour voir à quel point le modèle s'est 
    bien comporté sur chacune des commandes de l'ensemble de test.
    '''
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    DataVisualizator.confusion_matrix_tensor(cf=confusion_mtx,
                                             group_names=commands)




if __name__ == "__main__":
    
    MODEL_NAME = "speech-recognition-cnn.model"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_model = os.path.join(directory_of_script,"model")
    directory_of_results = os.path.join(directory_of_script,"results")
    directory_of_data = os.path.join(directory_of_script,"DATA","voice_commands")
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