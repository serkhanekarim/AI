#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from modules.modeling.image_detection import ImageModels
from modules.modeling.speech_recognition import SpeechModels
from modules.preprocessing.audio import AudioPreprocessor
from modules.visualization.visualization import DataVisualizator

    
def main(args):
    '''
    Classification de base : classer les images de vêtements
    Ce guide forme un modèle de réseau neuronal pour classer des images de vêtements, 
    comme des baskets et des chemises. Ce n'est pas grave si vous ne comprenez pas tous 
    les détails ; il s'agit d'un aperçu rapide d'un programme TensorFlow complet avec les 
    détails expliqués au fur et à mesure.
    Ce guide utilise tf.keras , une API de haut niveau pour créer et entraîner des modèles 
    dans TensorFlow.
    '''  
    
    '''
    Importer le jeu de données Fashion MNIST
    Ce guide utilise le jeu de données Fashion MNIST qui contient 70 000 images en niveaux 
    de gris dans 10 catégories. Les images montrent des vêtements individuels en basse 
    résolution (28 par 28 pixels), comme on le voit ici : 
        
    Fashion MNIST est destiné à remplacer l'ensemble de données MNIST classique, souvent 
    utilisé comme "Hello, World" des programmes d'apprentissage automatique pour la vision 
    par ordinateur. L'ensemble de données MNIST contient des images de chiffres manuscrits 
    (0, 1, 2, etc.) dans un format identique à celui des vêtements que vous utiliserez ici.
    
    Ce guide utilise Fashion MNIST pour la variété, et parce que c'est un problème légèrement 
    plus difficile que le MNIST ordinaire. Les deux ensembles de données sont relativement petits 
    et sont utilisés pour vérifier qu'un algorithme fonctionne comme prévu. Ce sont de bons points 
    de départ pour tester et déboguer du code.
    
    Ici, 60 000 images sont utilisées pour former le réseau et 10 000 images pour évaluer 
    avec quelle précision le réseau a appris à classer les images. Vous pouvez accéder au 
    Fashion MNIST directement depuis TensorFlow. Importez et chargez les données Fashion 
    MNIST directement depuis TensorFlow :
    '''    
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    '''    
    Le chargement de l'ensemble de données renvoie quatre tableaux NumPy :
    
       Les tableaux train_images et train_labels sont l' ensemble d'apprentissage — les données 
       que le modèle utilise pour apprendre.
       Le modèle est testé contre l'ensemble de test, les test_images et test_labels tableaux.
    
    Les images sont des tableaux NumPy 28x28, avec des valeurs de pixel allant de 0 à 255. 
    Les étiquettes sont un tableau d'entiers, allant de 0 à 9. Celles-ci correspondent à la classe 
    de vêtements que l'image représente :
        
    Étiqueter	Classer
    0	T-shirt/haut
    1	Pantalon
    2	Arrêtez-vous
    3	Robe
    4	Manteau
    5	Sandale
    6	La chemise
    7	Baskets
    8	Sac
    9	Bottine
    
    Chaque image est mappée sur une seule étiquette. Étant donné que les noms de classe ne sont pas inclus dans l'ensemble de données, stockez-les ici pour les utiliser ultérieurement lors du traçage des images :
    '''
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    '''
    Prétraiter les données
    Les données doivent être prétraitées avant d'entraîner le réseau.
    Si vous inspectez la première image de l'ensemble d'apprentissage, 
    vous verrez que les valeurs de pixels sont comprises entre 0 et 255 :
    '''
    DataVisualizator.display_image(train_images[0])
    
    '''
    Mettez ces valeurs à l'échelle sur une plage de 0 à 1 avant de les transmettre au 
    modèle de réseau de neurones. Pour ce faire, divisez les valeurs par 255. 
    Il est important que l' ensemble d'apprentissage et l' ensemble de test soient 
    prétraités de la même manière :
    '''

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    '''
    Pour vérifier que les données sont au format correct et que vous êtes prêt à 
    créer et à entraîner le réseau, affichons les 25 premières images de l' ensemble d'entraînement et 
    affichons le nom de la classe sous chaque image.
    '''
    DataVisualizator.display_image_table(data_set=train_images, 
                                         num_image=25, 
                                         labels=class_names, 
                                         index_labels=train_labels)
    
    train_images = tf.expand_dims(train_images, -1)
    test_images = tf.expand_dims(test_images, -1)
    print(tf.convert_to_tensor(train_images))
    print(tf.convert_to_tensor(train_images).shape)
    model = ImageModels().basic_CNN(input_shape=train_images[0].shape,
                                    num_labels=len(class_names))
    #model = ImageModels().basic_NN(input_shape=train_images[0].shape,
    #                        num_labels=len(class_names))
    
    '''
    Compiler le modèle

    Avant que le modèle ne soit prêt pour l'entraînement, il a besoin de quelques 
    réglages supplémentaires. Ceux-ci sont ajoutés lors de l'étape de compilation du modèle :

    Fonction de perte : mesure la précision du modèle pendant l'entraînement. Vous voulez 
    minimiser cette fonction pour "orienter" le modèle dans la bonne direction.
    Optimiseur : c'est ainsi que le modèle est mis à jour en fonction des données qu'il 
    voit et de sa fonction de perte.
    Métriques : utilisées pour surveiller les étapes de formation et de test. L'exemple suivant 
    utilise precision , la fraction des images qui sont correctement classées.
    '''
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    '''
    Former le modèle

    La formation du modèle de réseau de neurones nécessite les étapes suivantes :

    Fournissez les données d'entraînement au modèle. Dans cet exemple, les données d'apprentissage 
    se train_images dans les tableaux train_images et train_labels .
    Le modèle apprend à associer des images et des étiquettes.
    Vous demandez au modèle de faire des prédictions sur un ensemble de tests, dans cet exemple, 
    le tableau test_images .
    Vérifiez que les prédictions correspondent aux étiquettes du tableau test_labels .

    Nourrir le modèle
    Pour commencer l'entraînement, appelez la méthode model.fit , ainsi appelée parce 
    qu'elle « ajuste » le modèle aux données d'entraînement :
    '''
    history = model.fit(train_images, train_labels, epochs=5)
    
    '''
    Vérifions les courbes de perte d'entraînement et de validation pour voir comment 
    votre modèle s'est amélioré pendant l'entraînement.
    '''
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
    plt.legend(['loss', 'accuracy'])
    plt.show()
    
    '''
    Évaluer la précision
    Ensuite, comparez les performances du modèle sur l'ensemble de données de test :
    '''
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    '''
    Afficher une matrice de confusion
    Une matrice de confusion est utile pour voir à quel point le modèle s'est 
    bien comporté sur chacune des commandes de l'ensemble de test.
    '''
    y_pred = np.argmax(model.predict(test_images), axis=1)
    y_true = test_labels
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    DataVisualizator.confusion_matrix_tensor(cf=confusion_mtx,
                                             group_names=class_names)



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