import pickle
import os
import urllib.request
import gzip
import shutil

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder

from principal_DNN_MNIST import DNN

def lire_alpha_digit(L):
        """
        Charge la base Binary AlphaDigits et retourne les images sous forme matricielle.
        Args:
            L (list): Liste des indices des caractères à récupérer.
            Les données sont au lien suivant: 
            https://www.kaggle.com/datasets/angevalli/binary-alpha-digits?select=binaryalphadigs.mat
        Returns:
            numpy.ndarray: Matrice contenant les images.
        """
        path = 'Data/binaryalphadigs.mat'
        data = sio.loadmat(path)
        X = data['dat'][L[0]]
        for i in range(1, len(L)):
            X_bis = data['dat'][L[i]]
            X = np.concatenate((X, X_bis), axis=0)
        n=X.shape[0]
        m = data['dat'][0][0].shape[0]*data['dat'][0][0].shape[1]
        X=np.concatenate(X).reshape((n,m))
        return X

def lire_mnist(path="Data/"):
    """
    Charge les données de la base MNIST et les binarise.
    Args:
        path (str): Chemin du dossier contenant les fichiers MNIST.
    Returns:
        tuple: Tuple contenant les données d'entraînement et de test binarisées.
    """
    X_train, y_train = loadlocal_mnist(
        images_path=path + "train-images.idx3-ubyte",
        labels_path=path + "train-labels.idx1-ubyte",
    )
    X_test, y_test = loadlocal_mnist(
        images_path=path + "t10k-images.idx3-ubyte",
        labels_path=path + "t10k-labels.idx1-ubyte",
    )

    # Binarisation des données
    X_train = (X_train >= 127).astype(int)
    X_test = (X_test >= 127).astype(int)

    encode = OneHotEncoder()
    y_train = encode.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encode.transform(y_test.reshape(-1, 1)).toarray()

    return X_train, X_test, y_train, y_test

def afficher_alpha_digit_random(data, nb_images=5):
    """
    Affiche un échantillon d'images de la base Binary AlphaDigits.

    Args:
        data (numpy.ndarray): Matrice contenant les images en ligne.
        nb_images (int): Nombre d'images à afficher (par défaut 5).
    """
    nb_images = min(nb_images, data.shape[0])  # Limite au nombre d'images disponibles
    indices = np.random.choice(data.shape[0], nb_images, replace=False)
    figure, axes = plt.subplots(1, nb_images, figsize=(10, 7))
    for i, idx in enumerate(indices):
        axe = axes[i]
        axe.imshow(data[idx].reshape(20, 16), cmap="gray")
        axe.set_xticks([])
        axe.set_yticks([])
        axe.set_title(f"Image {idx}")
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()
    plt.show()
    
def import_model(filename: str) -> DNN:     # Retourner un objet de type DNN
    """
    param filename: chemin du fichier
    return: modèle entraîné
    Cette fonction importe un modèle entraîné à partir d'un fichier spécifié.
    """
    with open("models/"+filename, "rb") as file:
        model = pickle.load(file)
        return model


def save_model(filename: str, model):
    """
    :param filename: chemin du fichier
    :param model: modèle entraîné à sauvegarder
    Cette fonction sauvegarde un modèle entraîné dans un fichier spécifié.
    """
    with open("models/"+filename, "wb") as file:
        pickle.dump(model, file)