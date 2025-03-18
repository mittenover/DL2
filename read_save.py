import pickle

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

def afficher_alpha_digit_random(data, nb_images=5):
    """
    Affiche un échantillon d'images de la base Binary AlphaDigits.

    Args:
        data (numpy.ndarray): Matrice contenant les images en ligne.
        nb_images (int): Nombre d'images à afficher (par défaut 5).
    """
    nb_images = min(nb_images, data.shape[0])  # Limite au nombre d'images disponibles
    indices = np.random.choice(data.shape[0], nb_images, replace=False)
    plt.figure(figsize=(nb_images * 2, 2))
    
    for i, idx in enumerate(indices):
        plt.subplot(1, nb_images, i + 1)
        plt.imshow(data[idx].reshape(20, 16), cmap="gray")
        plt.axis("off")
        plt.title(f"Image {idx + 1}")
    
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