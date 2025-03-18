"""Ce fichier implémente une classe DBN pour la méthode Deep Belief Network."""
import numpy as np
import matplotlib.pyplot as plt
import copy

from principal_RBM_alpha_coherent import RBM

class DBN:
    """Deep Belief Network construit comme une liste de RBM."""

    def __init__(self, layer_sizes):
        """Initialisation du DBN avec des RBM successifs.
        Args:
            layer_sizes (list): Liste des tailles des couches [entrée, cachée1, cachée2, ..., sortie].
        """
        self.RBM_layers = []
        self.losses = []  # Stocke l'évolution des erreurs d'entraînement

        for i in range(len(layer_sizes) - 1):
            self.RBM_layers.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    
    def train_DBN(self, nb_epochs, learning_rate, mini_batch_size, X, verbose=False, step=30):       # verbose = True, affiche des messages sur la progression de l'entrainement
        """Entraînement du DBN couche par couche en utilisant train_RBM.
        Args:
            X (numpy.ndarray): Données d'entrée pour l'entraînement.
            learning_rate (float): Taux d'apprentissage.
            batch_size (int): Taille des mini-batchs.
            nb_epochs (int): Nombre d'epochs d'entraînement.
            verbose (bool): Afficher les logs d'entraînement.
            step (int): Fréquence d'enregistrement des images et erreurs.
        """
        X_copy = X.copy()

        if verbose:
            print(f"Entraînement DBN avec {len(self.RBM_layers)} RBMs")

        for i, rbm in enumerate(self.RBM_layers):
            if verbose:
                print(f"Entraînement RBM {i + 1} / {len(self.RBM_layers)}")
            rbm.train_RBM(nb_epochs, learning_rate, mini_batch_size, X_copy, verbose=verbose, step=step)
            X_copy = rbm.entree_sortie_RBM(X_copy)      # passage des données à la couche suivante
            self.losses.append(rbm.losses)              # ajout de la perte de chaque RBM

        return self

    
    def generer_image_DBN(self, nb_images, nb_iter_gibbs):
        """
        Génère des images en échantillonnant via le DBN.
        Args:
            nb_images (int): Nombre d'images à générer.
            nb_iter_gibbs (int): Nombre d'itérations de l'échantillonneur de Gibbs.
        """
        for i in range(nb_images):
            v = self.RBM_layers[-1].generer_image_RBM_without_plot(nb_iter_gibbs)

            for j in reversed(range(len(self.RBM_layers) - 1)):
                v = self.RBM_layers[j].sortie_entree_RBM(v)

            v = v.reshape(20, 16)
            plt.subplot(1, nb_images, i + 1)
            plt.imshow(v, cmap="gray")
            plt.axis("off")
        plt.show()


    def display_loss_DBN(self):
        """
        Affiche l'évolution de la perte du DBN au cours de l'entraînement.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses[-1], label="Perte du DBN")
        plt.xlabel("Épochs")
        plt.ylabel("Erreur de reconstruction")
        plt.title("Évolution de la perte du DBN")
        plt.legend()
        plt.show()