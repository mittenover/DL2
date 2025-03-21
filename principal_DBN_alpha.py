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
        self.losses = []        # Stocke l'évolution des erreurs d'entraînement        
        
        for i in range(len(layer_sizes) - 1):
            self.RBM_layers.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    
    def train_DBN(self, nb_epochs, learning_rate, mini_batch_size, X, verbose=False, step=30):       # verbose = True, affiche des messages sur la progression de l'entrainement
        """Entraînement du DBN couche par couche en utilisant train_RBM."""
        X_copy = X.copy()
        self.nb_epochs = nb_epochs                  # recupération de ces valeurs pour l'analyse
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size

        if verbose:
            print(f"Entraînement DBN avec {len(self.RBM_layers)} RBMs")

        for i, RBM_model in enumerate(self.RBM_layers):
            if verbose:
                print(f"Entraînement RBM {i + 1} / {len(self.RBM_layers)}")
            RBM_model.train_RBM(nb_epochs, learning_rate, mini_batch_size, X_copy, verbose=verbose, step=step)
            X_copy = RBM_model.entree_sortie_RBM(X_copy)      # passage des données à la couche suivante
            self.losses.append(RBM_model.losses)              # ajout de les pertes de chaque RBM

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


    def afficher_loss_DBN(self):
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

    def analyse_DBN(self, nb_gibbs, nb_image=5, param_analysed="nb_epochs"):
        """Affiche des images générées et une courbe de perte pour différents hyperparamètres."""

        figure = plt.figure(figsize=(nb_image * 2, 2))

        for i in range(nb_image):
            v = self.RBM_layers[-1].generer_image_RBM_without_plot(nb_gibbs)        # génère pour la dernière couche du DBN
            for j in reversed(range(len(self.RBM_layers) - 1)):
                v = self.RBM_layers[j].sortie_entree_RBM(v)                         # passe au couche précédente

            axe = plt.subplot2grid((1, nb_image), (i // nb_image, i % nb_image), fig=figure)
            axe.imshow(np.reshape(v, (20, 16)), cmap="gray")
            axe.axis("off")

        if param_analysed == "nb_epochs": 
            plt.title(f"Reconstruction DBN ({param_analysed} = {self.nb_epochs})", fontsize=7, loc="left")

        elif param_analysed == "learning_rate":
            plt.title(f"Reconstruction DBN ({param_analysed} = {self.learning_rate})", fontsize=7, loc="left")

        elif param_analysed == "mini_batch_size":
            plt.title(f"Reconstruction DBN ({param_analysed} = {self.mini_batch_size})", fontsize=7, loc="left")
        
        elif param_analysed == "layer_sizes":
            plt.title(f"Reconstruction DBN ({param_analysed} = {len(self.RBM_layers) + 1})", fontsize=7, loc="left")

        plt.tight_layout()
        plt.show()