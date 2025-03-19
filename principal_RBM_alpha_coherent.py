"""Ce fichier implémente une classe RBM pour la méthode Restricted Boltzmann Machine."""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import copy

class RBM:
    """Restricted Boltzmann Machine (RBM) class."""

    def __init__(self, p, q):
        self.W = np.random.normal(0, 0.01, (p, q))
        self.b = np.zeros(q)
        self.a = np.zeros(p)

        self.losses = []            # suivi des erreurs d'entrainement
        self.X_list = []            # images originales à différentes epochs
        self.X_rec_list = []        # images reconstruites à différentes epochs
        self.epoch_RBM = []         # numéros d'epoch correspondants aux images sauvegardées
        self.errors_images = []     # erreurs de reconstruction des images sauvegardées

    def entree_sortie_RBM(self, X):
        """
        Calcule les probabilités d'activation des neurones cachés à partir des entrées visibles.
        Args:
            X (numpy.ndarray): Données visibles (entrée du RBM).
        Returns:
            numpy.ndarray: Probabilités d'activation des neurones cachés.
        """
        return 1 / (1 + np.exp(-self.b - X@self.W))
    
    def sortie_entree_RBM(self, H):
        """ 
        Calcule les probabilités de reconstruction des entrées visibles à partir des neurones cachés.
        Args:
            H (numpy.ndarray): Activations cachées.
        Returns:
            numpy.ndarray: Probabilités de reconstruction des entrées visibles.
        """
        return 1 / (1 + np.exp(-self.a - H@self.W.T))
    
    def train_RBM(self, nb_epochs, learning_rate, mini_batch_size, X, verbose=False, step=30):   # verbose = True, affiche des messages sur la progression de l'entrainement
        """
        Entraînement de la RBM par l'algorithme Contrastive-Divergence-1.
        Args:
            X (numpy.ndarray): Données d'entrée pour l'entraînement.
            learning_rate (float): Taux d'apprentissage.
            batch_size (int): Taille des mini-batchs.
            nb_epochs (int): Nombre d'epochs d'entraînement.
            verbose (bool): Afficher les logs d'entraînement.
            step (int): Fréquence d'enregistrement des images reconstruites.
        """
        X_copy = X.copy()
        n = X_copy.shape[0]
        p = self.W.shape[0]
        q = self.W.shape[1]
        self.nb_epochs = nb_epochs                  # recupération de ces valeurs pour l'analyse
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size

        for epoch in range(nb_epochs):
            np.random.shuffle(X_copy)
            for i in range(0, n, mini_batch_size):
                X_batch = X_copy[i:min(i + mini_batch_size, n), :]
                t_batch = X_batch.shape[0]
                v_0 = copy.deepcopy(X_batch)
                p_b_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = (np.random.rand(t_batch, q) < p_b_v_0) * 1
                p_b_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = (np.random.rand(t_batch, p) < p_b_h_0) * 1
                p_b_v_1 = self.entree_sortie_RBM(v_1)

                grad_a = np.sum(v_0 - v_1, axis=0) / t_batch
                grad_b = np.sum(p_b_v_0 - p_b_v_1, axis=0) / t_batch
                grad_W = (v_0.T@p_b_v_0 - v_1.T@p_b_v_1) / t_batch

                self.W += learning_rate * grad_W
                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b

            # Affichage erreur de reconstruction
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            loss = np.mean((X - X_rec)**2)
            self.losses.append(loss)                                    # sauvegarde de l'erreur de reconstruction
            
            # Enregistrement des images tous les 'step' epochs
            if epoch % step == 0 or epoch == nb_epochs - 1:
                random_idx = np.random.randint(0, n)                    # choisir une image différente
                self.X_list.append(X[random_idx])
                self.X_rec_list.append(X_rec[random_idx])
                
                self.epoch_RBM.append(f"Epoch{epoch + 1}/{nb_epochs}")
                
                error_selected = np.mean((X[random_idx] - X_rec[random_idx])**2)
                self.errors_images.append(error_selected)               # sauvegarde de l'erreur de reconstruction de l'image sélectionnée

                if verbose:
                    print(f"Epoch {epoch + 1}/{nb_epochs}, erreur de reconstruction: {loss}")

        return self
    
    def generer_image_RBM(self, nb_images, nb_iter_gibbs):
        """Génération d'images à partir du RBM."""
        p = self.W.shape[0]
        q = self.W.shape[1]
        figure, axes = plt.subplots(nb_images // 5, 5, figsize=(10, 2 * (nb_images // 5)))
        for i in range(nb_images):
            x_new = (np.random.rand(p) < 0.5) * 1   # créer un vecteur binaire aléatoire avec une distribution uniforme de 0 ou 1 
            for j in range(nb_iter_gibbs):
                h = (np.random.rand(q) < self.entree_sortie_RBM(x_new)) * 1
                x_new = (np.random.rand(p) < self.sortie_entree_RBM(h)) * 1
            axe = axes[i // 5, i % 5]
            axe.imshow(x_new.reshape(20, 16), cmap='gray')
            axe.axis("off")
        plt.show()
    
    def generer_image_RBM_without_plot(self, nb_iter_gibbs):
        """Génère une image sans l'afficher (utile pour le DBN)."""
        x_new = (np.random.rand(self.W.shape[0]) < 0.5) * 1
        for _ in range(nb_iter_gibbs):
            h = (np.random.rand(self.W.shape[1]) < self.entree_sortie_RBM(x_new)) * 1
            x_new = (np.random.rand(self.W.shape[0]) < self.sortie_entree_RBM(h)) * 1
        return x_new

    def display_image_RBM_vs_original(self):
        """Affiche les images originales et reconstruites durant l'entraînement."""
        for i in range(len(self.epoch_RBM)):
            print(self.epoch_RBM[i])
            figure, axes = plt.subplots(1, 2, figsize=(5, 3))
            axes[0].imshow(self.X_list[i].reshape(20, 16), cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(f"Original ", fontsize=8)
            
            axes[1].imshow(self.X_rec_list[i].reshape(20, 16), cmap="gray")
            axes[1].axis("off")
            axes[1].set_title(f"Reconstruit (RMSE={self.errors_images[i]:.4f})", fontsize=8)
            
            figure.tight_layout()
            plt.show()

    
    def generate_for_analysis(self, nb_gibbs, nb_image=5, param_analysed="nb_epochs"):
        """Affiche des images générées et une courbe de perte pour différents hyperparamètres."""
        
        p = self.W.shape[0]
        q = self.W.shape[1]
        figure = plt.figure(figsize=(nb_image * 2, 2))

        for i in range(nb_image):
            v = (np.random.rand(p) < 1 / 2) * 1
            for j in range(nb_gibbs):
                h = (np.random.rand(q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(p) < self.sortie_entree_RBM(h)) * 1

            axe = plt.subplot2grid((1, nb_image), (i // nb_image, i % nb_image), fig=figure)
            axe.imshow(np.reshape(v, (20, 16)), cmap="gray")
            axe.axis("off")

        if param_analysed == "nb_epochs": 
            plt.title(f"Reconstruction ({param_analysed} = {self.nb_epochs})", fontsize=7, loc="left")

        elif param_analysed == "learning_rate":
            plt.title(f"Reconstruction ({param_analysed} = {self.learning_rate})", fontsize=7, loc="left")

        elif param_analysed == "mini_batch_size":
            plt.title(f"Reconstruction ({param_analysed} = {self.mini_batch_size})", fontsize=7, loc="left")
        
        elif param_analysed == "q":
            plt.title(f"Reconstruction ({param_analysed} = {self.W.shape[1]})", fontsize=7, loc="left")

        plt.tight_layout()
        plt.show()