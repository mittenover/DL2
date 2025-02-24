"""Ce fichier implémente une classe RBM pour la méthode Restricted Boltzmann Machine."""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import copy

def lire_alpha_digit(L):
        """
        Récupération sous forme matricielle des données de la base Binary AlphaDigits.
        Les données sont au lien suivant: 
        https://www.kaggle.com/datasets/angevalli/binary-alpha-digits?select=binaryalphadigs.mat
        L est l'indice des caractères à récupérer
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

class RBM:
    """Restricted Boltzmann Machine class."""
    def __init__(self, p, q):
        """Inititialisation de la classe RBM."""
        self.W = np.random.normal(0, 0.01, (p, q))
        self.b = np.zeros(q)
        self.a = np.zeros(p)

    def entree_sortie_RBM(self, X):
        """Calcul de la sortie de la couche visible."""
        # print(self.b.shape)
        # print(X.shape)
        # print(self.W.shape)
        return 1/(1+np.exp(-self.b - X@self.W))
    
    def sortie_entree_RBM(self, H):
        """Calcul de la sortie de la couche cachée."""
        return 1/(1+np.exp(-self.a - H@self.W.T))
    
    def train_RBM(self, nb_epochs, learning_rate, mini_batch_size, X):
        """Entraînement de la RBM."""
        n = X.shape[0]
        p = self.W.shape[0]
        q = self.W.shape[1]
        for epoch in range(nb_epochs):
            np.random.shuffle(X)
            for i in range(0, n, mini_batch_size):
                X_batch = X[i:min(i+mini_batch_size,n),:]
                t_batch = X_batch.shape[0]
                v_0 = copy.deepcopy(X_batch)
                p_b_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = (np.random.rand(t_batch,q) < p_b_v_0)*1
                p_b_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = (np.random.rand(t_batch,p) < p_b_h_0)*1
                p_b_v_1 = self.entree_sortie_RBM(v_1)
                grad_a = np.sum(v_0 - v_1, axis=0)/t_batch
                grad_b = np.sum(p_b_v_0 - p_b_v_1, axis=0)/t_batch
                grad_W = (v_0.T@p_b_v_0 - v_1.T@p_b_v_1)/t_batch
                self.W += learning_rate*grad_W
                self.a += learning_rate*grad_a
                self.b += learning_rate*grad_b
            # Affichage erreur de reconstruction
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            print(f"Epoch {epoch+1}/{nb_epochs}, erreur de reconstruction: {np.mean((X - X_rec)**2)}")
        return self
    
    def generer_image_RBM(self, nb_images, nb_iter_gibbs):
        p = self.W.shape[0]
        q = self.W.shape[1]
        for i in range(nb_images):
            x_new = (np.random.rand(p) < np.random.rand(p))*1
            for j in range(nb_iter_gibbs):
                h = (np.random.rand(q) < self.entree_sortie_RBM(x_new))*1
                x_new = (np.random.rand(p) < self.sortie_entree_RBM(h))*1
            plt.imshow(x_new.reshape(20,16), cmap='gray')
            plt.show()
        return self
    