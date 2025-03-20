"""Ce fichier implémente une classe DNN pour la méthode Deep Neural Network."""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import copy
from principal_DBN_alpha import DBN

class DNN(DBN):
    def __init__(self, layer_sizes):
        super().__init__(layer_sizes)
        self.classif_layer = None

    def pretrain_DBN(self, nb_epochs, learning_rate, mini_batch_size, X, verbose=False, step=30):
        """Pré-entraînement du DNN couche par couche en utilisant train_DBN.
        Args:
            X (numpy.ndarray): Données d'entrée pour l'entraînement.
            learning_rate (float): Taux d'apprentissage.
            batch_size (int): Taille des mini-batchs.
            nb_epochs (int): Nombre d'epochs d'entraînement.
            verbose (bool): Afficher les logs d'entraînement.
            step (int): Fréquence d'enregistrement des images et erreurs.
        """
        return super().train_DBN(nb_epochs, learning_rate, mini_batch_size, X, verbose=verbose, step=step)
    
    def calcul_softmax(self, X, layer):
        """Calcul de la fonction softmax pour la couche de classification"""
        z = X@layer.W + layer.b
        e_z = np.exp(z)
        return(e_z/np.sum(e_z, axis=1).reshape(-1,1))
    
    def entree_sortie_reseau(self, X):
        X_copy = X.copy()
        outputs = [X_copy]

        for i, rbm in enumerate(self.RBM_layers[:-1]):
            X_copy = rbm.entree_sortie_RBM(X_copy)
            outputs.append(X_copy)

        outputs.append(self.calcul_softmax(X_copy, self.classif_layer[-1]))

        return outputs
    
    def cross_entropy(self, y_true, y_pred):
        loss = []
        for k in range(y_true.shape[0]):
            loss.append(-np.sum([y_true[k,j] * np.log(y_pred[k,j]) for j in range(y_true.shape[1])]))
        return loss
    
    def retropropagation(self, X, y, nb_epochs, learning_rate, mini_batch_size, verbose=False):

        if verbose:
            print(f"Entraînement DBN avec {len(self.RBM_layers)} RBMs")

        for i in range(nb_epochs):
            loss = []
            X_copy = X.copy()
            y_copy = y.copy()
            for batch in range(0, X.shape[0], mini_batch_size):
                X_batch = X_copy[batch:batch+mini_batch_size]
                y_batch = y_copy[batch:batch+mini_batch_size]
                batch_size = X_batch.shape[0]
                # Forward pass
                outputs = self.entree_sortie_reseau(X_batch)
                outputs_diff = outputs[-1] - y_batch
                # Backward pass
                self.RBM_layers[-1].W -= learning_rate * outputs[-2].T @ outputs_diff / batch_size
                self.RBM_layers[-1].b -= learning_rate * np.sum(outputs_diff, axis=0) / batch_size

                for i in range(len(self.RBM_layers)-2, -1, -1):
                    outputs_diff = (outputs_diff @ self.RBM_layers[i+1].W.T) * outputs[i+1] * (1 - outputs[i+1])
                    self.RBM_layers[i].W -= learning_rate * outputs[i].T @ outputs_diff / batch_size
                    self.RBM_layers[i].b -= learning_rate * np.sum(outputs_diff, axis=0) / batch_size
                
                loss+=self.cross_entropy(y_batch, outputs[-1])
        
            self.losses.append(np.mean(loss))
            print(f"Epoch {i+1}/{nb_epochs}, loss: {np.mean(loss)}")
        return self
    
    def test_DNN(self, X, y):
        outputs = self.entree_sortie_reseau(X)
        y_pred = np.argmax(outputs[-1], axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        print(f"Accuracy: {accuracy}")
        return accuracy





        
        
