# Deep Learning II – Final Project  
**Authors**: Pauline Zhou & Lucien Perdrix  
**Course**: Deep Learning II – ENSTA  
**Instructor**: Yohan Petetin  
**Date**: March 2025

---

## Project Overview

This project focuses on the implementation and analysis of deep neural network architectures built from Restricted Boltzmann Machines (RBM) and Deep Belief Networks (DBN). The work is divided into two main parts:

1. **Binary AlphaDigits Dataset** – Unsupervised generative modeling and analysis of RBM/DBN performance.  
2. **MNIST Dataset** – Supervised classification with a Deep Neural Network (DNN), with and without unsupervised pretraining.

---

## Objectives

- Implement a full pipeline for RBM, DBN, and DNN from scratch using Python.  
- Tune hyperparameters (learning rate, batch size, epochs, number of hidden neurons).  
- Compare performance in generation (Binary AlphaDigits) and classification (MNIST).  
- Analyze the effect of pretraining vs random initialization on a supervised classification task.

---

## Repository Structure

📁 DL2/
- principal_RBM_alpha_coherent.py      # RBM class for Binary AlphaDigits
- principal_DBN_alpha.py               # DBN built from stacked RBMs
- principal_DNN_MNIST.py               # Supervised DNN with/without pretraining
- explo_paupau.ipynb                   # Notebook for Binary AlphaDigits exploration
- explo_paupau_MNIST_last.ipynb        # Notebook for MNIST exploration
- README.md 

---

## Binary AlphaDigits Analysis

We implemented and tested RBMs and DBNs on the Binary AlphaDigits dataset (36×39 grayscale images, 20×16 pixels each).

Key aspects analyzed:
- **Learning rate**: Best values between 0.1 and 0.3.
- **Epochs**: 300 epochs offer good convergence.
- **Mini-batch size**: 32 gives stable results.
- **Hidden units (`q`)**: Larger `q` improves quality but slows training.
- **DBN depth**: More than 3 stacked RBMs slows convergence significantly.

We found DBNs outperform RBMs in generating distinct characters.

---

## MNIST Classification

We evaluated the impact of pretraining on MNIST classification:

- **Pretrained DNN**: Weights initialized via DBN.
- **Random DNN**: Randomly initialized weights.

**Results**:
- Pretraining improves accuracy, especially with fewer data or deeper architectures.
- Achieved **>98%** accuracy on MNIST with architecture `[784, 500, 500, 10]`.

---

## Hyperparameters Used

| Parameter       | Value                  |
|-----------------|------------------------|
| Learning rate   | 0.1                    |
| Epochs (RBM)    | 100                    |
| Epochs (DNN)    | 200                    |
| Batch size      | 32                     |
| Hidden layers   | 2                      |
| Hidden units    | 500                    |

---

## Results Highlights

- RBM/DBN: Capable of generating coherent characters.
- DNN: High classification accuracy on MNIST.
- Pretraining significantly boosts performance in deeper networks or with limited data.

---

## 🚀 How to Run

```bash
# Train and test RBM/DBN on Binary AlphaDigits
python principal_RBM_alpha_coherent.py
python principal_DBN_alpha.py
Run notebook explo_paupau.ipynb for exploration


# Train and evaluate DNN on MNIST
python principal_DNN_MNIST.py
Run notebook explo_paupau_MNIST_last.ipynb for exploration

