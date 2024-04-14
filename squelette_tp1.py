import numpy as np
import matplotlib.pyplot as plt
from mltools import plot_data, plot_frontiere, make_grid, gen_arti

# Fonctions pour les coûts et les gradients

def mse(w, x, y):
    return np.mean(np.square(x.dot(w) - y))

def mse_grad(w, x, y):
    return np.mean(2 * (x.dot(w) - y) * x, axis=0)

def reglog(w, x, y):
    return np.mean(np.maximum(0, 1 - y * x.dot(w)))

def reglog_grad(w, x, y):
    indicator = np.where(y * x.dot(w) < 1, -y, 0)
    return np.mean(indicator * x, axis=0)

def hinge(w, x, y):
    margins = 1 - y * x.dot(w)
    return np.mean(np.maximum(0, margins))

def hinge_grad(w, x, y):
    margins = 1 - y * x.dot(w)
    indicator = np.where(margins > 0, -y, 0)
    return np.mean(indicator * x, axis=0)

# Fonction pour le taux de classification

def classification_rate(w, x, y):
    predictions = np.sign(x.dot(w))
    return np.mean(predictions == y)

# Fonction de descente de gradient

def gradient_descent(x_train, y_train, x_test, y_test, cost_func, grad_func, learning_rate=0.01, num_iterations=100):
    w = np.random.randn(x_train.shape[1], 1)
    costs_train = []
    costs_test = []
    classification_rates_train = []
    classification_rates_test = []
    for _ in range(num_iterations):
        gradient = grad_func(w, x_train, y_train)
        w -= learning_rate * gradient.reshape(-1, 1)
        
        cost_train = cost_func(w, x_train, y_train)
        costs_train.append(cost_train)
        cost_test = cost_func(w, x_test, y_test)
        costs_test.append(cost_test)
        
        classification_rate_train = classification_rate(w, x_train, y_train)
        classification_rates_train.append(classification_rate_train)
        classification_rate_test = classification_rate(w, x_test, y_test)
        classification_rates_test.append(classification_rate_test)
        
    return w, costs_train, costs_test, classification_rates_train, classification_rates_test

# Fonction pour afficher les graphiques de coût et de taux de classification

def plot_cost_and_classification_trajectories(costs_train, costs_test, classification_rates_train, classification_rates_test):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(costs_train, label='Train')
    plt.plot(costs_test, label='Test')
    plt.xlabel('Iterations')
    plt.ylabel('Coût')
    plt.title('Évolution du coût lors de la descente de gradient')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(classification_rates_train, label='Train')
    plt.plot(classification_rates_test, label='Test')
    plt.xlabel('Iterations')
    plt.ylabel('Taux de bonne classification')
    plt.title('Évolution du taux de bonne classification')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    np.random.seed(0)
    # Génération des données
    datax_train, datay_train = gen_arti(data_type=0, epsilon=0.1)
    datax_test, datay_test = gen_arti(data_type=0, epsilon=0.1)

    # Ajout de la colonne de biais
    datax_train = np.hstack((np.ones((datax_train.shape[0], 1)), datax_train))
    datax_test = np.hstack((np.ones((datax_test.shape[0], 1)), datax_test))

    # Entraînement de la régression linéaire
    w_mse, costs_mse_train, costs_mse_test, rates_mse_train, rates_mse_test = gradient_descent(
        datax_train, datay_train, datax_test, datay_test, mse, mse_grad)
    
    # Entraînement du coût hinge-loss
    w_hinge, costs_hinge_train, costs_hinge_test, rates_hinge_train, rates_hinge_test = gradient_descent(
        datax_train, datay_train, datax_test, datay_test, hinge, hinge_grad)

    # Entraînement du coût de régression logistique
    w_reglog, costs_reglog_train, costs_reglog_test, rates_reglog_train, rates_reglog_test = gradient_descent(
        datax_train, datay_train, datax_test, datay_test, reglog, reglog_grad)

    # Affichage des résultats pour MSE
    plt.figure(figsize=(18, 18))

    plt.subplot(3, 4, 1)
    plot_frontiere(datax_train[:, 1:], lambda x: np.sign(x.dot(w_mse[1:])), step=100)
    plot_data(datax_train[:, 1:], datay_train)
    plt.title('Régression Linéaire - Entraînement (MSE)')

    plt.subplot(3, 4, 2)
    plot_frontiere(datax_test[:, 1:], lambda x: np.sign(x.dot(w_mse[1:])), step=100)
    plot_data(datax_test[:, 1:], datay_test)
    plt.title('Régression Linéaire - Test (MSE)')
  
    plt.subplot(3, 4, 3)
    plot_cost_and_classification_trajectories(costs_mse_train, costs_mse_test, rates_mse_train, rates_mse_test)
    plt.title('Régression Linéaire - Coût et Classification (MSE)')

    # Affichage des résultats pour Hinge Loss
    plt.figure(figsize=(18, 18))

    plt.subplot(3, 4, 1)
    plot_frontiere(datax_train[:, 1:], lambda x: np.sign(x.dot(w_hinge[1:])), step=100)
    plot_data(datax_train[:, 1:], datay_train)
    plt.title('Hinge Loss - Entraînement')
 
    plt.subplot(3, 4, 2)
    plot_frontiere(datax_test[:, 1:], lambda x: np.sign(x.dot(w_hinge[1:])), step=100)
    plot_data(datax_test[:, 1:], datay_test)
    plt.title('Hinge Loss - Test')
    
    plt.subplot(3, 4, 3)
    plot_cost_and_classification_trajectories(costs_hinge_train, costs_hinge_test, rates_hinge_train, rates_hinge_test)
    plt.title('Hinge Loss - Coût et Classification')

    # Affichage des résultats pour RegLog
    plt.figure(figsize=(18, 18))

    plt.subplot(3, 4, 1)
    plot_frontiere(datax_train[:, 1:], lambda x: np.sign(x.dot(w_reglog[1:])), step=100)
    plot_data(datax_train[:, 1:], datay_train)
    plt.title('Régression Logistique - Entraînement')
   
    plt.subplot(3, 4, 2)
    plot_frontiere(datax_test[:, 1:], lambda x: np.sign(x.dot(w_reglog[1:])), step=100)
    plot_data(datax_test[:, 1:], datay_test)
    plt.title('Régression Logistique - Test')

    plt.subplot(3, 4, 3)
    plot_cost_and_classification_trajectories(costs_reglog_train, costs_reglog_test, rates_reglog_train, rates_reglog_test)
    plt.title('Régression Logistique - Coût et Classification')

    plt.show()
