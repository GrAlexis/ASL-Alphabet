import matplotlib.pyplot as plt

def plot_overfitting(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Trace les courbes de perte et d'accuracy pour l'entraînement et le test afin de détecter le surapprentissage.

    :param train_losses: Liste des pertes sur l'ensemble d'entraînement
    :param test_losses: Liste des pertes sur l'ensemble de test
    :param train_accuracies: Liste des précisions sur l'ensemble d'entraînement
    :param test_accuracies: Liste des précisions sur l'ensemble de test
    """
    epochs = range(1, len(train_losses) + 1)

    # Tracer la perte (loss)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')
    plt.title('Perte (Loss) pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()

    # Tracer la précision (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='red')
    plt.title('Précision (Accuracy) pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
