import matplotlib.pyplot as plt

def plot_training_metrics(losses, accuracies):
    """
    Affiche un graphique des pertes et de la précision pendant l'entraînement.
    
    :param losses: Liste des pertes par époque.
    :param accuracies: Liste des précisions par époque.
    """
    epochs = range(1, len(losses) + 1)

    # Créer un graphique pour la perte
    plt.figure(figsize=(12, 5))
    
    # Subplot pour la perte
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.title('Pertes pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Subplot pour la précision
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.title('Précision pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Vous pouvez maintenant importer cette fonction dans `réseau.py` ou dans un autre fichier pour l'utiliser.
