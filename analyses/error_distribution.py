import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_error_distribution(model, testloader, classes, device):
    """
    Calcule la distribution des erreurs par classe et affiche un graphique.
    
    :param model: Le modèle entraîné
    :param testloader: DataLoader pour les données de test
    :param classes: Liste des classes du jeu de données
    :param device: Périphérique (CPU ou GPU)
    """
    model.eval()  # Mettre le modèle en mode évaluation
    class_errors = defaultdict(int)  # Dictionnaire pour stocker le nombre d'erreurs par classe
    
    with torch.no_grad():  # Pas besoin de calculer les gradients pendant l'évaluation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Passer les données dans le modèle
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Enregistrer les erreurs par classe
            for true_label, predicted_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                if true_label != predicted_label:
                    class_errors[true_label] += 1  # Ajouter une erreur pour la classe
    
    # Convertir les erreurs en listes pour l'affichage
    error_classes = list(class_errors.keys())
    error_counts = list(class_errors.values())
    
    # Tracer le graphique des erreurs par classe
    plt.figure(figsize=(10, 6))
    plt.bar(error_classes, error_counts, color='red')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'erreurs')
    plt.title('Distribution des erreurs par classe')
    plt.xticks(error_classes, [classes[i] for i in error_classes], rotation=90)  # Afficher les noms des classes
    plt.show()
    
    # Afficher les classes avec le plus d'erreurs
    sorted_classes = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)
    print("Classes avec le plus d'erreurs :")
    for class_idx, error_count in sorted_classes:
        print(f"{classes[class_idx]} : {error_count} erreurs")
