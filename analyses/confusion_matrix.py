from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()  # Met le modèle en mode évaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Pas besoin de calculer les gradients pour l'évaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Déplacer les données vers le bon périphérique
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())  # Prédictions sur CPU
            all_labels.extend(labels.cpu().numpy())    # Labels sur CPU

    # Calcul de la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)

    # Affichage de la matrice de confusion avec un heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice de confusion')
    plt.show()

