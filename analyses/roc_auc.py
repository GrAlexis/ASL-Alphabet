import torch  # Assurez-vous d'importer torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(model, testloader, num_classes, device):
    """
    Trace la courbe ROC et calcule l'AUC pour chaque classe dans un problème multi-classes.

    :param model: Le modèle pré-entrainé.
    :param testloader: DataLoader pour les données de test.
    :param num_classes: Le nombre de classes dans le problème de classification.
    :param device: Le périphérique sur lequel le modèle et les données doivent être chargés (CPU ou GPU).
    """
    model.eval()  # Mettre le modèle en mode évaluation
    
    # Initialisation des variables pour les vraies étiquettes et les scores
    all_labels = []
    all_probs = []

    # Itérer sur les données de test
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Déplacer vers le device (GPU/CPU)
            
            # Obtenir les probabilités de prédiction
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Convertir les sorties en probabilités
            all_labels.append(labels.cpu().numpy())  # Ajouter les étiquettes vraies
            all_probs.append(probs.cpu().numpy())  # Ajouter les probabilités prédites
    
    # Convertir les listes en arrays
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Binariser les étiquettes (One-hot encoding)
    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    
    # Calculer la courbe ROC et AUC pour chaque classe
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Tracer les courbes ROC pour chaque classe
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    # Tracer la diagonale (aucune performance)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc
