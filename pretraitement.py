import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import filedialog

# Initialiser Tkinter et cacher la fenêtre principale
root = tk.Tk()
root.withdraw()  # Ne pas afficher la fenêtre principale

# Ouvrir la boîte de dialogue pour choisir un dossier
dataset_dir = filedialog.askdirectory(title="Sélectionner le dossier contenant les images")

# Vérifier si un dossier a été sélectionné
if not dataset_dir:
    print("Aucun dossier sélectionné.")
    exit()

# Définir les transformations que vous souhaitez appliquer aux images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir l'image en tenseur
])

# Charger le dataset à partir du dossier
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Créer un DataLoader pour itérer sur le dataset
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fonction pour afficher plusieurs images
def show_all_images(dataloader):
    """ Affiche toutes les images du dataset dans un seul lot """
    for i, (images, labels) in enumerate(dataloader):
        imshow(images)
        if i == len(dataloader) - 1:  # Si c'est le dernier lot
            break  # Arrêter après le dernier lot

# Fonction pour afficher une seule image
def show_one_image(image):
    """ Affiche une seule image donnée en paramètre """
    imshow(image.unsqueeze(0))  # Ajouter une dimension pour correspondre à un lot de taille 1

# Fonction pour afficher les images (utilisé par les deux fonctions précédentes)
def imshow(images):
    """ Affiche les images en utilisant matplotlib """
    images = images.numpy().transpose((0, 2, 3, 1))  # Convertir en format (N, H, W, C)
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(2, 2, i + 1)  # Disposer les images en 2x2
        plt.imshow(images[i])
        plt.axis('off')  # Masquer les axes
    plt.show()

# Exemple d'utilisation :

# Afficher toutes les images du dataset
# show_all_images(dataloader)

# Afficher une seule image (ici, nous utilisons la première image du premier lot)
data_iter = iter(dataloader)
images, labels = next(data_iter)
show_one_image(images[0])  # Afficher la première image du lot
