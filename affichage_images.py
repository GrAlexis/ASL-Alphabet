import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

def show_single_image(image_path):
    # Vérifier si le fichier existe
    if not os.path.isfile(image_path):
        print(f"L'image spécifiée '{image_path}' n'existe pas.")
        return
    
    # Appliquer une transformation pour convertir l'image en tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Charger l'image
    image = Image.open(image_path)  # Charger l'image
    image_tensor = transform(image)  # Appliquer la transformation

    # Convertir le tensor en image (de format [C, H, W] vers [H, W, C])
    image = image_tensor.permute(1, 2, 0).numpy()

    # Afficher l'image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')  # Ne pas afficher les axes
    plt.title("Image affichée")
    plt.show()


# Fonction pour afficher les images
def show_images_from_folder(folder_path, nb_images):
    # Appliquer une transformation pour convertir les images en tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Lister les fichiers dans le dossier
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("Aucune image trouvée dans le dossier.")
        return

    # Créer une figure pour afficher les images
    num_images_to_show = min(nb_images, len(image_files))  # Limiter à 10 images au maximum
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(15, 15))
    
    # Si un seul axe est retourné, on le met dans une liste pour que ça soit itérable
    if num_images_to_show == 1:
        axes = [axes]

    for i in range(num_images_to_show):
        image_path = os.path.join(folder_path, image_files[i])
        image = Image.open(image_path)  # Charger l'image
        image_tensor = transform(image)  # Appliquer la transformation
        
        # Convertir le tensor en image (de format [C, H, W] vers [H, W, C])
        image = image_tensor.permute(1, 2, 0).numpy()

        # Afficher l'image sur l'axe correspondant
        axes[i].imshow(image)
        axes[i].axis('off')  # Ne pas afficher les axes
        axes[i].set_title(f"Image {i+1}")
    
    plt.tight_layout()
    plt.show()




#####MAIN######


# Demander à l'utilisateur de saisir le chemin du dossier d'images
path = input("Veuillez entrer le chemin de l'image ou du dossier : ")
show_images_from_folder(path,8)
#show_single_image(path)
