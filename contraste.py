import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
from torchvision import transforms

# Fonction pour améliorer le contraste d'une image
def enhance_contrast(image_path, factor=3.0):
    # Charger l'image
    image = Image.open(image_path)

    # Appliquer l'amélioration du contraste
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)  # factor > 1 pour augmenter le contraste

    # Afficher l'image originale et l'image avec contraste amélioré
    plt.figure(figsize=(10, 10))

    # Afficher l'image originale
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image originale")

    # Afficher l'image avec contraste amélioré
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image)
    plt.axis('off')
    plt.title(f"Contraste amélioré (facteur={factor})")

    plt.show()

    return enhanced_image

# Fonction pour améliorer le contraste de toutes les images dans un dossier
def enhance_contrast_in_folder(folder_path, nb_images, factor=3.0):
    # Lister les fichiers dans le dossier
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("Aucune image trouvée dans le dossier.")
        return

    # Créer une figure pour afficher les images
    num_images_to_show = min(nb_images, len(image_files))  # Limiter à nb_images images
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(15, 15))
    
    # Si un seul axe est retourné, on le met dans une liste pour que ça soit itérable
    if num_images_to_show == 1:
        axes = [axes]

    for i in range(num_images_to_show):
        image_path = os.path.join(folder_path, image_files[i])

        # Améliorer le contraste de l'image
        enhanced_image = enhance_contrast(image_path, factor)

        # Afficher l'image avec contraste amélioré sur l'axe correspondant
        axes[i].imshow(enhanced_image)
        axes[i].axis('off')  # Ne pas afficher les axes
        axes[i].set_title(f"Image {i+1} - Contraste amélioré")
    
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation : Améliorer le contraste d'une image spécifique
image_path = input("Veuillez entrer le chemin de l'image à améliorer : ")
enhance_contrast(image_path, factor=3.0)

# Exemple d'utilisation : Améliorer le contraste des images dans un dossier
#folder_path = input("Veuillez entrer le chemin du dossier contenant les images : ")
#enhance_contrast_in_folder(folder_path, nb_images=5, factor=3.0)
