import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2  # OpenCV pour l'égalisation d'histogramme

# Fonction pour appliquer l'égalisation d'histogramme sur une image
def histogram_equalization(image_path):
    # Charger l'image en niveau de gris
    image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    image = np.array(image)

    # Afficher l'image originale et son histogramme
    plt.figure(figsize=(12, 6))

    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    # Afficher l'histogramme de l'image originale
    plt.subplot(1, 2, 2)
    plt.hist(image.flatten(), bins=256, range=(0, 256), color='black')
    plt.title("Histogramme de l'image originale")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

    # Appliquer l'égalisation d'histogramme avec OpenCV
    image_equalized = cv2.equalizeHist(image)

    # Afficher l'image égalisée et son histogramme
    plt.figure(figsize=(12, 6))

    # Image égalisée
    plt.subplot(1, 2, 1)
    plt.imshow(image_equalized, cmap='gray')
    plt.title("Image après égalisation d'histogramme")
    plt.axis('off')

    # Afficher l'histogramme de l'image égalisée
    plt.subplot(1, 2, 2)
    plt.hist(image_equalized.flatten(), bins=256, range=(0, 256), color='black')
    plt.title("Histogramme après égalisation")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

    return image_equalized

# Fonction pour appliquer l'égalisation d'histogramme sur un dossier d'images
def histogram_equalization_in_folder(folder_path, nb_images):
    # Lister les fichiers dans le dossier
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("Aucune image trouvée dans le dossier.")
        return

    # Limiter le nombre d'images à afficher
    num_images_to_show = min(nb_images, len(image_files))  # Limiter à nb_images images
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(15, 15))

    # Si un seul axe est retourné, on le met dans une liste pour que ça soit itérable
    if num_images_to_show == 1:
        axes = [axes]

    for i in range(num_images_to_show):
        image_path = os.path.join(folder_path, image_files[i])

        # Appliquer l'égalisation d'histogramme
        histogram_equalization(image_path)

# Exemple d'utilisation : Appliquer l'égalisation d'histogramme à une image spécifique
image_path = input("Veuillez entrer le chemin de l'image à traiter : ")
histogram_equalization(image_path)

# Exemple d'utilisation : Appliquer l'égalisation d'histogramme sur un dossier d'images
#folder_path = input("Veuillez entrer le chemin du dossier contenant les images : ")
#histogram_equalization_in_folder(folder_path, nb_images=5)
