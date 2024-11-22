import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Fonction pour appliquer le filtre Canny et isoler l'objet (la main)
def apply_canny_and_isolate_hand(image_path, low_threshold=70, high_threshold=130, gray=True):
    # Charger l'image avec PIL et la convertir en niveaux de gris avec OpenCV
    image = Image.open(image_path)
    image = np.array(image)  # Convertir l'image PIL en array numpy pour l'utiliser avec OpenCV
    
    if gray :
        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else :
        gray_image = image
    
    # Appliquer le filtre Canny pour détecter les contours
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    # Convertir le résultat en image RGB pour l'affichage
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return edges_rgb

# Fonction pour appliquer le Canny à un ensemble d'images et les afficher en grille
def show_images_from_folder_with_canny(folder_path, nb_images, low_threshold=100, high_threshold=200):
    # Lister les fichiers dans le dossier
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("Aucune image trouvée dans le dossier.")
        return

    # Limiter au nombre d'images demandé
    num_images_to_show = min(nb_images, len(image_files))
    
    # Définir la disposition de la grille (par exemple, 4 lignes x 7 colonnes pour 28 images)
    num_rows = int(np.ceil(num_images_to_show / 7))  # Adapté pour 7 images par ligne
    num_cols = 7  # Nombre d'images par ligne
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.ravel()  # Applatir la grille pour un accès plus simple

    for i in range(num_images_to_show):
        image_path = os.path.join(folder_path, image_files[i])

        # Appliquer Canny et obtenir les contours
        edges_rgb = apply_canny_and_isolate_hand(image_path, low_threshold, high_threshold)
        
        # Afficher l'image des contours
        axes[i].imshow(edges_rgb)
        axes[i].axis('off')
    
    # Masquer les sous-graphes inutilisés
    for j in range(num_images_to_show, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def save_canny_image_in_folder(src_folder, dst_folder, low_threshold, high_threshold):
    # Récupération de l'ensemble des images du dossier sources
    images_path = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    canny_list = [Image.fromarray(apply_canny_and_isolate_hand(src_folder + image_path,  low_threshold=low_threshold, high_threshold=high_threshold, gray=False)) for image_path in images_path]

    for i in range(len(images_path)):
        canny_list[i].save(dst_folder + images_path[i])

if __name__ == '__main__':
    # Exemple d'utilisation
    folder_path = input("Veuillez entrer le chemin du dossier contenant les images: ")
    show_images_from_folder_with_canny(folder_path, nb_images=len(os.listdir(folder_path)))

