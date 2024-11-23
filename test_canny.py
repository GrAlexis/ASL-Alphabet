import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

# Fonction pour appliquer Canny à une image avec des seuils donnés
def apply_canny(image_path, low_threshold, high_threshold):
    # Charger l'image
    image = Image.open(image_path)
    image = np.array(image)  # Convertir en array numpy
    
    # Convertir en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Appliquer le filtre Canny
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

# Fonction pour tester différentes combinaisons de seuils sur une image
def test_canny_thresholds(image_path, low_range, high_range, step):
    # Créer une figure pour les résultats
    num_tests = len(low_range) * len(high_range)
    num_cols = 4  # Afficher 4 images par ligne
    num_rows = int(np.ceil(num_tests / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.ravel()  # Aplatir les axes pour un accès plus simple
    
    idx = 0  # Compteur pour le sous-graphe
    for low in low_range:
        for high in high_range:
            if high <= low:
                continue  # Ignorer les combinaisons invalides
            
            # Appliquer Canny
            edges = apply_canny(image_path, low, high)
            
            # Ajouter à la figure
            if idx < len(axes):  # Assurer qu'on reste dans les limites
                axes[idx].imshow(edges, cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f"Low: {low}, High: {high}")
                idx += 1

    # Masquer les axes inutilisés
    for j in range(idx, len(axes)):
        axes[j].axis('off')
    
    return fig

# Fonction principale pour tester toutes les images d'un dossier
def process_folder_and_save_results(src_folder, low_min=80, low_max=150, high_min=100, high_max=220, step=10):
    # Créer le dossier de destination
    dst_folder = os.path.join(src_folder, "test_canny")
    os.makedirs(dst_folder, exist_ok=True)
    
    # Générer les plages de seuils
    low_range = list(range(low_min, low_max + 1, step))
    high_range = list(range(high_min, high_max + 1, step))
    
    # Lister les fichiers images dans le dossier source
    image_files = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(src_folder, image_file)
        
        # Tester les seuils pour cette image et générer une figure
        fig = test_canny_thresholds(image_path, low_range, high_range, step)
        
        # Sauvegarder les résultats dans un PDF
        pdf_path = os.path.join(dst_folder, f"{os.path.splitext(image_file)[0]}_canny_test.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Résultats sauvegardés pour {image_file} dans {pdf_path}")

if __name__ == '__main__':
    # Demander le dossier source
    src_folder = input("Veuillez entrer le chemin du dossier contenant les images : ").strip()
    
    # Lancer le traitement avec les seuils optimisés
    process_folder_and_save_results(src_folder, low_min=80, low_max=140, high_min=100, high_max=220, step=10)
