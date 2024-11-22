import os
import shutil
import random

def split_dataset(base_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Divise une base de données d'images en ensembles d'entraînement, de validation et de test.

    Args:
        base_dir (str): Chemin du dossier contenant les images classées par sous-dossier.
        output_dir (str): Chemin du dossier où seront sauvegardés les ensembles.
        train_ratio (float): Proportion des données pour l'entraînement.
        val_ratio (float): Proportion des données pour la validation.
        test_ratio (float): Proportion des données pour le test.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Les proportions doivent totaliser 1."

    # Créer les dossiers de sortie
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for folder in [train_dir, val_dir, test_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Parcourir les sous-dossiers dans la base de données
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        if not os.path.isdir(class_path):
            continue  # Ignore les fichiers qui ne sont pas des dossiers

        # Créer les dossiers par classe dans chaque ensemble
        for folder in [train_dir, val_dir, test_dir]:
            class_output_dir = os.path.join(folder, class_name)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)

        # Liste des images dans la classe
        images = os.listdir(class_path)
        random.shuffle(images)  # Mélanger les images aléatoirement

        # Calcul des indices pour division
        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        # Diviser les images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Copier les images dans les ensembles correspondants
        for image_set, folder in zip([train_images, val_images, test_images], [train_dir, val_dir, test_dir]):
            for image in image_set:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(folder, class_name, image)
                shutil.copy(src_path, dst_path)

        print(f"Classe '{class_name}' : {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("Division terminée ! Les ensembles sont prêts.")

# Exécution
base_dir = input("Entrez le chemin du dossier contenant les données (images classées par sous-dossier) : ")
output_dir = input("Entrez le chemin du dossier de sortie pour les ensembles divisés : ")
split_dataset(base_dir, output_dir)
