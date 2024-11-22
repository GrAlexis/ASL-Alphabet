import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

def redimensionnage_all(src_folder, dst_folder, size=64):
    # Récupération de l'ensemble des images du dossier sources
    images_path = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Lecture des images
    image_list = [Image.open(src_folder + image_path) for image_path in images_path]

    # Création de la transformation à appliquer
    resize_transform = T.Resize(size)

    # Redimmensionnement de l'ensemble des images
    resize_list = [resize_transform(image) for image in image_list]

    # Création du dossier destination s'il n'existe pas
    os.makedirs(dst_folder,exist_ok=True)

    # Sauvegarde de l'ensemble des images dans le dossier destination
    for i in range(len(images_path)):
        resize_list[i].save(dst_folder + images_path[i])