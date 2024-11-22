from egalisation_spectre import save_equalized_image_in_folder
from detourage import save_canny_image_in_folder
from redimensionnage_images import redimensionnage_all
import os

def pre_traitement(src_folder, tmp_folder, dst_folder):
    print("Starting pretraitement")
    for letter in os.listdir(src_folder):

        print(f"Pretraitement: {letter}")

        print(f"{letter} - Equalisation du spectre")
        os.makedirs(tmp_folder+f'equalized\\{letter}\\',exist_ok=True)
        save_equalized_image_in_folder(src_folder+letter+"\\", tmp_folder+f'equalized\\{letter}\\')
        
        print(f"{letter} - Canny")
        os.makedirs(tmp_folder+f'canny\\{letter}\\',exist_ok=True)
        save_canny_image_in_folder(tmp_folder+f'equalized\\{letter}\\', tmp_folder+f'canny\\{letter}\\', low_threshold=160, high_threshold=220)
        
        print(f"{letter} - Redimmensionnement")
        os.makedirs(dst_folder+letter+"\\",exist_ok=True)
        #redimensionnage_all(tmp_folder+f'canny\\{letter}\\', dst_folder+letter+"\\")

pre_traitement(
    #src_folder=r"C:\Users\cohel\OneDrive\Documents\4TC\TIP\ASL-Alphabet\archive\asl_alphabet_train\asl_alphabet_train\\",
    src_folder=r"C:\Users\cohel\OneDrive\Documents\4TC\TIP\ASL-Alphabet\archive\asl_alphabet_test\\",
    tmp_folder=r"C:\Users\cohel\OneDrive\Documents\4TC\TIP\ASL-Alphabet\archive\tmp_pretraitement\\",
    dst_folder=r"C:\Users\cohel\OneDrive\Documents\4TC\TIP\ASL-Alphabet\archive\pretraitement\\"
)