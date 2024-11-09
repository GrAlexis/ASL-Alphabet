import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Paramètres
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Vérification de la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Entraînement sur le périphérique : {device}")

# Transformation des images (redimensionnement et normalisation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionnement des images
    transforms.ToTensor(),          # Conversion en tenseur
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation
])

# Demander à l'utilisateur le chemin du dossier pour l'entraînement et le test
train_dir = input("Entrez le chemin du dossier d'images d'entraînement : ")
test_dir = input("Entrez le chemin du dossier d'images de test : ")

# Chargement des données avec ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Affichage du nombre d'images et de classes
print(f"Nombre d'images d'entraînement : {len(train_data)}")
print(f"Nombre d'images de test : {len(test_data)}")
print(f"Nombre de classes : {len(train_data.classes)}")

# Modèle pré-entrainé (ResNet18) pour la classification
class ASLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ASLClassifier, self).__init__()
        # Charger le modèle pré-entrainé
        self.model = models.resnet18(pretrained=True)
        
        # Remplacer la dernière couche pour s'adapter au nombre de classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Initialiser le modèle avec le nombre de classes
model = ASLClassifier(num_classes=len(train_data.classes))

# Déplacer le modèle vers le GPU (ou le CPU si le GPU n'est pas disponible)
model.to(device)

# Fonction de perte (cross-entropy pour classification)
criterion = nn.CrossEntropyLoss()

# Optimiseur (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entraînement du modèle
for epoch in range(num_epochs):
    model.train()  # Mettre le modèle en mode entraînement
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Déplacer les données vers le GPU ou le CPU
        
        # Initialiser les gradients
        optimizer.zero_grad()
        
        # Passer les données dans le modèle
        outputs = model(inputs)
        
        # Calcul de la perte
        loss = criterion(outputs, labels)
        
        # Rétropropagation
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 99:  # Affichage de la perte toutes les 100 itérations
            print(f"[Epoch {epoch + 1}, Step {i + 1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    
    # Afficher l'accuracy pour chaque époque
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}%")

# Sauvegarder le modèle après l'entraînement
torch.save(model.state_dict(), 'asl_model.pth')

# Évaluation du modèle
model.eval()  # Mettre le modèle en mode évaluation
correct = 0
total = 0

with torch.no_grad():  # Pas besoin de calculer les gradients pendant l'évaluation
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Déplacer les données vers le GPU ou le CPU
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy sur les données de test : {accuracy:.2f}%")

# Afficher quelques images et leurs prédictions
def imshow(img):
    img = img / 2 + 0.5  # Désnormalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Prendre quelques images de test
dataiter = iter(testloader)
images, labels = next(dataiter)  # Utilisation de 'next()' sur l'itérateur

# Afficher 4 images de test
imshow(torchvision.utils.make_grid(images[:4]))
print('GroundTruth: ', ' '.join(f'{test_data.classes[labels[j]]}' for j in range(4)))

# Prédictions sur les images affichées
outputs = model(images[:4].to(device))  # Déplacer les images vers le GPU ou le CPU
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{test_data.classes[predicted[j]]}' for j in range(4)))
