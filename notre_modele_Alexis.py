import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F  # Importation de torch.nn.functional
import time  # Importation du module time pour mesurer la durée

# Vérification de la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement sur le périphérique : {device}")

# Transformation des images (redimensionnement et normalisation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionnement des images
    transforms.ToTensor(),          # Conversion en tenseur
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation
])

# Chemins des dossiers d'entraînement et de validation
train_dir = input("Entrez le chemin du dossier d'images d'entraînement : ")
val_dir = input("Entrez le chemin du dossier d'images de validation : ")

# Chargement des données avec ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64, shuffle=False)

# Définition du modèle
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Bloc 1 : Convolution + ReLU + BatchNorm + MaxPooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 2 : Convolution + ReLU + BatchNorm + MaxPooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 3 : Convolution + ReLU + BatchNorm + MaxPooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 4 : Convolution + ReLU + BatchNorm
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Couches entièrement connectées
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialiser le modèle avec le nombre de classes
model = CustomCNN(num_classes=len(train_data.classes))
model.to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listes pour stocker les pertes et précisions
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

# Entraînement du modèle
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()  # Enregistrement du début de l'époque
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Entraînement
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct / total
    train_losses.append(running_loss / len(trainloader))
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    # Évaluation sur l'ensemble de validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss / len(valloader))
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    end_time = time.time()  # Enregistrement de la fin de l'époque
    epoch_duration = end_time - start_time  # Calcul du temps d'une époque
    print(f"Temps d'une époque: {epoch_duration:.2f} secondes")

# Sauvegarde du modèle
torch.save(model.state_dict(), "custom_cnn.pth")
print("Modèle sauvegardé sous 'custom_cnn.pth'")

# Visualisation des courbes d'entraînement
plt.figure(figsize=(12, 5))

# Graphique de la perte
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')

# Graphique de l'exactitude
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs Epochs')

plt.tight_layout()
plt.show()

# Visualisation de quelques prédictions
def imshow(img):
    img = img / 2 + 0.5  # Dé-normalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(valloader)
images, labels = next(dataiter)

# Affichage des images
imshow(torchvision.utils.make_grid(images[:4]))
print('GroundTruth:', ' '.join(f'{val_data.classes[labels[j]]}' for j in range(4)))

# Prédictions
outputs = model(images[:4].to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted:', ' '.join(f'{val_data.classes[predicted[j]]}' for j in range(4)))