import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Definition du modele de base
class BreastCancerNet(nn.Module):
    """
    Reseau de neurones simple pour la classification binaire.
    Isolé dans une classe pour pouvoir le réutiliser dans les clients et server central.
    """
    def __init__(self, input_dim: int = 30):
        super(BreastCancerNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.layer_out = nn.Linear(16, 1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer_out(x)
        return x


#Fonction pour charger et prétraiter les données
def get_dataloaders(batch_size: int = 32):
    """
    Charge et prétraite les données du cancer du sein, le normalise et prépare les DataLoaders pour l'entraînement et le test.
    """
    # Chargement des données via Scikit-learn
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Séparation Entrainement (80%) et Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation (centre autour de 0 et on réduit la variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ajout d'une dimension pour la sortie
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Création des DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#Fonction pour entraîner le modèle et évaluer les performances
def train_and_evaluate():
    #paramètres d'entraînement
    EPOCHS = 20
    LEARNING_RATE = 0.001

    # Préparation des DataLoaders
    train_loader, test_loader = get_dataloaders()
    model = BreastCancerNet()

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.BCEWithLogitsLoss()  # Utilisation de BCEWithLogitsLoss pour la classification binaire
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print ("Début de l'entraînement centralisé du modèle de base...")

    # Boucle d'entraînement
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()               # 1. Remise à zéro des gradients
            outputs = model(batch_X)            # 2. Prédiction (Forward pass)
            loss = criterion(outputs, batch_y)  # 3. Calcul de l'erreur
            loss.backward()                     # 4. Rétropropagation (Backward pass)
            optimizer.step()                    # 5. Mise à jour des poids (Apprentissage)
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")


    # Évaluation du modèle sur l'ensemble de test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            # On passe les logits dans une sigmoïde, puis on arrondit (0 ou 1)
            predictions = torch.round(torch.sigmoid(outputs))
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = (correct / total) * 100
    print("-" * 30)
    print(f"Entraînement terminé")
    print(f"Précision sur les données de test : {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    train_and_evaluate()