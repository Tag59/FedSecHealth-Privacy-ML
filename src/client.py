import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import flwr as fl
import argparse
from collections import OrderedDict
from opacus import PrivacyEngine

# Importation du modèle et des fonctions de prétraitement depuis baseline.py
from baseline import BreastCancerNet, get_dataloaders

# Simulation du découpage des données pour chaque client
def get_client_data(client_id: int, num_clients: int = 3):
    """
    Prend le dataset complet et le découpe en parts égales.
    Chaque hôpital (client_id) ne verra que SA portion des données.
    """
    train_loader, test_loader = get_dataloaders(batch_size=32)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Calcul de la taille de chaque morceau
    train_len = len(train_dataset) // num_clients
    train_lengths = [train_len] * (num_clients - 1)
    train_lengths.append(len(train_dataset) - sum(train_lengths)) # Le reste pour le dernier

    test_len = len(test_dataset) // num_clients
    test_lengths = [test_len] * (num_clients - 1)
    test_lengths.append(len(test_dataset) - sum(test_lengths))

    # Découpage avec une "seed" fixe pour que ce soit reproductible
    generator = torch.Generator().manual_seed(42)
    train_partitions = random_split(train_dataset, train_lengths, generator=generator)
    test_partitions = random_split(test_dataset, test_lengths, generator=generator)

    # On renvoie uniquement la portion de l'hôpital demandé
    return (
        DataLoader(train_partitions[client_id], batch_size=32, shuffle=True),
        DataLoader(test_partitions[client_id], batch_size=32, shuffle=False)
    )

# Définition du client fédéré
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, test_loader):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        print ("Activation du bouclier Opacus pour cet hôpital...")
        self.privacy_engine = PrivacyEngine()
        self.net, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.net,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.2,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        """Extrait les poids. Opacus ajoute une carapace '_module' qu'il faut traverser."""
        model_to_extract = self.net._module if hasattr(self.net, '_module') else self.net
        return [val.cpu().numpy() for _, val in model_to_extract.state_dict().items()]

    def set_parameters(self, parameters):
        """Met à jour le modèle local avec les données du serveur."""
        model_to_update = self.net._module if hasattr(self.net, '_module') else self.net
        params_dict = zip(model_to_update.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model_to_update.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Entraînement local sécurisé."""
        self.set_parameters(parameters)
        self.net.train()
        
        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.net(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.net(batch_X)
                loss += self.criterion(outputs, batch_y).item()
                predictions = torch.round(torch.sigmoid(outputs))
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
        accuracy = correct / total
        print(f"Évaluation Hôpital | Précision avec DP: {accuracy * 100:.2f}%")
        return float(loss), int(total), {"accuracy": float(accuracy)}

# Lancement du client fédéré
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hôpital Client Sécurisé")
    parser.add_argument("--id", type=int, required=True, help="ID de l'hôpital")
    args = parser.parse_args()

    print(f"Démarrage de l'Hôpital {args.id}...")
    
    net = BreastCancerNet()
    train_loader, test_loader = get_client_data(client_id=args.id)
    
    client = HospitalClient(net, train_loader, test_loader)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
