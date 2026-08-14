import torch
import torch.nn as nn
import torch.optim as optim
from baseline import BreastCancerNet, get_dataloaders

# On fixe la "seed" pour avoir des résultats reproductibles (même bruit aléatoire à chaque exécution pour la démonstration)
torch.manual_seed(42)

def run_attack():
    print("Début de l'attaque DLG (Deep Leakage from Gradients)...")
    
    #Contextualisation : on simule un hôpital qui envoie ses gradients au serveur central
    _, test_loader = get_dataloaders(batch_size=1)
    real_data, real_label = next(iter(test_loader))
    
    model = BreastCancerNet()
    criterion = nn.BCEWithLogitsLoss()
    
    # L'hôpital fait son calcul (Forward pass & Backward pass)
    pred = model(real_data)
    loss = criterion(pred, real_label)
    
    # L'hôpital calcule les gradients
    real_gradients = torch.autograd.grad(loss, model.parameters())
    
    # L'attaquant (le serveur) intercepte ces gradients et tente de reconstruire les données du patient
    print("\nLe serveur intercepte les gradients du patient...")
    print("Tentative de reconstruction du dossier médical à partir de zéro...")
    
    # L'attaquant crée un "faux patient" avec des données totalement aléatoires (du bruit)
    dummy_data = torch.randn(real_data.size(), requires_grad=True)
    # Pour simplifier ce PoC, on suppose que l'attaquant devine le label (ex: Tumeur = 1)
    dummy_label = real_label.clone() 
    
    # L'attaquant va maintenant optimiser les données du faux patient pour que ses gradients correspondent à ceux du vrai patient
    optimizer = optim.Adam([dummy_data], lr=0.1)
    
    for it in range(500):
        optimizer.zero_grad()
        
        # Le serveur calcule les gradients générés par son faux patient
        dummy_pred = model(dummy_data)
        dummy_loss = criterion(dummy_pred, dummy_label)
        dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        # L'objectif de l'attaquant : minimiser la différence entre vrais et faux gradients
        grad_diff = 0
        for gx, gy in zip(dummy_gradients, real_gradients):
            grad_diff += ((gx - gy) ** 2).sum()
            
        grad_diff.backward()
        optimizer.step() # On modifie le faux patient
        
        if it % 100 == 0:
            print(f"Itération {it} | Marge d'erreur des gradients : {grad_diff.item():.6f}")
            
    # Résultat final : le faux patient est maintenant très proche du vrai patient en termes de gradients
    print("\nAttaque terminée. Comparons les résultats :")
    
    print("\n Vraies données du patient (5 premières caractéristiques) :")
    # On arrondit pour une lecture plus facile
    print([round(val, 4) for val in real_data[0][:5].tolist()])
    
    print("\nDonnées reconstruites par l'attaquant :")
    print([round(val, 4) for val in dummy_data[0][:5].tolist()])
    
    diff = torch.abs(real_data - dummy_data).mean().item()
    print(f"\nDifférence moyenne par caractéristique médicale : {diff:.4f}")
    print("Résultat : Le hacker a reconstruit l'ADN médical du patient à partir d'une simple équation de gradients")

if __name__ == "__main__":
    run_attack()