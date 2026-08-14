import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader

from baseline import BreastCancerNet, get_dataloaders

# On fixe la "seed" pour avoir des résultats reproductibles (même bruit aléatoire à chaque exécution pour la démonstration)
torch.manual_seed(42)

def run_attack_with_dp():
    print("Début de l'attaque DLG avec Differential Privacy (DP) activée...")
    
    #Contextualisation : on simule un hôpital qui envoie ses gradients au serveur central
    _, test_loader = get_dataloaders(batch_size=1)
    real_data, real_label = next(iter(test_loader))
    
    model = BreastCancerNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Activation de la Differential Privacy avec Opacus
    print("L'Hôpital active le PrivacyEngine (Opacus)...")
    privacy_engine = PrivacyEngine()
    
    # Opacus va "emballer" (wrapper) notre modèle, notre optimiseur et notre DataLoader
    # pour intercepter les calculs et y injecter du bruit mathématique.
    model, optimizer, secure_test_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=test_loader,
        noise_multiplier=10, # La "force" du bouclier (plus c'est haut, plus c'est privé)
        max_grad_norm=1.0,    # Coupe les gradients trop importants qui pourraient trahir l'identité
    )
    
    # L'hôpital fait son calcul sécurisé
    optimizer.zero_grad()
    pred = model(real_data)
    loss = criterion(pred, real_label)
    
    # 1. On utilise backward() pour qu'Opacus intercepte et "coupe" (clip) les gradients
    loss.backward()
    
    # 2. On déclenche l'optimiseur : c'est LÀ qu'Opacus injecte le bruit gaussien !
    optimizer.step()
    
    # 3. On récupère les gradients qui viennent d'être bruités en mémoire
    real_gradients_with_noise = [p.grad.clone() for p in model.parameters()]

    # L'attaquant (le serveur) intercepte ces gradients bruités et tente de reconstruire les données du patient
    print("\nLe serveur intercepte les gradients bruités...")
    print("Tentative de reconstruction du dossier médical...")
    
    dummy_data = torch.randn(real_data.size(), requires_grad=True)
    dummy_label = real_label.clone() 
    
    # On utilise le modèle ORIGINAL (sans Opacus) pour l'attaquant, 
    # car l'attaquant simule les calculs mathématiques purs de son côté.
    attack_model = BreastCancerNet()
    attack_optimizer = optim.Adam([dummy_data], lr=0.1)
    
    for it in range(500):
        attack_optimizer.zero_grad()
        
        dummy_pred = attack_model(dummy_data)
        dummy_loss = criterion(dummy_pred, dummy_label)
        dummy_gradients = torch.autograd.grad(dummy_loss, attack_model.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_gradients, real_gradients_with_noise):
            grad_diff += ((gx - gy) ** 2).sum()
            
        grad_diff.backward()
        attack_optimizer.step()
        
        if it % 100 == 0:
            print(f"Itération {it} | Marge d'erreur apparente : {grad_diff.item():.6f}")

    # Résultat final
    print("\nAttaque terminée. L'attaquant regarde ce qu'il a volé :")
    
    print("\nVraies données du patient (5 caractéristiques) :")
    print([round(val, 4) for val in real_data[0][:5].tolist()])
    
    print("\nDonnées reconstruites par l'attaquant :")
    print([round(val, 4) for val in dummy_data[0][:5].tolist()])
    
    diff = torch.abs(real_data - dummy_data).mean().item()
    print(f"\nDifférence moyenne (L'erreur de l'attaquant) : {diff:.4f}")
    
    if diff > 0.5:
         print("Résultat : L'attaquant n'a obtenu que du bruit (garbage data) !")
    else:
         print("Résultat : L'attaquant a pu extraire des informations pertinentes.")

if __name__ == "__main__":
    run_attack_with_dp()