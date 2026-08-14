# FedSecHealth: Privacy-Preserving Federated Learning in Healthcare

Ce dépôt contient une Preuve de Concept (PoC) développée dans le cadre d'un Master en Cybersécurité. Le projet explore la sécurisation des architectures de Machine Learning distribué appliquées aux données de santé critiques.

## Contexte et Problématique
L'entraînement de modèles d'Intelligence Artificielle en milieu hospitalier se heurte à des contraintes réglementaires strictes (RGPD, certification HDS) empêchant la centralisation des données des patients. 

L'Apprentissage Fédéré (Federated Learning) répond partiellement à ce problème en décentralisant l'entraînement. Cependant, cette architecture reste vulnérable aux attaques d'inférence, notamment le **Deep Leakage from Gradients (DLG)**, qui permet à un acteur malveillant (ou un serveur compromis) de reconstruire les données brutes d'un patient à partir des simples poids mathématiques échangés sur le réseau.

## Architecture et Contributions
Ce projet implémente une architecture complète démontrant la vulnérabilité puis sa remédiation :

1. **Baseline Centralisée :** Étalonnage d'un réseau de neurones sur le dataset *Breast Cancer Wisconsin*.
2. **Attaque DLG (Gradient Inversion) :** Simulation d'un serveur compromis parvenant à reconstruire les données médicales d'un patient avec une marge d'erreur quasi nulle.
3. **Remédiation par Confidentialité Différentielle (DP) :** Intégration de la bibliothèque Opacus pour injecter un bruit gaussien maîtrisé lors du calcul des gradients locaux.
4. **Fédération Sécurisée :** Déploiement du modèle sécurisé sur un réseau fédéré simulé (3 clients hospitaliers) via le framework Flower.

## Stack Technique
* **Framework ML :** PyTorch
* **Federated Learning :** Flower (`flwr`)
* **Differential Privacy :** Opacus
* **Données :** Scikit-Learn (Dataset Breast Cancer Wisconsin)

---

## Instructions de Déploiement et d'Exécution (Local)

### Étape 1 : Prérequis et Installation
Clonez le dépôt et installez l'environnement virtuel avec les dépendances requises.

```bash
git clone [https://github.com/votre-profil/FedSecHealth-Privacy-ML.git](https://github.com/votre-profil/FedSecHealth-Privacy-ML.git)
cd FedSecHealth-Privacy-ML
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Étape 2 : L'Étalonnage (Baseline)
Cette étape permet de vérifier la capacité d'apprentissage du modèle dans un environnement centralisé classique (sans contrainte de sécurité ni de réseau).
```bash
python src/baseline.py
```
Résultat attendu : La fonction de perte (Loss) diminue progressivement. La précision finale sur l'ensemble de test doit se situer aux alentours de 97-98%. Ceci constitue notre Gold Standard

### Étape 2 : Démonstration de la Vulnérabilité (Attaque DLG)
Cette simulation démontre pourquoi l'Apprentissage Fédéré seul ne garantit pas l'anonymat. Le script simule un patient envoyant ses gradients, et un serveur interceptant ces derniers pour forcer l'inversion.
```bash
python src/attack.py
```
Résultat attendu : L'optimiseur de l'attaquant réduit la différence des gradients à 0.0000. Les données médicales reconstruites sont identiques aux vraies données du patient. La compromission est totale.

### Étape 4 : Application du Bouclier (Differential Privacy)
Le même scénario d'attaque est rejoué, mais l'hôpital cible active cette fois-ci le PrivacyEngine d'Opacus
```bash
python src/attack_dp.py
```
Résultat attendu : L'algorithme d'inversion échoue à faire converger les gradients. La différence moyenne explose. Les données reconstruites par l'attaquant sont des valeurs aberrantes (garbage data). La vie privée du patient est mathématiquement garantie.

### Étape 5 : Simulation du Réseau Fédéré Sécurisé
L'objectif final est de prouver que le modèle peut toujours apprendre de manière distribuée, malgré le bruit cryptographique ajouté à l'étape précédente.

Il faut ouvrir quatre terminaux distincts (il faut que que l'environnement virtuel soit activé dans chacun d'eux) :
Terminal 1 (Serveur central) :

```bash
python src/server.py
```

Terminaux 2, 3 et 4 (Nœuds hospitaliers) :

```bash
python src/client.py --id 0
python src/client.py --id 1
python src/client.py --id 2
```
Résultat et Analyse : Le serveur détecte les 3 clients et orchestre 3 cycles d'entraînement (Rounds). La Loss globale diminue à chaque cycle. L'évaluation locale des hôpitaux affiche une précision allant de 82% à 92%.

Conclusion : Nous observons une perte de précision (Trade-off) d'environ 10% par rapport à la baseline centralisée, ce qui représente le coût de la sécurisation absolue (Differential Privacy) appliquée sur un réseau décentralisé (Non-IID). Le compromis reste viable pour une exploitation clinique.
