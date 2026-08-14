import flwr as fl

def main():
    print("Démarrage du Serveur Central...")
    print ("En attente de la connexion des clients...")

    # Configuration du serveur Flower : on veut trois cycles d'entraînement
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,  # Nombre minimum de clients pour l'entraînement
        min_evaluate_clients=3,  # Nombre minimum de clients pour l'évaluation
        min_available_clients=3,  # Nombre minimum de clients disponibles pour participer
    )

    # Lancement du serveur en local
    fl.server.start_server(
        server_address="localhost:8080",  
        config=fl.server.ServerConfig(num_rounds=3),  # Nombre de cycles d'entraînement
        strategy=strategy,
    )

if __name__ == "__main__":
    main()