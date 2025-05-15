import pickle

def sauvegarder_modele(modele, chemin):
    """Sauvegarde un modèle entraîné dans un fichier avec pickle."""
    with open(chemin, "wb") as f:
        pickle.dump(modele, f)
    print(f"Modèle sauvegardé dans {chemin}")

def charger_modele(chemin):
    """Charge un modèle entraîné depuis un fichier pickle."""
    with open(chemin, "rb") as f:
        modele = pickle.load(f)
    print(f"Modèle chargé depuis {chemin}")
    return modele