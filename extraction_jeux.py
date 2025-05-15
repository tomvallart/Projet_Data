import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def charger_dataframe(filepath):
    """Charge le DataFrame préparé à partir d'un fichier CSV"""
    return pd.read_csv(filepath)

def extraire_top_features(df, n=3):
    """
    Retourne les n variables les plus corrélées avec la cible 'outcome'
    (on saute 'outcome' lui-même qui est toujours la plus corrélée à elle-même)
    """
    correlations = df.corr(numeric_only=True)["outcome"].abs().sort_values(ascending=False)
    top_features = correlations.index[1:n+1]  # On saute 'outcome' lui-même
    return list(top_features)

def creer_jeux(X, y, test_size=0.2, random_state=42):
    """
    Sépare les données en jeux d'apprentissage et de test,
    sauvegarde les jeux en fichiers .npy et retourne les tableaux numpy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    return X_train, X_test, y_train, y_test

def afficher_proportions(X_train, X_test, df):
    """
    Affiche la taille des jeux d'apprentissage et de test
    en nombre de lignes et en pourcentage du total.
    """
    print(f"Taille du jeu d'apprentissage : {X_train.shape[0]} lignes ({X_train.shape[0]/df.shape[0]:.2%})")
    print(f"Taille du jeu de test : {X_test.shape[0]} lignes ({X_test.shape[0]/df.shape[0]:.2%})")

def afficher_histogrammes(X_train, top_features):
    """
    Affiche les histogrammes des variables explicatives sélectionnées
    sur le jeu d'apprentissage.
    """
    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(top_features):
        plt.hist(X_train[:, i], bins=20, alpha=0.5, label=feature)
    plt.title("Distribution des 3 variables explicatives les plus corrélées (jeu d'apprentissage)")
    plt.xlabel("Valeur")
    plt.ylabel("Nombre d'occurrences")
    plt.legend()
    plt.tight_layout()
    plt.show()

def pipeline_extraction_jeux(filepath, n_features=3, test_size=0.2, random_state=42, afficher=True):
    """
    Pipeline complet :
    - Charge le DataFrame
    - Sélectionne les n variables les plus corrélées avec la cible
    - Sépare en jeux d'apprentissage et de test
    - Affiche les proportions et histogrammes si demandé
    - Retourne X_train, X_test, y_train, y_test, top_features
    """
    df = charger_dataframe(filepath)
    top_features = extraire_top_features(df, n=n_features)
    X = df[top_features]
    y = df["outcome"]
    X_train, X_test, y_train, y_test = creer_jeux(X, y, test_size=test_size, random_state=random_state)
    if afficher:
        afficher_proportions(X_train, X_test, df)
        afficher_histogrammes(X_train, top_features)
    return X_train, X_test, y_train, y_test, top_features

if __name__ == "__main__":
    pipeline_extraction_jeux("car_insurance_prepared.csv")