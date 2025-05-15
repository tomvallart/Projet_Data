import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib.colors import ListedColormap

def charger_et_preparer_donnees(filepath):
    """Charge, encode et normalise les données"""
    df = pd.read_csv(filepath)
    # Encodage des variables qualitatives
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # Normalisation des variables numériques
    features = df.drop(columns=["outcome"])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled["outcome"] = df["outcome"].values
    return df, df_scaled

def afficher_heatmap_correlation(corr_matrix):
    """Affiche la heatmap de la matrice de corrélation"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.round(2), annot=True, cmap='coolwarm', center=0, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    plt.title("Heatmap des corrélations (avec diagonale)", fontsize=16)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.show()

def afficher_scatter_matrix(df_scaled, top_corr_vars):
    """Affiche la scatter matrix des variables les plus corrélées avec la cible"""
    vars_to_plot = top_corr_vars + ["outcome"]
    cmap = ListedColormap(sns.color_palette("Set1", n_colors=2).as_hex())
    scatter_matrix(df_scaled[vars_to_plot], figsize=(10, 8), diagonal='hist', alpha=0.8,
                   c=df_scaled["outcome"], cmap=cmap, marker='o', hist_kwds={'bins': 20})
    plt.suptitle("Scatter matrix des variables les plus corrélées avec la cible (+ outcome)", fontsize=16)
    plt.show()

def main(filepath):
    # Chargement et préparation des données
    df, df_scaled = charger_et_preparer_donnees(filepath)
    # Calcul de la matrice de corrélation
    corr_matrix = df_scaled.corr()
    print("\n--- Matrice de corrélation ---")
    print(corr_matrix)
    # Affichage de la heatmap
    afficher_heatmap_correlation(corr_matrix)
    # Corrélation de chaque variable avec la variable cible
    print("\n--- Corrélation avec la variable cible (outcome) ---")
    print(corr_matrix["outcome"].sort_values(ascending=False).round(2))
    # Afficher uniquement les corrélations > 0.3 ou < -0.3
    seuil = 0.3
    corr_cible = corr_matrix["outcome"].abs()
    print("\n--- Variables avec |corr| > 0.3 ---")
    print(corr_cible[corr_cible > seuil])
    # Sélection des variables les plus corrélées avec la cible
    top_corr_vars = corr_cible[corr_cible > seuil].index.tolist()
    top_corr_vars = [var for var in top_corr_vars if var != "outcome"]
    print(f"\nVariables les plus corrélées avec la cible (|corr| > {seuil}): {top_corr_vars}")
    # Affichage scatter matrix
    if top_corr_vars:
        afficher_scatter_matrix(df_scaled, top_corr_vars)
    else:
        print("Aucune variable n'a une corrélation absolue > 0.3 avec la cible.")

if __name__ == "__main__":
    main("car_insurance_prepared.csv")