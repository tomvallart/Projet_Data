import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def charger_dataframe(filepath):
    """Charge le DataFrame brut"""
    return pd.read_csv(filepath)

def afficher_apercu(df):
    print("\n--- Aperçu des données ---")
    print(df.head())

def afficher_dimensions(df):
    print("\n--- Dimensions ---")
    print(df.shape)

def afficher_infos(df):
    print("\n--- Informations générales ---")
    print(df.info())

def afficher_stats(df):
    print("\n--- Statistiques descriptives ---")
    print(df.describe())

def afficher_valeurs_uniques(df):
    print("\n--- Valeurs uniques par colonne ---")
    for col in df.columns:
        print(f"\n{col} : {df[col].nunique()} valeurs uniques")
        print(df[col].value_counts())

def afficher_valeurs_manquantes(df):
    print("\n--- Valeurs manquantes ---")
    print(df.isnull().sum())

def afficher_repartition_cible(df):
    print("\n--- Répartition de la variable cible ---")
    sns.countplot(data=df, x="outcome")
    plt.title("Répartition de la variable cible : Outcome")
    plt.xlabel("Demande d'indemnisation")
    plt.ylabel("Nombre de clients")
    plt.show()

def afficher_histogrammes(df):
    print("\n--- Histogrammes des variables numériques ---")
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histogrammes des variables numériques")
    plt.show()

def main(filepath):
    df = charger_dataframe(filepath)
    afficher_apercu(df)
    afficher_dimensions(df)
    afficher_infos(df)
    afficher_stats(df)
    afficher_valeurs_uniques(df)
    afficher_valeurs_manquantes(df)
    afficher_repartition_cible(df)
    afficher_histogrammes(df)

if __name__ == "__main__":
    main("car_insurance.csv")