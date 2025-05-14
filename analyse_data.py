import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("car_insurance.csv")

# Aperçu des données
print("\n--- Aperçu des données ---")
print(df.head())

# Dimensions du dataset
print("\n--- Dimensions ---")
print(df.shape)

# Infos sur les types et les valeurs manquantes
print("\n--- Informations générales ---")
print(df.info())

# Statistiques descriptives pour les colonnes numériques
print("\n--- Statistiques descriptives ---")
print(df.describe())

# Valeurs uniques par colonne
print("\n--- Valeurs uniques par colonne ---")
for col in df.columns:
    print(f"\n{col} : {df[col].nunique()} valeurs uniques")
    print(df[col].value_counts())

# Valeurs manquantes
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

# Répartition de la variable cible
print("\n--- Répartition de la variable cible ---")
sns.countplot(data=df, x="outcome")
plt.title("Répartition de la variable cible : Outcome")
plt.xlabel("Demande d'indemnisation")
plt.ylabel("Nombre de clients")
plt.show()

# Histogrammes des variables numériques
print("\n--- Histogrammes des variables numériques ---")
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histogrammes des variables numériques")
plt.show()
