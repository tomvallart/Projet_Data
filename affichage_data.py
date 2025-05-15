import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("car_insurance.csv")

# Aperçu des 5 premières lignes
print(df.head())

# Dimensions du jeu de données
print("Dimensions :", df.shape)

# Informations sur les types de variables et valeurs manquantes
print(df.info())

# Statistiques descriptives (seulement pour les variables numériques)
print(df.describe())