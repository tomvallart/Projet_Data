import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Ajoute cet import en haut du fichier

# Chargement du jeu de données préparé
df = pd.read_csv("car_insurance_prepared.csv")

# Afficher les noms de colonnes pour identifier la variable cible
print(df.columns)

# Utilise 'outcome' comme variable cible
X = df.drop(columns=["outcome"])  # Variables explicatives
y = df["outcome"]                 # Variable cible

# Séparation en jeu d'apprentissage (80%) et jeu de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

print(f"Taille du jeu d'apprentissage : {X_train.shape[0]} lignes")
print(f"Taille du jeu de test : {X_test.shape[0]} lignes")
print(f"Proportion apprentissage : {X_train.shape[0] / df.shape[0]:.2f}")
print(f"Proportion test : {X_test.shape[0] / df.shape[0]:.2f}")


# Affichage graphique de la distribution de la variable cible dans les jeux d'apprentissage et de test
plt.figure(figsize=(8, 4))
plt.hist(y_train, bins=20, alpha=0.7, label='y_train')
plt.hist(y_test, bins=20, alpha=0.7, label='y_test')
plt.title("Distribution de la variable cible (outcome)")
plt.xlabel("Valeur")
plt.ylabel("Nombre d'occurrences")
plt.legend()
plt.show()