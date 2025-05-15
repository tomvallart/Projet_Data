import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Chargement du jeu de données préparé
df = pd.read_csv("car_insurance_prepared.csv")

# Calcul des corrélations avec la variable cible
correlations = df.corr(numeric_only=True)["outcome"].abs().sort_values(ascending=False)
top_features = correlations.index[1:4]  # 3 variables les plus corrélées

# Utilise uniquement ces variables comme variables explicatives
X = df[top_features]
y = df["outcome"]

# Séparation en jeu d'apprentissage (80%) et jeu de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# Sauvegarde des jeux sous forme de fichiers numpy
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Affichage des proportions
print(f"Taille du jeu d'apprentissage : {X_train.shape[0]} lignes ({X_train.shape[0]/df.shape[0]:.2%})")
print(f"Taille du jeu de test : {X_test.shape[0]} lignes ({X_test.shape[0]/df.shape[0]:.2%})")

# Affichage graphique : histogrammes des 3 variables explicatives sur le jeu d'apprentissage
plt.figure(figsize=(10, 6))
for i, feature in enumerate(top_features):
    plt.hist(X_train[:, i], bins=20, alpha=0.5, label=feature)
plt.title("Distribution des 3 variables explicatives les plus corrélées (jeu d'apprentissage)")
plt.xlabel("Valeur")
plt.ylabel("Nombre d'occurrences")
plt.legend()
plt.tight_layout()
plt.show()