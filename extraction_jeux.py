import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement du jeu de données préparé
df = pd.read_csv("car_insurance_prepared.csv")

# Afficher les noms de colonnes pour identifier la variable cible
print(df.columns)

# Remplacez 'target' par le nom correct de la colonne cible
# Par exemple, si la colonne s'appelle 'class', utilisez :
# X = df.drop(columns=["class"])
# y = df["class"]

# Supposons que la variable cible s'appelle 'target'
X = df.drop(columns=["target"])  # Variables explicatives
y = df["target"]                 # Variable cible

# Séparation en jeu d'apprentissage (80%) et jeu de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Taille du jeu d'apprentissage : {X_train.shape[0]} lignes")
print(f"Taille du jeu de test : {X_test.shape[0]} lignes")
print(f"Proportion apprentissage : {X_train.shape[0] / df.shape[0]:.2f}")
print(f"Proportion test : {X_test.shape[0] / df.shape[0]:.2f}")

# Les jeux sont sous forme de tableaux Numpy (ou DataFrame/Series selon le paramétrage)