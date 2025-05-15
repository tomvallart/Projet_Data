import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# Chargement du jeu d'entraînement préparé par extraction_jeux.py
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Définition des classifieurs à comparer
classifieurs = {
    "Régression logistique": LogisticRegression(),
    "Perceptron": Perceptron(),
    "K plus proches voisins (K=5)": KNeighborsClassifier(n_neighbors=5)
}

# Validation croisée (KFold)
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

moyennes = []
ecarts = []
noms = []

print(f"Comparaison des classifieurs avec validation croisée ({k} folds) :\n")
for nom, clf in classifieurs.items():
    # Apprentissage et évaluation par validation croisée
    scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    print(f"{nom} :")
    print(f"  Scores : {np.round(scores, 3)}")
    print(f"  Moyenne : {scores.mean():.3f} | Écart-type : {scores.std():.3f}\n")
    moyennes.append(scores.mean())
    ecarts.append(scores.std())
    noms.append(nom)

# Affichage graphique des scores moyens avec barres d'erreur (écart-type)
plt.figure(figsize=(8, 5))
bars = plt.bar(noms, moyennes, yerr=ecarts, capsize=10, color=['royalblue', 'orange', 'green'])
plt.ylabel("Score moyen (accuracy)")
plt.title("Comparaison des classifieurs (validation croisée)")
plt.ylim(0, 1)
for bar, moyenne in zip(bars, moyennes):
    plt.text(bar.get_x() + bar.get_width()/2, moyenne + 0.02, f"{moyenne:.3f}", ha='center', fontsize=11)
plt.tight_layout()
plt.show()