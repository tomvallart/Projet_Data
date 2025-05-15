import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# Chargement du jeu d'entraînement
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Définition du modèle de régression logistique
logreg = LogisticRegression()

# Définition de la validation croisée (KFold)
k = 25  # nombre de plis (folds)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Application de la validation croisée
scores = cross_val_score(logreg, X, y, cv=kf, scoring='accuracy')

print(f"Scores de validation croisée (accuracy) pour chaque fold ({k} folds) :")
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: {score:.3f}")

print(f"\nMoyenne des scores : {scores.mean():.3f}")
print(f"Écart-type des scores : {scores.std():.3f}")

# Visualisation améliorée des scores
plt.figure(figsize=(9, 5))
bars = plt.bar(range(1, k+1), scores, color='royalblue', edgecolor='black', width=0.6)
plt.axhline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne = {scores.mean():.3f}')
plt.axhline(scores.mean() + scores.std(), color='green', linestyle=':', linewidth=1, label=f'+1 écart-type')
plt.axhline(scores.mean() - scores.std(), color='green', linestyle=':', linewidth=1, label=f'-1 écart-type')

# Ajout des valeurs sur chaque barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', va='bottom', fontsize=10)

plt.xlabel("Fold", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"Scores de validation croisée ({k} folds) - Régression logistique", fontsize=14)
plt.ylim(0, 1)
plt.xticks(range(1, k+1))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("""
Analyse :
- La validation croisée permet d'obtenir une estimation plus robuste de la performance du modèle.
- La moyenne des scores est à comparer avec le score obtenu sans validation croisée.
- Un écart-type faible indique une bonne stabilité du modèle sur différents sous-ensembles.
""")