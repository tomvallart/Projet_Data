import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter

# Chargement des jeux numpy
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Entraînement du modèle de régression logistique
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print("Score sur le jeu d'apprentissage :", logreg.score(X_train, y_train))
print("Score sur le jeu de test :", logreg.score(X_test, y_test))

# Comptage des classes pour chaque jeu
train_counts = Counter(y_train)
test_counts = Counter(y_test)
labels = sorted(set(y_train) | set(y_test))

train_values = [train_counts.get(l, 0) for l in labels]
test_values = [test_counts.get(l, 0) for l in labels]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
bars1 = plt.bar([i - width/2 for i in x], train_values, width, label='Apprentissage', color='royalblue')
bars2 = plt.bar([i + width/2 for i in x], test_values, width, label='Test', color='orange')

# Ajout des valeurs sur les barres
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.annotate(f'{height}\n({height/(len(y_train) if bar in bars1 else len(y_test))*100:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points de décalage vertical
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

plt.xticks(x, labels)
plt.xlabel("Valeur de la variable cible (outcome)")
plt.ylabel("Nombre d'occurrences")
plt.title("Répartition des classes dans les jeux d'apprentissage et de test")
plt.legend()
plt.tight_layout()
plt.show()