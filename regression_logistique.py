import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

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

# Prédictions sur le jeu de test
y_pred = logreg.predict(X_test)

# Affichage pour chaque échantillon
print("\nClasse réelle | Classe prédite")
for vrai, predit in zip(y_test, y_pred):
    print(f"{int(vrai):>12} | {int(predit):>13}")

# Calcul des métriques
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- Évaluation du modèle ---")
print(f"Accuracy (exactitude)      : {acc:.3f}")
print(f"Precision (précision)      : {prec:.3f}")
print(f"Recall (rappel/sensibilité): {rec:.3f}")
print(f"F1-score                   : {f1:.3f}")
print("Matrice de confusion :\n", cm)

print("""
Signification des métriques :
- Accuracy : proportion de bonnes prédictions (tous labels confondus).
- Precision : parmi les prédits positifs, combien sont vraiment positifs.
- Recall : parmi les vrais positifs, combien sont retrouvés par le modèle.
- F1-score : moyenne harmonique de la précision et du rappel.
- Matrice de confusion : tableau croisant les classes réelles et prédites.
""")

# Comptage des classes pour chaque jeu
train_counts = Counter(y_train)
test_counts = Counter(y_test)
labels = sorted(set(y_train) | set(y_test))

train_values = [train_counts.get(l, 0) for l in labels]
test_values = [test_counts.get(l, 0) for l in labels]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(9, 6))
bars1 = plt.bar([i - width/2 for i in x], train_values, width, label='Apprentissage', color='royalblue')
bars2 = plt.bar([i + width/2 for i in x], test_values, width, label='Test', color='orange')

# Ajout des valeurs et pourcentages sur les barres
for i, bar in enumerate(bars1):
    height = bar.get_height()
    percent = height / len(y_train) * 100
    plt.annotate(f'{height}\n({percent:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, color='navy')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    percent = height / len(y_test) * 100
    plt.annotate(f'{height}\n({percent:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, color='darkorange')

plt.xticks(x, labels)
plt.xlabel("Valeur de la variable cible (outcome)")
plt.ylabel("Nombre d'occurrences")
plt.title("Répartition des classes dans les jeux d'apprentissage et de test")

# Ajout d'un encadré avec les scores principaux (cadre plus grand et plus bas)
scores_text = (
    f"Accuracy : {acc:.3f}\n"
    f"Précision : {prec:.3f}\n"
    f"Rappel : {rec:.3f}\n"
    f"F1-score : {f1:.3f}"
)
plt.gca().text(
    1.05, 0.4, scores_text, transform=plt.gca().transAxes,
    fontsize=13,
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1.2', linewidth=1.5)
)

plt.legend()
plt.tight_layout()
plt.show()