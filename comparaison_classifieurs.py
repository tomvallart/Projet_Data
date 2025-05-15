import matplotlib.pyplot as plt
from validation_croisee import validation_croisee_modele
from modeles import get_modele
from extraction_jeux import pipeline_extraction_jeux

# Extraction des jeux de données et des features
X_train, X_test, y_train, y_test, top_features = pipeline_extraction_jeux(afficher=False)

classifieurs = {
    "Régression logistique": get_modele("logistic"),
    "Perceptron": get_modele("perceptron"),
    "K plus proches voisins (K=5)": get_modele("knn", n_neighbors=5)
}

k = 25  # nombre de folds pour la validation croisée

moyennes_cv = []
ecarts_cv = []
scores_test = []
noms = []

print(f"Comparaison des classifieurs : score test vs validation croisée ({k} folds) :\n")
for nom, clf in classifieurs.items():
    # Validation croisée
    scores = validation_croisee_modele(clf, X_train, y_train, k=k)
    # Score sur le jeu de test classique
    clf.fit(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    print(f"{nom} :")
    print(f"  Score test : {score_test:.3f}")
    print(f"  Moyenne CV : {scores.mean():.3f} | Écart-type CV : {scores.std():.3f}\n")
    moyennes_cv.append(scores.mean())
    ecarts_cv.append(scores.std())
    scores_test.append(score_test)
    noms.append(nom)

# Affichage graphique : barres groupées
import numpy as np
x = np.arange(len(noms))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, scores_test, width, label='Score test', color='royalblue')
bars2 = plt.bar(x + width/2, moyennes_cv, width, yerr=ecarts_cv, capsize=8, label='Validation croisée', color='orange')

for bar, val in zip(bars1, scores_test):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha='center', fontsize=11)
for bar, val in zip(bars2, moyennes_cv):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha='center', fontsize=11)

plt.ylabel("Accuracy")
plt.title("Comparaison des classifieurs : test vs validation croisée")
plt.ylim(0, 1)
plt.xticks(x, noms)
plt.legend()
plt.tight_layout()
plt.show()