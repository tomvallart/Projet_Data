import matplotlib.pyplot as plt
from validation_croisee import validation_croisee_modele
from modeles import get_modele
from extraction_jeux import pipeline_extraction_jeux
from sauvegarde import sauvegarder_modele
import numpy as np

def comparer_classifieurs(X_train, X_test, y_train, y_test, k=25):
    """Compare les classifieurs, affiche les scores et retourne les résultats"""
    classifieurs = {
        "Régression logistique": get_modele("logistic"),
        "Perceptron": get_modele("perceptron"),
        "K plus proches voisins (K=5)": get_modele("knn", n_neighbors=5)
    }

    moyennes_cv = []
    ecarts_cv = []
    scores_test = []
    noms = []
    modeles_fits = []

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
        modeles_fits.append(clf)
    return noms, scores_test, moyennes_cv, ecarts_cv, modeles_fits

def afficher_resultats(noms, scores_test, moyennes_cv, ecarts_cv):
    """Affiche le graphique comparatif des scores"""
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

def sauvegarder_meilleur_modele(modeles_fits, noms, scores_test):
    """Sauvegarde le meilleur modèle selon le score test"""
    idx_best = int(np.argmax(scores_test))
    meilleur_modele = modeles_fits[idx_best]
    nom_meilleur = noms[idx_best]
    chemin = f"meilleur_modele_{nom_meilleur.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
    sauvegarder_modele(meilleur_modele, chemin)
    print(f"Le meilleur modèle ({nom_meilleur}) a été sauvegardé.")

def main(filepath):
    # Extraction des jeux de données et des features
    X_train, X_test, y_train, y_test, top_features = pipeline_extraction_jeux(filepath,afficher=False)
    noms, scores_test, moyennes_cv, ecarts_cv, modeles_fits = comparer_classifieurs(X_train, X_test, y_train, y_test, k=25)
    afficher_resultats(noms, scores_test, moyennes_cv, ecarts_cv)
    sauvegarder_meilleur_modele(modeles_fits, noms, scores_test)

if __name__ == "__main__":
    main("car_insurance_prepared.csv")