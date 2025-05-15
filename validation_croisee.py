import numpy as np
from sklearn.model_selection import KFold, cross_val_score

def validation_croisee_modele(model, X, y, k=10, random_state=42):
    """Effectue une validation croisée et retourne les scores"""
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return scores