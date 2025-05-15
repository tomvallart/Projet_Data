import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Charge les données depuis un fichier CSV"""
    return pd.read_csv(filepath)

def check_missing_values(df):
    """Affiche le nombre de valeurs manquantes pour chaque variable"""
    print("\n--- Nombre de valeurs manquantes par variable ---")
    print(df.isna().sum())
    return df

def handle_missing_values(df):
    """
    Gère les valeurs manquantes :
    - Supprime la colonne si > 33% de valeurs manquantes
    - Sinon, remplace par la valeur la plus fréquente (catégoriel) ou la médiane (numérique)
    """
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            if missing_percentage > 33:
                df = df.drop(columns=[column])
                print(f"Colonne {column} supprimée : {missing_percentage:.2f}% de valeurs manquantes")
            else:
                if df[column].dtype == 'object':
                    most_frequent = df[column].mode()[0]
                    df[column] = df[column].fillna(most_frequent)
                else:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
    return df

def remove_useless_columns(df):
    """Supprime les colonnes non utiles pour la classification"""
    columns_to_drop = ["id", "postal_code"]
    df = df.drop(columns=columns_to_drop)
    return df

def handle_outliers(df):
    """
    Traite les valeurs aberrantes :
    - 'children' > 10 remplacé par la médiane, puis arrondi à l'entier
    - 'speeding_violations' > 30 supprimées, puis arrondi à l'entier
    - 'credit_score' borné entre 0 inclus et 1 exclus
    """
    # Pour children : valeurs > 10 remplacées par la médiane, puis arrondi à l'entier
    median_children = df.loc[df["children"] <= 10, "children"].median()
    df.loc[df["children"] > 10, "children"] = median_children
    df["children"] = np.round(df["children"]).astype(int)

    # Pour speeding_violations : valeurs > 30 supprimées, puis arrondi à l'entier
    df = df[df["speeding_violations"] <= 30]
    df["speeding_violations"] = np.round(df["speeding_violations"]).astype(int)

    # Pour credit_score : doit être >=0 et <1, sinon ramener dans l'intervalle
    df["credit_score"] = df["credit_score"].clip(lower=0, upper=0.999999)
    
    return df

def encode_categorical_variables(df):
    """
    Encode les variables catégorielles en utilisant LabelEncoder
    (transforme les chaînes de caractères en entiers)
    """
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    return df

def normalize_features(df):
    """
    Normalise les features numériques avec StandardScaler :
    - Centre et réduit toutes les variables sauf la cible 'outcome'
    """
    scaler = StandardScaler()
    target = df['outcome']
    features = df.drop(columns=['outcome'])
    features_normalized = scaler.fit_transform(features)
    df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
    df_normalized['outcome'] = target.values
    return df_normalized

def main(filepath):
    # Chargement des données brutes
    df = load_data(filepath)
    # Vérification des valeurs manquantes
    df = check_missing_values(df)
    # Suppression des colonnes inutiles
    df = remove_useless_columns(df)
    # Gestion des valeurs manquantes
    df = handle_missing_values(df)
    # Traitement des valeurs aberrantes
    df = handle_outliers(df)
    # Encodage des variables catégorielles
    df = encode_categorical_variables(df)
    # Normalisation des variables numériques
    df = normalize_features(df)
    # Aperçu des données finales
    print("\n--- Aperçu des données préparées ---")
    print(df.head())
    print("\n--- Vérification finale des valeurs manquantes ---")
    print(df.isna().sum())
    # Création du nom du fichier de sortie
    base, ext = os.path.splitext(filepath)
    output_file = f"{base}_prepared.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDonnées préparées sauvegardées dans '{output_file}'")
    return output_file

if __name__ == "__main__":
    main("car_insurance.csv")