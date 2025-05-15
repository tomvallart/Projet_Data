import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Charge les données depuis un fichier CSV"""
    return pd.read_csv(filepath)

def check_missing_values(df):
    """Vérifie et affiche les valeurs manquantes pour chaque variable"""
    print("\n--- Nombre de valeurs manquantes par variable ---")
    print(df.isna().sum())
    return df

def handle_missing_values(df):
    """Gère les valeurs manquantes selon les critères du projet"""
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
    """Traite les valeurs aberrantes identifiées dans les données"""
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
    """Encode les variables catégorielles en utilisant LabelEncoder"""
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    return df

def normalize_features(df):
    """Normalise les features numériques avec StandardScaler"""
    scaler = StandardScaler()
    target = df['outcome']
    features = df.drop(columns=['outcome'])
    features_normalized = scaler.fit_transform(features)
    df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
    df_normalized['outcome'] = target.values
    return df_normalized

def main():
    df = load_data("car_insurance.csv")
    df = check_missing_values(df)
    df = remove_useless_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categorical_variables(df)
    df = normalize_features(df)
    print("\n--- Aperçu des données préparées ---")
    print(df.head())
    print("\n--- Vérification finale des valeurs manquantes ---")
    print(df.isna().sum())
    df.to_csv("car_insurance_prepared.csv", index=False)
    print("\nDonnées préparées sauvegardées dans 'car_insurance_prepared.csv'")

if __name__ == "__main__":
    main()