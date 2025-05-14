import pandas as pd

# Chargement des données
df = pd.read_csv("car_insurance.csv")

# Suppression des colonnes inutiles
df.drop(columns=["id", "postal_code"], inplace=True)

# Correction des valeurs aberrantes dans 'children'
df.loc[df["children"] > 10, "children"] = df.loc[df["children"] <= 10, "children"].median()

# Gestion des valeurs aberrantes dans 'speeding_violations' (exclusion des valeurs extrêmes aberrantes)
df = df[df["speeding_violations"] < 1000]

# Encodage de 'driving_experience'
driving_map = {
    "0-9y": 0,
    "10-19y": 1,
    "20-29y": 2,
    "30y+": 3
}
df["driving_experience"] = df["driving_experience"].map(driving_map)

# Encodage de 'education'
education_map = {
    "none": 0,
    "high school": 1,
    "university": 2
}
df["education"] = df["education"].map(education_map)

# Encodage de 'income'
income_map = {
    "poverty": 0,
    "working class": 1,
    "middle class": 2,
    "upper class": 3
}
df["income"] = df["income"].map(income_map)

# Transformation de 'vehicle_year' → 'after_2015'
df["after_2015"] = df["vehicle_year"].apply(lambda x: 1 if x == "after 2015" else 0)
df.drop(columns=["vehicle_year"], inplace=True)

# Transformation de 'vehicle_type' → 'sedan_type'
df["sedan_type"] = df["vehicle_type"].apply(lambda x: 1 if x == "sedan" else 0)
df.drop(columns=["vehicle_type"], inplace=True)

# Imputation des valeurs manquantes par la médiane ou mode selon le type de variable
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Vérification
print("\n--- Valeurs manquantes après traitement ---")
print(df.isnull().sum())

print("\n--- Aperçu des données prêtes à l'emploi ---")
print(df.head())

# (optionnel) Sauvegarde du jeu de données prêt pour modélisation
df.to_csv("car_insurance_prepared.csv", index=False)