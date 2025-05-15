import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib.colors import ListedColormap

# Chargement des données préparées
df = pd.read_csv("car_insurance_prepared.csv")

# Encodage des variables qualitatives
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Normalisation des variables numériques
features = df.drop(columns=["outcome"])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled["outcome"] = df["outcome"].values

# Calcul de la matrice de corrélation
corr_matrix = df_scaled.corr()
print("\n--- Matrice de corrélation ---")
print(corr_matrix)

# Affichage optimisé de la matrice de corrélation (heatmap améliorée)
# Affichage optimisé de la matrice de corrélation (heatmap avec diagonale)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.round(2), annot=True, cmap='coolwarm', center=0, fmt=".2f",
            linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)  # Suppression du masque
plt.title("Heatmap des corrélations (avec diagonale)", fontsize=16)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.show()

# Corrélation de chaque variable avec la variable cible
print("\n--- Corrélation avec la variable cible (outcome) ---")
print(corr_matrix["outcome"].sort_values(ascending=False).round(2))

# Afficher uniquement les corrélations > 0.3 ou < -0.3
seuil = 0.3
corr_cible = corr_matrix["outcome"].abs()
print(corr_cible[corr_cible > seuil])

# Sélection des variables les plus corrélées avec la cible
top_corr = corr_matrix["outcome"].abs().sort_values(ascending=False)[1:4]  # 3 plus corrélées hors outcome
print("\nVariables les plus corrélées avec la cible :", list(top_corr.index))

# Visualisation scatter matrix améliorée
vars_to_plot = list(top_corr.index) + ["outcome"]
cmap = ListedColormap(sns.color_palette("Set1", n_colors=2).as_hex())  # Palette pour la variable cible
scatter_matrix(df_scaled[vars_to_plot], figsize=(10, 8), diagonal='hist', alpha=0.8,
               c=df_scaled["outcome"], cmap=cmap, marker='o', hist_kwds={'bins': 20})
plt.suptitle("Scatter matrix des variables les plus corrélées avec la cible (améliorée)", fontsize=16)
plt.show()

# # Boxplot pour les variables les plus corrélées
# for var in top_corr.index:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x="outcome", y=var, data=df_scaled, palette="Set2")
#     plt.title(f"Boxplot de {var} en fonction de la cible")
#     plt.show()