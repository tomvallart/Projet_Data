def analyse_data():
    print("\n=== Analyse des données ===")
    exec(open("analyse_data.py", encoding="utf-8").read())

def preparation_donnees():
    print("\n=== Préparation des données ===")
    exec(open("preparation_donnees.py", encoding="utf-8").read())

def recherche_correlation():
    print("\n=== Recherche de corrélations ===")
    exec(open("recherche_correlation.py", encoding="utf-8").read())

if __name__ == "__main__":
    analyse_data()
    preparation_donnees()
    recherche_correlation()