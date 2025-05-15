import analyse_data
import preparation_donnees
import recherche_correlation
import comparaison_classifieurs

def run_analyse(file):
    print("\n=== Analyse des données ===")
    analyse_data.main(file)

def run_preparation(file):
    print("\n=== Préparation des données ===")
    return preparation_donnees.main(file)

def run_correlation(file):
    print("\n=== Recherche de corrélations ===")
    recherche_correlation.main(file)

def run_comparaison(file):
    print("\n=== Comparaison des classifieurs ===")
    comparaison_classifieurs.main(file)

if __name__ == "__main__":
    file = input("Entrez le nom du fichier CSV à analyser (ex: car_insurance.csv) : ")
    run_analyse(file)
    file_prepared=run_preparation(file)
    run_correlation(file_prepared)
    run_comparaison(file_prepared)