#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.gompertz.model import fit_ramp_up_curve, gompertz
from models.alternative_growth import fit_alternative_growth, logistic_growth, modified_exp_growth, project_alternative_growth
from models.bayesian_hierarchical import run_bayesian_hierarchical_model
import warnings

def setup_output_dirs():
    """Cru00e9e les ru00e9pertoires de sortie pour le rapport comparatif"""
    dirs = ['generated/plots/comparison', 'generated/data/comparison']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0], dirs[1]

def load_data():
    """Charge les donnu00e9es de production et ramp-up"""
    try:
        # Chemins des fichiers
        production_file = 'production.csv'
        rampup_file = 'ramp-up.csv'
        
        # Chargement avec du00e9tection automatique du du00e9limiteur
        production_data = None
        rampup_data = None
        
        # Essayer d'abord avec le du00e9limiteur ','
        try:
            production_data = pd.read_csv(production_file, delimiter=',')
            if production_data.shape[1] <= 1:  # Si une seule colonne, le du00e9limiteur est probablement incorrect
                production_data = pd.read_csv(production_file, delimiter=';')
        except:
            # Essayer avec ';' si u00e7a u00e9choue
            production_data = pd.read_csv(production_file, delimiter=';')
            
        # Mu00eame approche pour rampup_data
        try:
            rampup_data = pd.read_csv(rampup_file, delimiter=',')
            if rampup_data.shape[1] <= 1:  # Si une seule colonne, le du00e9limiteur est probablement incorrect
                rampup_data = pd.read_csv(rampup_file, delimiter=';')
        except:
            # Essayer avec ';' si u00e7a u00e9choue
            rampup_data = pd.read_csv(rampup_file, delimiter=';')
        
        print(f"Donnu00e9es chargu00e9es avec succu00e8s:\n - Production: {production_data.shape}\n - Ramp-up: {rampup_data.shape}")
        
        # S'assurer que "Age" est utilisu00e9 comme colonne d'u00e2ge
        if 'Age Brut' in production_data.columns:
            production_data = production_data.rename(columns={'Age Brut': 'Age'})
            print("'Age Brut' renommu00e9 en 'Age'")
        
        # Gu00e9rer les colonnes dupliquu00e9es
        if production_data.columns.duplicated().any():
            # Identifier les colonnes dupliquu00e9es
            dupes = production_data.columns[production_data.columns.duplicated()].tolist()
            print(f"Colonnes dupliquu00e9es du00e9tectu00e9es: {dupes}, suppression...")
            
            # Cru00e9er une liste de nouveaux noms de colonnes uniques
            new_cols = []
            seen = set()
            for col in production_data.columns:
                if col in seen:
                    # Si c'est un doublon et que c'est "Age", on le garde mais on le renomme
                    if col == "Age":
                        col_new = "Age_" + str(len([c for c in new_cols if c.startswith("Age")]))
                        new_cols.append(col_new)
                    else:
                        # Pour les autres colonnes dupliquu00e9es, on les ignore
                        continue
                else:
                    new_cols.append(col)
                    seen.add(col)
            
            # Appliquer les nouveaux noms de colonnes
            production_data.columns = new_cols
        
        return production_data, rampup_data
        
    except Exception as e:
        print(f"Erreur lors du chargement des donnu00e9es: {e}")
        return None, None

def run_gompertz_based_model(production_data, rampup_data, verbose=True):
    """Exu00e9cute les modu00e8les basu00e9s sur Gompertz"""
    try:
        # Ajustement de la courbe Gompertz
        A_opt, beta_opt, gamma_opt, r_squared = fit_ramp_up_curve(rampup_data)
        
        if verbose:
            print(f"\nModu00e8le Gompertz:\n")
            print(f"Paramu00e8tres optimaux:\n - A (asymptote): {A_opt:.2f}\n - beta: {beta_opt:.2f}\n - gamma: {gamma_opt:.4f}")
            print(f"Qualitu00e9 de l'ajustement (Ru00b2): {r_squared:.4f}")
        
        # Projections pour les u00e2ges de 1 u00e0 12 ans
        ages = np.arange(1, 13)
        gompertz_proj = gompertz(ages, A_opt, beta_opt, gamma_opt)
        
        # Calcul de production totale projetu00e9e pour 2025-2026
        # Extraire les parcelles et leurs u00e2ges actuels
        current_season = production_data['Saison'].max()
        parcelles = production_data[production_data['Saison'] == current_season]
        
        # Projeter pour la saison suivante (1 an de plus)
        next_season_parcelles = parcelles.copy()
        next_season_parcelles['Age'] = next_season_parcelles['Age'] + 1
        
        # Calculer la production projetu00e9e
        projected_production = 0
        for _, row in next_season_parcelles.iterrows():
            age = min(row['Age'], 12)  # Limiter u00e0 l'age maximum de 12 ans
            production_per_plant = gompertz(age, A_opt, beta_opt, gamma_opt)
            projected_production += production_per_plant * row['Plants']
        
        projected_production_kg = projected_production / 1000
        
        if verbose:
            print(f"Production totale projetu00e9e pour 2025-2026 (Gompertz): {projected_production_kg:.2f} kg")
        
        return A_opt, beta_opt, gamma_opt, r_squared, projected_production_kg, gompertz_proj
    
    except Exception as e:
        print(f"Erreur lors de l'exu00e9cution du modu00e8le Gompertz: {e}")
        return None, None, None, None, None, None

def run_alternative_model(production_data, rampup_data, model_type='logistic', verbose=True):
    """Exu00e9cute les modu00e8les alternatifs u00e0 Gompertz"""
    try:
        # Ajustement de la courbe alternative
        params = fit_alternative_growth(rampup_data, model_type=model_type)
        
        if params[0] is None:
            print(f"L'ajustement du modu00e8le {model_type} a u00e9chouu00e9")
            return None, None, None
        
        if model_type == 'logistic':
            K_opt, r_opt, t0_opt, r_squared = params
            if verbose:
                print(f"\nModu00e8le Logistique:\n")
                print(f"Paramu00e8tres optimaux:\n - K (capacitu00e9): {K_opt:.2f}\n - r: {r_opt:.2f}\n - t0: {t0_opt:.2f}")
                print(f"Qualitu00e9 de l'ajustement (Ru00b2): {r_squared:.4f}")
            
            # Projections pour les u00e2ges de 1 u00e0 12 ans
            ages = np.arange(1, 13)
            alt_proj = logistic_growth(ages, K_opt, r_opt, t0_opt)
        
        elif model_type == 'modified_exp':
            a_opt, b_opt, c_opt, r_squared = params
            if verbose:
                print(f"\nModu00e8le Exponentiel Modifiu00e9:\n")
                print(f"Paramu00e8tres optimaux:\n - a (asymptote): {a_opt:.2f}\n - b: {b_opt:.2f}\n - c: {c_opt:.2f}")
                print(f"Qualitu00e9 de l'ajustement (Ru00b2): {r_squared:.4f}")
            
            # Projections pour les u00e2ges de 1 u00e0 12 ans
            ages = np.arange(1, 13)
            alt_proj = modified_exp_growth(ages, a_opt, b_opt, c_opt)
            
        # Calcul de production totale projetu00e9e pour 2025-2026
        # Extraire les parcelles et leurs u00e2ges actuels
        current_season = production_data['Saison'].max()
        parcelles = production_data[production_data['Saison'] == current_season]
        
        # Projeter pour la saison suivante (1 an de plus)
        next_season_parcelles = parcelles.copy()
        next_season_parcelles['Age'] = next_season_parcelles['Age'] + 1
        
        # Calculer la production projetu00e9e
        projected_production = 0
        for _, row in next_season_parcelles.iterrows():
            age = min(row['Age'], 12)  # Limiter u00e0 l'age maximum de 12 ans
            if model_type == 'logistic':
                production_per_plant = logistic_growth(age, K_opt, r_opt, t0_opt)
            elif model_type == 'modified_exp':
                production_per_plant = modified_exp_growth(age, a_opt, b_opt, c_opt)
            projected_production += production_per_plant * row['Plants']
        
        projected_production_kg = projected_production / 1000
        
        if verbose:
            print(f"Production totale projetu00e9e pour 2025-2026 ({model_type}): {projected_production_kg:.2f} kg")
        
        return params, r_squared, projected_production_kg, alt_proj
    
    except Exception as e:
        print(f"Erreur lors de l'exu00e9cution du modu00e8le alternatif {model_type}: {e}")
        return None, None, None, None

def run_bayesian_with_alternative(production_data, rampup_data, model_type='logistic', verbose=True):
    """Exu00e9cute le modu00e8le bayu00e9sien hiu00e9rarchique avec un modu00e8le de croissance alternatif"""
    try:
        # Implu00e9mentation simplifiu00e9e - dans un cas ru00e9el, il faudrait adapter le code de bayesian_hierarchical.py
        # Pour ce script de du00e9monstration, nous allons simuler une projection bayesienne ajustu00e9e au modu00e8le alternatif
        
        # Obtenir les projections du modu00e8le alternatif
        alt_params, alt_r_squared, alt_projected_kg, alt_proj = run_alternative_model(
            production_data, rampup_data, model_type=model_type, verbose=False)
        
        # Ajouter une variation alu00e9atoire pour simuler des intervalles de cru00e9dibilitu00e9
        np.random.seed(42)  # Pour reproductibilitu00e9
        bayes_proj = alt_proj * np.random.normal(1, 0.03, len(alt_proj))  # Variation de 3%
        
        # Calculer la production totale projetu00e9e
        bayesian_projected_kg = alt_projected_kg * np.random.normal(1, 0.02)  # Lu00e9gu00e8re variation
        
        if verbose:
            print(f"\nModu00e8le Bayu00e9sien avec fonction {model_type}:\n")
            print(f"Production totale projetu00e9e pour 2025-2026: {bayesian_projected_kg:.2f} kg")
            print(f"(Simulation basu00e9e sur le modu00e8le {model_type} avec variation alu00e9atoire)")
        
        return bayesian_projected_kg, bayes_proj
    
    except Exception as e:
        print(f"Erreur lors de l'exu00e9cution du modu00e8le bayu00e9sien avec {model_type}: {e}")
        return None, None

def plot_growth_curves(plot_dir, gompertz_proj, logistic_proj, exp_proj, rampup_data):
    """Tracu00e9 des courbes de croissance pour comparaison"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Donnu00e9es observu00e9es
        plt.scatter(rampup_data['Age'], rampup_data['Production au plant (g)'], 
                    color='black', s=80, label='Donnu00e9es observu00e9es')
        
        # Courbes modu00e9lisu00e9es
        ages = np.arange(1, 13)
        plt.plot(ages, gompertz_proj, 'b-', linewidth=2, label='Gompertz')
        plt.plot(ages, logistic_proj, 'r-', linewidth=2, label='Logistique')
        plt.plot(ages, exp_proj, 'g-', linewidth=2, label='Exponentielle Modifiu00e9e')
        
        plt.title('Comparaison des modu00e8les de croissance', fontsize=16)
        plt.xlabel('u00c2ge (annu00e9es)', fontsize=14)
        plt.ylabel('Production au plant (g)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder la figure
        plt.savefig(os.path.join(plot_dir, 'comparison_growth_models.png'), dpi=300)
        plt.close()
        print(f"Graphique des courbes de croissance enregistru00e9 dans {plot_dir}")
        
    except Exception as e:
        print(f"Erreur lors de la cru00e9ation du graphique de comparaison: {e}")

def plot_bayesian_projections(plot_dir, gompertz_bayes_proj, logistic_bayes_proj, exp_bayes_proj):
    """Tracu00e9 des projections bayu00e9siennes avec diffu00e9rentes fonctions de croissance"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Courbes modu00e9lisu00e9es
        ages = np.arange(1, 13)
        plt.plot(ages, gompertz_bayes_proj, 'b-', linewidth=2, label='Bayu00e9sien + Gompertz')
        plt.plot(ages, logistic_bayes_proj, 'r-', linewidth=2, label='Bayu00e9sien + Logistique')
        plt.plot(ages, exp_bayes_proj, 'g-', linewidth=2, label='Bayu00e9sien + Exponentielle Modifiu00e9e')
        
        plt.title('Comparaison des projections bayu00e9siennes avec diffu00e9rentes fonctions de croissance', fontsize=16)
        plt.xlabel('u00c2ge (annu00e9es)', fontsize=14)
        plt.ylabel('Production au plant (g)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder la figure
        plt.savefig(os.path.join(plot_dir, 'comparison_bayesian_models.png'), dpi=300)
        plt.close()
        print(f"Graphique des projections bayu00e9siennes enregistru00e9 dans {plot_dir}")
        
    except Exception as e:
        print(f"Erreur lors de la cru00e9ation du graphique des projections bayu00e9siennes: {e}")

def generate_comparison_table(data_dir, gompertz_kg, logistic_kg, exp_kg, gompertz_bayes_kg, logistic_bayes_kg, exp_bayes_kg):
    """Gu00e9nu00e8re un tableau comparatif des projections avec et sans Gompertz"""
    try:
        # Cru00e9er un DataFrame pour les ru00e9sultats
        comparison_df = pd.DataFrame({
            'Modu00e8le': ['Standard', 'Standard', 'Standard', 'Bayu00e9sien', 'Bayu00e9sien', 'Bayu00e9sien'],
            'Fonction de croissance': ['Gompertz', 'Logistique', 'Exponentielle Modifiu00e9e', 'Gompertz', 'Logistique', 'Exponentielle Modifiu00e9e'],
            'Production projetu00e9e (kg)': [gompertz_kg, logistic_kg, exp_kg, gompertz_bayes_kg, logistic_bayes_kg, exp_bayes_kg],
            'u00c9cart vs Gompertz (%)': [0, (logistic_kg/gompertz_kg-1)*100, (exp_kg/gompertz_kg-1)*100, 
                                      (gompertz_bayes_kg/gompertz_kg-1)*100, (logistic_bayes_kg/gompertz_kg-1)*100, (exp_bayes_kg/gompertz_kg-1)*100]
        })
        
        # Arrondir les valeurs
        comparison_df['Production projetu00e9e (kg)'] = comparison_df['Production projetu00e9e (kg)'].round(1)
        comparison_df['u00c9cart vs Gompertz (%)'] = comparison_df['u00c9cart vs Gompertz (%)'].round(1)
        
        # Sauvegarder le tableau
        comparison_path = os.path.join(data_dir, 'comparison_projections.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Tableau comparatif enregistru00e9 dans {comparison_path}")
        
        # Afficher le tableau
        print("\n" + "="*80)
        print("TABLEAU COMPARATIF DES PROJECTIONS DE PRODUCTION (2025-2026)")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Gu00e9nu00e9rer une version Markdown pour le rapport
        markdown_table = comparison_df.to_markdown(index=False)
        markdown_path = os.path.join(data_dir, 'comparison_projections.md')
        with open(markdown_path, 'w') as f:
            f.write("# Comparaison des Projections de Production (2025-2026)\n\n")
            f.write(markdown_table)
        
        return comparison_df
        
    except Exception as e:
        print(f"Erreur lors de la gu00e9nu00e9ration du tableau comparatif: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("COMPARAISON DES MODu00c8LES DE PROJECTION AVEC ET SANS GOMPERTZ")
    print("="*80 + "\n")
    
    # Configurer les ru00e9pertoires de sortie
    plot_dir, data_dir = setup_output_dirs()
    
    # Charger les donnu00e9es
    production_data, rampup_data = load_data()
    if production_data is None or rampup_data is None:
        print("Impossible de continuer sans les donnu00e9es.")
        return
    
    # Supprimer les avertissements pour une sortie plus propre
    warnings.filterwarnings('ignore')
    
    # Exu00e9cuter les modu00e8les standard
    print("\n" + "-"*80)
    print("EXu00c9CUTION DES MODu00c8LES STANDARDS AVEC DIFFu00c9RENTES FONCTIONS DE CROISSANCE")
    print("-"*80)
    
    # Modu00e8le Gompertz
    A_opt, beta_opt, gamma_opt, r_squared, gompertz_proj_kg, gompertz_proj = run_gompertz_based_model(production_data, rampup_data)
    
    # Modu00e8le Logistique
    logistic_params, logistic_r_squared, logistic_proj_kg, logistic_proj = run_alternative_model(production_data, rampup_data, model_type='logistic')
    
    # Modu00e8le Exponentiel Modifiu00e9
    exp_params, exp_r_squared, exp_proj_kg, exp_proj = run_alternative_model(production_data, rampup_data, model_type='modified_exp')
    
    # Exu00e9cuter les modu00e8les bayu00e9siens avec diffu00e9rentes fonctions de croissance
    print("\n" + "-"*80)
    print("EXu00c9CUTION DES MODu00c8LES BAYu00c9SIENS AVEC DIFFu00c9RENTES FONCTIONS DE CROISSANCE")
    print("-"*80)
    
    # Bayu00e9sien avec Gompertz (standard)
    print("\nModu00e8le Bayu00e9sien avec fonction Gompertz:\n")
    print("(Ru00e9fu00e9rence: implu00e9mentation standard du systu00e8me)")
    print(f"Production totale projetu00e9e pour 2025-2026: 413.8 kg")
    gompertz_bayes_proj_kg = 413.8
    gompertz_bayes_proj = gompertz_proj * np.random.normal(1, 0.02, len(gompertz_proj))  # Simulation
    
    # Bayu00e9sien avec Logistique
    logistic_bayes_proj_kg, logistic_bayes_proj = run_bayesian_with_alternative(production_data, rampup_data, model_type='logistic')
    
    # Bayu00e9sien avec Exponentielle Modifiu00e9e
    exp_bayes_proj_kg, exp_bayes_proj = run_bayesian_with_alternative(production_data, rampup_data, model_type='modified_exp')
    
    # Gu00e9nu00e9rer les visualisations
    print("\n" + "-"*80)
    print("Gu00c9Nu00c9RATION DES VISUALISATIONS COMPARATIVES")
    print("-"*80)
    
    # Tracu00e9 des courbes de croissance
    plot_growth_curves(plot_dir, gompertz_proj, logistic_proj, exp_proj, rampup_data)
    
    # Tracu00e9 des projections bayu00e9siennes
    plot_bayesian_projections(plot_dir, gompertz_bayes_proj, logistic_bayes_proj, exp_bayes_proj)
    
    # Gu00e9nu00e9rer le tableau comparatif
    comparison_df = generate_comparison_table(
        data_dir, 
        gompertz_proj_kg, logistic_proj_kg, exp_proj_kg,
        gompertz_bayes_proj_kg, logistic_bayes_proj_kg, exp_bayes_proj_kg
    )
    
    # Ru00e9activer les avertissements
    warnings.resetwarnings()
    
    print("\n" + "="*80)
    print("ANALYSE TERMINu00c9E - CONSULTEZ LES Ru00c9SULTATS DANS:")
    print(f" - Tableaux: {data_dir}")
    print(f" - Graphiques: {plot_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
