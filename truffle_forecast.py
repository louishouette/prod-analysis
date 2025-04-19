#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Orchestrateur principal pour l'analyse et la prévision de production de truffes.
Ce script intègre plusieurs méthodes statistiques pour générer des prévisions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Ajouter les répertoires au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les modules utilitaires
from models.utils import load_data, prepare_data, setup_directories

# Importer les modèles statistiques
from models.gompertz.model import (
    gompertz, fit_ramp_up_curve, build_bayesian_model,
    sample_posterior, project_future_production
)
from models.state_space.model import (
    build_state_space_model, project_state_space_production
)
from models.linear_trend.model import (
    build_linear_trend_model, project_linear_trend, prepare_time_series_data
)
from models.exponential_smoothing.model import (
    build_exponential_smoothing_model, project_exponential_smoothing
)
from models.holts_method.model import (
    build_holts_trend_model, project_holts_trend
)

# Aliases pour compatibilité avec le pipeline principal
run_linear_trend_model = build_linear_trend_model
run_exponential_smoothing_model = build_exponential_smoothing_model
run_holts_trend_model = build_holts_trend_model


def run_gompertz_model(data, rampup_data, next_season=2025, max_age=12):
    """
    Exécute le modèle Gompertz pour projeter la production future.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    rampup_data : DataFrame avec la courbe de montée en production
    next_season : année de la prochaine saison à prévoir
    max_age : âge maximum pour les projections
    
    Retourne:
    ---------
    projections, paramètres du modèle
    """
    print("\n" + "=" * 80)
    print("Exécution du modèle de croissance Gompertz")
    print("=" * 80)
    
    # Ajuster la courbe de ramp-up pour obtenir les paramètres fixes
    A_fixed, beta_fixed, gamma_fixed, r_squared = fit_ramp_up_curve(rampup_data)
    
    print(f"\nParamètres d'effet fixe de la courbe de ramp-up:")
    print(f"A (asymptote): {A_fixed:.2f}")
    print(f"beta: {beta_fixed:.4f}")
    print(f"gamma: {gamma_fixed:.4f}")
    print(f"R-carré: {r_squared:.4f}")
    
    # Construire le modèle bayésien hiérarchique
    print("\nConstruction du modèle bayésien hiérarchique...")
    hierarchical_model, model_data, parcels = build_bayesian_model(data, gamma_fixed, A_fixed, beta_fixed)
    
    # Échantillonner la distribution postérieure
    trace = sample_posterior(hierarchical_model, chains=2, tune=1000, draws=1000)
    
    # Projeter la production future
    print("\nProjection de la production future avec le modèle Gompertz...")
    gompertz_projections = project_future_production(trace, model_data, parcels, gamma_fixed, max_age)
    print(f"[DEBUG] gompertz_projections index: unique={gompertz_projections.index.is_unique}, duplicated={gompertz_projections.index.duplicated().sum()}")
    if not gompertz_projections.index.is_unique:
        print("[DEBUG] Correction d'index dupliqué sur gompertz_projections...")
        gompertz_projections = gompertz_projections.reset_index(drop=True)
        gompertz_projections.index += 1  # Pour garder l'âge comme index (si pertinent)
    
    # Sauvegarder les projections
    os.makedirs('generated/data/projections', exist_ok=True)
    gompertz_projections.to_csv('generated/data/projections/gompertz_projected_production.csv')
    print("Projections Gompertz sauvegardées dans 'generated/data/projections/gompertz_projected_production.csv'")
    
    # Calcul des ratios par rapport à la courbe cible
    gompertz_ratio_df = gompertz_projections.copy()
    print(f"[DEBUG] gompertz_ratio_df index: unique={gompertz_ratio_df.index.is_unique}, duplicated={gompertz_ratio_df.index.duplicated().sum()}")
    if not gompertz_ratio_df.index.is_unique:
        print("[DEBUG] Correction d'index dupliqué sur gompertz_ratio_df...")
        gompertz_ratio_df = gompertz_ratio_df.reset_index(drop=True)
        gompertz_ratio_df.index += 1
    for age in gompertz_ratio_df.index:
        target_value = gompertz(age, A_fixed, beta_fixed, gamma_fixed)
        gompertz_ratio_df.loc[age] = gompertz_ratio_df.loc[age] / target_value if target_value > 0 else np.nan
    
    # Sauvegarder les ratios
    gompertz_ratio_df.to_csv('generated/data/projections/gompertz_production_ratios.csv')
    print("Ratios de production Gompertz sauvegardés dans 'generated/data/projections/gompertz_production_ratios.csv'")
    
    # Visualiser les projections
    os.makedirs('generated/plots/production_projections', exist_ok=True)
    plt.figure(figsize=(14, 10))
    gompertz_projections.plot(marker='o', linestyle='-')
    
    # Ajouter la courbe cible
    ages = np.arange(1, 13)
    target = [gompertz(age, A_fixed, beta_fixed, gamma_fixed) for age in ages]
    plt.plot(ages, target, 'k--', linewidth=2, label='Courbe Cible')
    
    plt.title('Modèle Gompertz: Production Projetée par Parcelle et Âge')
    plt.xlabel('Âge des Arbres (années)')
    plt.ylabel('Production par Plant (g)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/gompertz_projections.png', dpi=300, bbox_inches='tight')
    
    # Visualiser les ratios
    plt.figure(figsize=(14, 10))
    gompertz_ratio_df.plot(marker='o', linestyle='-')
    plt.axhline(y=1.0, color='k', linestyle='--', label='Ratio Cible (1.0)')
    plt.title('Modèle Gompertz: Ratios de Production par Rapport à la Courbe Cible')
    plt.xlabel('Âge des Arbres (années)')
    plt.ylabel('Ratio de Production (Réel/Cible)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/gompertz_ratios.png', dpi=300, bbox_inches='tight')
    
    return gompertz_projections, (A_fixed, beta_fixed, gamma_fixed, r_squared)


def run_state_space_model(data, rampup_data, gompertz_params, next_season=2025):
    """
    Exécute le modèle à espace d'états pour projeter la production future.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    rampup_data : DataFrame avec la courbe de montée en production
    gompertz_params : tuple avec les paramètres Gompertz (A_fixed, beta_fixed, gamma_fixed, r_squared)
    next_season : année de la prochaine saison à prévoir
    
    Retourne:
    ---------
    projections, données de prévision pour la prochaine saison
    """
    print("\n" + "=" * 80)
    print("Exécution du modèle hiérarchique bayésien à espace d'états")
    print("=" * 80)
    
    A_fixed, beta_fixed, gamma_fixed, _ = gompertz_params
    
    # Construire et ajuster le modèle à espace d'états
    parcel_species_models, parcel_species_forecasts, prod_time_series = build_state_space_model(
        data, rampup_data, gamma_fixed, A_fixed, beta_fixed)
    
    # Projeter la production future
    print("\nProjection de la production future avec le modèle à espace d'états...")
    ss_projections, next_season_data = project_state_space_production(
        parcel_species_forecasts, data, gamma_fixed, A_fixed, beta_fixed, next_season)
    
    # Sauvegarder les projections
    ss_projections.to_csv('generated/data/projections/state_space_projected_production.csv')
    print("Projections du modèle à espace d'états sauvegardées dans 'generated/data/projections/state_space_projected_production.csv'")
    
    # Sauvegarder les données détaillées pour la prochaine saison
    next_season_data.to_csv('generated/data/projections/next_season_detailed_forecast.csv', index=False)
    print("Prévisions détaillées pour la prochaine saison sauvegardées dans 'generated/data/projections/next_season_detailed_forecast.csv'")
    
    # Visualiser les projections
    plt.figure(figsize=(14, 10))
    ss_projections.plot(marker='o', linestyle='-')
    
    # Ajouter la courbe cible
    ages_extended = np.arange(1, ss_projections.index.max() + 1)
    target_curve = [gompertz(age, A_fixed, beta_fixed, gamma_fixed) for age in ages_extended]
    plt.plot(ages_extended, target_curve, 'k--', linewidth=2, label='Courbe Cible')
    
    plt.title('Modèle à Espace d\'États: Production Projetée par Parcelle et Âge')
    plt.xlabel('Âge des Arbres (années)')
    plt.ylabel('Production par Plant (g)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/state_space_projections.png', dpi=300, bbox_inches='tight')
    
    # Calcul des ratios par rapport à la courbe cible
    ss_ratio_df = ss_projections.copy()
    for age in ss_ratio_df.index:
        target_value = gompertz(age, A_fixed, beta_fixed, gamma_fixed)
        ss_ratio_df.loc[age] = ss_ratio_df.loc[age] / target_value if target_value > 0 else np.nan
    
    # Sauvegarder les ratios
    ss_ratio_df.to_csv('generated/data/projections/state_space_production_ratios.csv')
    print("Ratios de production du modèle à espace d'états sauvegardés dans 'generated/data/projections/state_space_production_ratios.csv'")
    
    # Visualiser les ratios
    plt.figure(figsize=(14, 10))
    ss_ratio_df.plot(marker='o', linestyle='-')
    plt.axhline(y=1.0, color='k', linestyle='--', label='Ratio Cible (1.0)')
    plt.title('Modèle à Espace d\'États: Ratios de Production par Rapport à la Courbe Cible')
    plt.xlabel('Âge des Arbres (années)')
    plt.ylabel('Ratio de Production (Réel/Cible)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/state_space_ratios.png', dpi=300, bbox_inches='tight')
    
    return ss_projections, next_season_data


def compare_models(data, gompertz_projections, state_space_projections):
    """
    Compare les résultats des modèles robustes de prévision (Gompertz, Espace d'États).

    Paramètres:
    -----------
    data : DataFrame avec les données de production
    gompertz_projections : DataFrame de projections Gompertz
    state_space_projections : DataFrame de projections à espace d'états

    Retourne:
    ---------
    DataFrame avec les résultats de la comparaison (alignés sur l'âge)
    """
    # Extraire la colonne principale (LP) si elle existe, sinon prendre la dernière colonne
    def extract_main(df):
        if isinstance(df, pd.DataFrame):
            cols = list(df.columns)
            # Si Age n'est pas une colonne mais l'index, le remettre comme colonne
            if 'Age' not in cols and getattr(df.index, 'name', None) == 'Age':
                df = df.reset_index()
                cols = list(df.columns)
            if 'LP' in cols and 'Age' in cols:
                return df[['Age', 'LP']].rename(columns={'LP': 'Gompertz'})
            elif 'Age' in cols:
                # Prendre la dernière colonne numérique
                col = [c for c in cols if c != 'Age'][-1]
                return df[['Age', col]].rename(columns={col: 'Gompertz'})
            else:
                # Si Age n'existe pas, créer à partir de l'index
                df = df.reset_index(drop=True)
                df['Age'] = df.index + 1
                col = df.columns[-2]  # dernière colonne avant Age
                return df[['Age', col]].rename(columns={col: 'Gompertz'})
        return None

    def extract_main_ss(df):
        if isinstance(df, pd.DataFrame):
            cols = list(df.columns)
            if 'Age' not in cols and getattr(df.index, 'name', None) == 'Age':
                df = df.reset_index()
                cols = list(df.columns)
            if 'LP' in cols and 'Age' in cols:
                return df[['Age', 'LP']].rename(columns={'LP': 'Espace d’États'})
            elif 'Age' in cols:
                col = [c for c in cols if c != 'Age'][-1]
                return df[['Age', col]].rename(columns={col: 'Espace d’États'})
            else:
                df = df.reset_index(drop=True)
                df['Age'] = df.index + 1
                col = df.columns[-2]
                return df[['Age', col]].rename(columns={col: 'Espace d’États'})
        return None

    gompertz_df = extract_main(gompertz_projections)
    state_space_df = extract_main_ss(state_space_projections)
    # Fusionner sur Age
    if gompertz_df is not None and state_space_df is not None:
        comparison_df = pd.merge(gompertz_df, state_space_df, on='Age', how='outer')
    else:
        # Fallback : concaténer les valeurs (peu probable)
        comparison_df = pd.DataFrame({'Gompertz': gompertz_projections, 'Espace d’États': state_space_projections})
    return comparison_df



def bayesian_hierarchical_simple(data, rampup_curve):
    """
    Squelette de modèle bayésien hiérarchique simple pour faible historique.
    Ce modèle suppose : production_observée = f(age) × effet_parcelle × bruit
    - f(age) : courbe cible connue (Gompertz ou autre)
    - effet_parcelle : prior centré sur 1, variance faible
    - bruit : normal, faible variance
    
    Ce squelette est à compléter selon vos besoins (PyMC, Stan, etc.).
    """
    # TODO : Implémenter une version PyMC3/PyMC4 ou Stan selon vos préférences
    pass

def main():
    """
    Fonction principale pour exécuter le système de prévision de production de truffes.
    Modèles exécutés :
    - Modèle Gompertz (non-linéaire à effets mixtes)
    - Modèle à Espace d'États
    - Modèle bayésien hiérarchique (par parcelle)
    """
    # Charger les données
    parser = argparse.ArgumentParser(description="Système de prévision de production de truffes")
    parser.add_argument('--data', default='data/production_historique.csv', help="Fichier CSV des données de production")
    parser.add_argument('--rampup', default='data/ramp-up.csv', help="Fichier CSV de la courbe de montée en production")
    parser.add_argument('--forecast_years', type=int, default=3, help="Nombre d'années à prévoir pour les modèles de tendance (défaut: 3)")
    parser.add_argument('--next-season', type=int, default=2025, help="Année de la prochaine saison à prévoir")
    args = parser.parse_args()
    
    # Créer les répertoires nécessaires
    setup_directories()
    
    # Charger les données avec le pipeline d'origine et logs explicites
    try:
        data = load_data(args.data)
        rampup_data = load_data(args.rampup)
        processed_data = prepare_data(data)
        print(f"\nNote importante: Utilisation de l'Âge Brut comme variable fondamentale pour")
        print("grouper les lots et estimer la progression de la production, conformément aux")
        print("conclusions des analyses antérieures qui ont démontré que le facteur de pénalité")
        print("appliqué à l'âge des arbres n'était pas scientifiquement fondé.")
    except Exception as e:
        print(f"Erreur lors du chargement ou de la préparation des données: {str(e)}")
        import sys
        sys.exit(1)
    
    # Variables pour stocker les résultats des modèles
    gompertz_projections = None
    state_space_projections = None
    linear_forecasts = None
    exp_smoothing_forecasts = None
    holts_forecasts = None
    next_season_forecast = None
    
    # Exécuter les modèles sélectionnés
    
    # Exécuter le modèle bayésien hiérarchique
    print("\n=== Modèle Bayésien Hiérarchique ===")
    try:
        from models.bayesian_hierarchical import run_bayesian_hierarchical_model
        summary, proj_df, trace = run_bayesian_hierarchical_model(data, rampup_data, output_dir='generated', max_age=12, verbose=True)
        print("Modèle bayésien hiérarchique exécuté avec succès")
    except Exception as e:
        print(f"Erreur dans le pipeline bayésien : {e}")
    
    # Exécuter le modèle Gompertz
    # Exécution du modèle Gompertz
    gompertz_projections, gompertz_params = run_gompertz_model(
        processed_data, rampup_data, args.next_season)
    
    # Exécuter le modèle à espace d'états
    # Exécution du modèle à espace d'états
    if gompertz_projections is not None:
        state_space_projections, next_season_forecast = run_state_space_model(
            processed_data, rampup_data, gompertz_params, args.next_season)
    else:
        print("\nAttention: Le modèle à espace d'états nécessite les paramètres du modèle Gompertz.")
        print("Exécution du modèle Gompertz d'abord...")
        gompertz_projections, gompertz_params = run_gompertz_model(
            processed_data, rampup_data, args.next_season)
        state_space_projections, next_season_forecast = run_state_space_model(
            processed_data, rampup_data, gompertz_params, args.next_season)
    
    # Exécuter le modèle de tendance linéaire
    # Exécution du modèle linéaire
    linear_model, linear_time_series = run_linear_trend_model(processed_data)
    linear_forecasts, _ = project_linear_trend(linear_model, linear_time_series, args.forecast_years)
    
    # Exécuter le modèle de lissage exponentiel
    # Exécution du modèle de lissage exponentiel
    exp_model, exp_time_series = run_exponential_smoothing_model(processed_data)
    exp_smoothing_forecasts, _ = project_exponential_smoothing(exp_model, exp_time_series, args.forecast_years)
    
    # Exécuter la méthode de Holt
    # Exécution du modèle de Holt
    holts_model, holts_time_series = run_holts_trend_model(processed_data)
    holts_forecasts, _ = project_holts_trend(holts_model, holts_time_series, args.forecast_years)
    
    # Comparer les modèles
    # Comparer les modèles si au moins deux projections sont disponibles
    if len([m for m in [gompertz_projections, state_space_projections, linear_forecasts, 
                        exp_smoothing_forecasts, holts_forecasts] if m is not None]) > 1:
        comparison = compare_models(processed_data, gompertz_projections, state_space_projections)
    
    print("\n" + "=" * 80)
    print("GÉNÉRATION DU RAPPORT DE PRÉVISION")
    print("=" * 80)
    print("\nLes projections sont disponibles dans le répertoire 'generated/data/projections/'")
    print("Les visualisations sont disponibles dans le répertoire 'generated/plots/'")
    
    if next_season_forecast is not None:
        print(f"\nPrévisions pour la saison {args.next_season}-{args.next_season + 1}:")
        print(f"[DEBUG] Colonnes de next_season_forecast: {list(next_season_forecast.columns)}")
        print(f"[DEBUG] Aperçu des données:\n{next_season_forecast.head()}\n")
        # Utilisation explicite de la colonne 'Total_Expected_Production' (projection principale)
        col = 'Total_Expected_Production'
        if col not in next_season_forecast.columns:
            print(f"[ERREUR] La colonne '{col}' n'existe pas dans next_season_forecast !")
            return
        total_production = next_season_forecast[col].sum()
        print(f"Production totale projetée: {total_production:.2f} g")
        print(f"Production totale projetée: {total_production/1000:.2f} kg")
        
        # Top 5 parcelles par production total projetée
        top_parcels = next_season_forecast.groupby('Parcelle')[col].sum().nlargest(5)
        print("\nTop 5 des parcelles par production totale projetée:")
        for parcel, production in top_parcels.items():
            print(f"{parcel}: {production:.2f} g ({production/1000:.2f} kg)")
    

if __name__ == "__main__":
    print("[DEBUG] Script truffle_forecast.py lancé")
    main()
