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
    build_linear_trend_model, project_linear_trend, analyze_parcel_trends
)
from models.exponential_smoothing.model import (
    build_exponential_smoothing_model, project_exponential_smoothing,
    exponential_smoothing_by_parcel
)
from models.holts_method.model import (
    build_holts_trend_model, project_holts_trend, holts_method_by_parcel,
    compare_holts_models
)


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


def run_linear_trend_model(data, forecast_years=3):
    """
    Exécute le modèle de tendance linéaire pour projeter la production future.
    Utilise l'Âge Brut comme variable d'âge.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    forecast_years : nombre d'années à prévoir
    
    Retourne:
    ---------
    résultats du modèle, prévisions
    """
    print("\n" + "=" * 80)
    print("Exécution du modèle de tendance linéaire")
    print("=" * 80)
    
    # Construire le modèle de tendance linéaire
    model_results, time_series = build_linear_trend_model(data)
    
    # Projeter la production future
    combined_data, forecasts = project_linear_trend(model_results, time_series, forecast_years)
    
    # Analyser les tendances par parcelle
    print("\nAnalyse des tendances par parcelle...")
    parcel_trends = analyze_parcel_trends(data)
    
    if not parcel_trends.empty:
        # Sauvegarder les tendances par parcelle
        parcel_trends.to_csv('generated/data/projections/linear_trend_by_parcel.csv', index=False)
        print("Tendances linéaires par parcelle sauvegardées dans 'generated/data/projections/linear_trend_by_parcel.csv'")
        
        # Afficher les parcelles avec les tendances les plus positives et négatives
        print("\nParcelles avec les tendances les plus positives:")
        print(parcel_trends.head(5))
        
        print("\nParcelles avec les tendances les plus négatives:")
        print(parcel_trends.tail(5))
    
    # Sauvegarder les prévisions
    forecasts.to_csv('generated/data/projections/linear_trend_forecasts.csv')
    print("Prévisions de tendance linéaire sauvegardées dans 'generated/data/projections/linear_trend_forecasts.csv'")
    
    # Visualiser les prévisions
    os.makedirs('generated/plots/linear_trend', exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Tracer les données historiques
    historical = combined_data[combined_data['Type'] == 'Historical']
    plt.plot(historical.index, historical['Production au plant (g)'], 'o-', label='Données historiques')
    
    # Tracer les prévisions
    forecast_data = combined_data[combined_data['Type'] == 'Forecast']
    plt.plot(forecast_data.index, forecast_data['Production au plant (g)'], 'r^-', label='Prévisions')
    
    # Tracer les intervalles de confiance
    if 'lower' in forecasts.columns and 'upper' in forecasts.columns:
        plt.fill_between(forecasts.index, forecasts['lower'], forecasts['upper'], 
                         color='r', alpha=0.1, label='Intervalle de confiance 95%')
    
    plt.title('Prévision de Production par Tendance Linéaire')
    plt.xlabel('Année')
    plt.ylabel('Production Moyenne par Plant (g)')
    plt.legend()
    plt.grid(True)
    plt.xticks(combined_data.index)
    plt.tight_layout()
    plt.savefig('generated/plots/linear_trend/linear_forecast.png', dpi=300, bbox_inches='tight')
    
    return model_results, forecasts


def run_exponential_smoothing_model(data, forecast_periods=3, alpha=None):
    """
    Exécute le modèle de lissage exponentiel simple pour projeter la production future.
    Utilise l'Âge Brut comme variable d'âge.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    forecast_periods : nombre de périodes à prévoir
    alpha : paramètre de lissage (None pour estimation automatique)
    
    Retourne:
    ---------
    résultats du modèle, prévisions
    """
    print("\n" + "=" * 80)
    print("Exécution du modèle de lissage exponentiel simple")
    print("=" * 80)
    
    # Construire le modèle de lissage exponentiel
    model_results, time_series = build_exponential_smoothing_model(data, alpha)
    
    # Projeter la production future
    combined_data, forecasts = project_exponential_smoothing(model_results, time_series, forecast_periods)
    
    # Appliquer le lissage exponentiel par parcelle
    print("\nApplication du lissage exponentiel par parcelle...")
    parcel_forecasts = exponential_smoothing_by_parcel(data, forecast_periods)
    
    # Sauvegarder les prévisions par parcelle
    if not parcel_forecasts.empty:
        parcel_forecasts.to_csv('generated/data/projections/exp_smoothing_by_parcel.csv', index=False)
        print("Prévisions par lissage exponentiel par parcelle sauvegardées dans 'generated/data/projections/exp_smoothing_by_parcel.csv'")
    
    # Sauvegarder les prévisions globales
    forecasts.to_csv('generated/data/projections/exp_smoothing_forecasts.csv')
    print("Prévisions par lissage exponentiel sauvegardées dans 'generated/data/projections/exp_smoothing_forecasts.csv'")
    
    # Visualiser les prévisions
    os.makedirs('generated/plots/exponential_smoothing', exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Tracer les données historiques et les valeurs ajustées
    historical = combined_data[combined_data['Type'] == 'Historical']
    plt.plot(historical.index, historical['Production au plant (g)'], 'o-', label='Données historiques')
    plt.plot(historical.index, model_results.fittedvalues, 'g--', label='Valeurs ajustées')
    
    # Tracer les prévisions
    forecast_data = combined_data[combined_data['Type'] == 'Forecast']
    plt.plot(forecast_data.index, forecast_data['Production au plant (g)'], 'r^-', label='Prévisions')
    
    # Tracer les intervalles de confiance
    if 'lower' in forecasts.columns and 'upper' in forecasts.columns:
        plt.fill_between(forecasts.index, forecasts['lower'], forecasts['upper'], 
                         color='r', alpha=0.1, label='Intervalle de confiance 95%')
    
    plt.title('Prévision de Production par Lissage Exponentiel Simple')
    plt.xlabel('Année')
    plt.ylabel('Production Moyenne par Plant (g)')
    plt.legend()
    plt.grid(True)
    plt.xticks(combined_data.index)
    plt.tight_layout()
    plt.savefig('generated/plots/exponential_smoothing/exp_smoothing_forecast.png', dpi=300, bbox_inches='tight')
    
    return model_results, forecasts


def run_holts_trend_model(data, forecast_periods=3):
    """
    Exécute le modèle de Holt (tendance linéaire) pour projeter la production future.
    Utilise l'Âge Brut comme variable d'âge.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    forecast_periods : nombre de périodes à prévoir
    
    Retourne:
    ---------
    résultats du modèle, prévisions
    """
    print("\n" + "=" * 80)
    print("Exécution du modèle de Holt (tendance linéaire)")
    print("=" * 80)
    
    # Comparer les modèles de Holt linéaire et exponentiel
    model_results, time_series, best_model = compare_holts_models(data, forecast_periods)
    
    # Projeter la production future
    combined_data, forecasts = project_holts_trend(model_results, time_series, forecast_periods)
    
    # Appliquer la méthode de Holt par parcelle
    print("\nApplication de la méthode de Holt par parcelle...")
    exponential = (best_model == "exponentiel")
    parcel_forecasts = holts_method_by_parcel(data, forecast_periods, exponential)
    
    # Sauvegarder les prévisions par parcelle
    if not parcel_forecasts.empty:
        parcel_forecasts.to_csv('generated/data/projections/holts_method_by_parcel.csv', index=False)
        print("Prévisions par méthode de Holt par parcelle sauvegardées dans 'generated/data/projections/holts_method_by_parcel.csv'")
    
    # Sauvegarder les prévisions globales
    forecasts.to_csv('generated/data/projections/holts_method_forecasts.csv')
    print("Prévisions par méthode de Holt sauvegardées dans 'generated/data/projections/holts_method_forecasts.csv'")
    
    # Visualiser les prévisions
    os.makedirs('generated/plots/holts_method', exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Tracer les données historiques et les valeurs ajustées
    historical = combined_data[combined_data['Type'] == 'Historical']
    plt.plot(historical.index, historical['Production au plant (g)'], 'o-', label='Données historiques')
    plt.plot(historical.index, model_results.fittedvalues, 'g--', label='Valeurs ajustées')
    
    # Tracer les prévisions
    forecast_data = combined_data[combined_data['Type'] == 'Forecast']
    plt.plot(forecast_data.index, forecast_data['Production au plant (g)'], 'r^-', label='Prévisions')
    
    # Tracer les intervalles de confiance
    if 'lower' in forecasts.columns and 'upper' in forecasts.columns:
        plt.fill_between(forecasts.index, forecasts['lower'], forecasts['upper'], 
                         color='r', alpha=0.1, label='Intervalle de confiance 95%')
    
    trend_type = "exponentielle" if exponential else "linéaire"
    plt.title(f'Prévision de Production par Méthode de Holt (tendance {trend_type})')
    plt.xlabel('Année')
    plt.ylabel('Production Moyenne par Plant (g)')
    plt.legend()
    plt.grid(True)
    plt.xticks(combined_data.index)
    plt.tight_layout()
    plt.savefig('generated/plots/holts_method/holts_forecast.png', dpi=300, bbox_inches='tight')
    
    return model_results, forecasts


def compare_models(data, gompertz_projections, state_space_projections, linear_forecasts, 
                 exp_smoothing_forecasts, holts_forecasts):
    """
    Compare les résultats des différents modèles de prévision.
    
    Paramètres:
    -----------
    data : DataFrame avec les données de production
    gompertz_projections : projections du modèle Gompertz
    state_space_projections : projections du modèle à espace d'états
    linear_forecasts : prévisions du modèle de tendance linéaire
    exp_smoothing_forecasts : prévisions du modèle de lissage exponentiel
    holts_forecasts : prévisions de la méthode de Holt
    
    Retourne:
    ---------
    DataFrame avec les résultats de la comparaison
    """
    print("\n" + "=" * 80)
    print("Comparaison des modèles de prévision")
    print("=" * 80)
    
    # Créer un DataFrame pour stocker les résultats de la comparaison
    model_comparison = pd.DataFrame()
    
    # Extraire les données historiques pour validation
    # Utiliser l'Âge Brut (sans pénalité) comme recommandé dans les analyses antérieures
    historical_data = data[
        (~data['Production au plant (g)'].isna()) & 
        (data['Production au plant (g)'] > 0) &
        (data['Age'] > 0)
    ].copy()
    
    # Calculer les statistiques de comparaison pour chaque modèle
    models = {
        'Gompertz': gompertz_projections,
        'Espace d\'États': state_space_projections,
        'Tendance Linéaire': linear_forecasts,
        'Lissage Exponentiel': exp_smoothing_forecasts,
        'Méthode de Holt': holts_forecasts
    }
    
    # Préparer les données pour la comparaison
    comparison_metrics = []
    
    print("\nCalibration des modèles avec les données historiques...")
    
    # Pour chaque modèle qui a des projections par âge
    for model_name in ['Gompertz', 'Espace d\'États']:
        if model_name in models and models[model_name] is not None and not models[model_name].empty:
            model_df = models[model_name]
            
            # Calculer les erreurs pour chaque âge disponible dans les données historiques
            errors = []
            ages = sorted(historical_data['Age'].unique())
            
            for age in ages:
                if age in model_df.index:
                    actual_values = historical_data[historical_data['Age'] == age]['Production au plant (g)'].values
                    if len(actual_values) > 0:
                        # Pour les modèles basés sur les parcelles, prendre la moyenne des projections
                        predicted_values = model_df.loc[age].mean()
                        errors.append(abs(predicted_values - actual_values.mean()))
            
            if errors:
                mae = np.mean(errors)
                comparison_metrics.append({
                    'Modèle': model_name,
                    'Type': 'Par Âge',
                    'MAE': mae,
                    'RMSE': np.sqrt(np.mean(np.square(errors))),
                    'MAPE': np.mean([abs(e/a)*100 if a > 0 else np.nan for e, a in 
                                    zip(errors, [historical_data[historical_data['Age'] == age]['Production au plant (g)'].mean() 
                                                for age in ages if age in model_df.index])])
                })
    
    # Pour les modèles de séries temporelles
    ts_models = {
        'Tendance Linéaire': linear_forecasts,
        'Lissage Exponentiel': exp_smoothing_forecasts,
        'Méthode de Holt': holts_forecasts
    }
    
    # Vérifier que nous avons assez de données pour valider les modèles de séries temporelles
    if len(historical_data) >= 2:  # Au moins 2 saisons pour valider
        historical_avg = historical_data.groupby('Saison')['Production au plant (g)'].mean()
        
        for model_name, forecast_df in ts_models.items():
            if forecast_df is not None and not forecast_df.empty:
                # Calculer les erreurs pour les saisons historiques disponibles dans les données
                errors = []
                for season in historical_avg.index:
                    if season in forecast_df.index:
                        actual = historical_avg[season]
                        predicted = forecast_df.loc[season, 'forecast']
                        errors.append(abs(predicted - actual))
                
                if errors:
                    mae = np.mean(errors)
                    comparison_metrics.append({
                        'Modèle': model_name,
                        'Type': 'Série Temporelle',
                        'MAE': mae,
                        'RMSE': np.sqrt(np.mean(np.square(errors))),
                        'MAPE': np.mean([abs(e/a)*100 if a > 0 else np.nan for e, a in 
                                        zip(errors, historical_avg.values)])
                    })
    
    # Créer le DataFrame de comparaison
    if comparison_metrics:
        model_comparison = pd.DataFrame(comparison_metrics)
        # Trier par erreur absolue moyenne croissante
        model_comparison = model_comparison.sort_values('MAE')
    
    if not model_comparison.empty:
        # Sauvegarder les métriques de comparaison
        os.makedirs('generated/data/model_comparison', exist_ok=True)
        model_comparison.to_csv('generated/data/model_comparison/model_metrics.csv', index=False)
        print("\nMétriques de comparaison des modèles sauvegardées dans 'generated/data/model_comparison/model_metrics.csv'")
        
        # Afficher les résultats de la comparaison
        print("\nRésultats de la comparaison des modèles (ordonnés par MAE croissante):")
        print(model_comparison.round(2))
        
        # Visualiser les comparaisons
        os.makedirs('generated/plots/model_comparison', exist_ok=True)
        
        # Graphique des MAE
        plt.figure(figsize=(12, 8))
        ax = model_comparison.plot(x='Modèle', y='MAE', kind='bar', legend=False)
        plt.title('Comparaison des Modèles - Erreur Absolue Moyenne (MAE)')
        plt.ylabel('MAE (g/plant)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(model_comparison['MAE']):
            ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
        plt.savefig('generated/plots/model_comparison/mae_comparison.png', dpi=300, bbox_inches='tight')
        
        # Graphique des MAPE
        plt.figure(figsize=(12, 8))
        ax = model_comparison.plot(x='Modèle', y='MAPE', kind='bar', legend=False)
        plt.title('Comparaison des Modèles - Erreur Pourcentage Absolue Moyenne (MAPE)')
        plt.ylabel('MAPE (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(model_comparison['MAPE']):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            
        plt.savefig('generated/plots/model_comparison/mape_comparison.png', dpi=300, bbox_inches='tight')
    
    return model_comparison


def main():
    """
    Fonction principale pour exécuter le système de prévision de production de truffes.
    Orchestration de tous les modèles statistiques et génération des rapports.
    """
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Système de prévision de production de truffes")
    parser.add_argument('--data', default='production.csv', help="Fichier CSV des données de production")
    parser.add_argument('--rampup', default='ramp-up.csv', help="Fichier CSV de la courbe de montée en production")
    parser.add_argument('--next-season', type=int, default=2025, help="Année de la prochaine saison à prévoir")
    parser.add_argument('--forecast-years', type=int, default=3, help="Nombre d'années à prévoir")
    parser.add_argument('--models', default='all', 
                        choices=['all', 'gompertz', 'state_space', 'linear', 'exponential', 'holts'],
                        help="Modèles à exécuter")
    args = parser.parse_args()
    
    print("=" * 80)
    print("SYSTÈME DE PRÉVISION DE PRODUCTION DE TRUFFES")
    print("=" * 80)
    print(f"\nChargement des données depuis {args.data} et {args.rampup}")
    
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
    run_all = args.models == 'all'
    
    # Exécuter le modèle Gompertz
    if run_all or args.models == 'gompertz':
        gompertz_projections, gompertz_params = run_gompertz_model(
            processed_data, rampup_data, args.next_season)
    
    # Exécuter le modèle à espace d'états
    if run_all or args.models == 'state_space':
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
    if run_all or args.models == 'linear':
        linear_model, linear_forecasts = run_linear_trend_model(processed_data, args.forecast_years)
    
    # Exécuter le modèle de lissage exponentiel
    if run_all or args.models == 'exponential':
        exp_model, exp_smoothing_forecasts = run_exponential_smoothing_model(processed_data, args.forecast_years)
    
    # Exécuter la méthode de Holt
    if run_all or args.models == 'holts':
        holts_model, holts_forecasts = run_holts_trend_model(processed_data, args.forecast_years)
    
    # Comparer les modèles
    if run_all or len([m for m in [gompertz_projections, state_space_projections, linear_forecasts, 
                               exp_smoothing_forecasts, holts_forecasts] if m is not None]) > 1:
        comparison = compare_models(
            processed_data, gompertz_projections, state_space_projections,
            linear_forecasts, exp_smoothing_forecasts, holts_forecasts
        )
    
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
    
    print("\nAnalyse terminée.")
    print("=" * 80)


if __name__ == "__main__":
    main()
