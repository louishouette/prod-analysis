#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
import os

# Avec notre nouvelle structure, nous importons directement de linear_trend_model
from models.linear_trend_model import prepare_time_series_data


def build_holts_trend_model(data, alpha=None, beta=None, exponential=False):
    """
    Construit un modèle de Holt (tendance linéaire) pour la production de truffes.
    Utilise l'âge Brut comme variable d'âge.
    
    Paramètres:
    -----------
    data : DataFrame des données de production
    alpha : paramètre de lissage pour le niveau (None pour estimation automatique)
    beta : paramètre de lissage pour la tendance (None pour estimation automatique)
    exponential : si True, utilise une tendance exponentielle plutôt que linéaire
    
    Retourne:
    ---------
    modèle ajusté, données de série temporelle
    """
    print(f"\nConstruction du modèle de Holt (tendance {'exponentielle' if exponential else 'linéaire'})...")
    
    # Préparation des données
    time_series = prepare_time_series_data(data)
    
    # Vérification des données suffisantes
    if len(time_series) < 3:
        print("Erreur: Données insuffisantes pour le modèle de Holt (minimum 3 points).")
        return None, time_series
    
    # Extraction de la série temporelle
    y = time_series['Production au plant (g)'].dropna()
    
    if len(y) < 3:
        print("Erreur: Données valides insuffisantes pour le modèle de Holt.")
        return None, time_series
    
    # Utilisation du damping=False par défaut (non amorti)
    # exponential=True utilise une tendance multiplicative plutôt qu'additive
    model = Holt(y, exponential=exponential, damped_trend=False)
    
    # Ajustement du modèle, avec optimisation automatique si les paramètres ne sont pas spécifiés
    if alpha is None or beta is None:
        fit_model = model.fit(optimized=True)
        print(f"Paramètres optimisés: alpha={fit_model.params['smoothing_level']:.4f}, beta={fit_model.params['smoothing_trend']:.4f}")
    else:
        fit_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        print(f"Paramètres fixés: alpha={alpha:.4f}, beta={beta:.4f}")
    
    # Métriques d'ajustement
    mse = ((fit_model.fittedvalues - y) ** 2).mean()
    rmse = np.sqrt(mse)
    print(f"RMSE (erreur quadratique moyenne): {rmse:.4f}")
    print(f"AIC: {fit_model.aic:.4f}")
    print(f"Type de tendance: {'exponentielle' if exponential else 'linéaire'}")
    
    return fit_model, time_series


def project_holts_trend(model_results, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modèle de Holt.
    
    Paramètres:
    -----------
    model_results : résultats du modèle de Holt
    time_series : série temporelle des données historiques
    forecast_periods : nombre de périodes à prédire
    
    Retourne:
    ---------
    Tuple(DataFrame avec les valeurs historiques et les prédictions, DataFrame des prédictions)
    """
    if model_results is None:
        print("Erreur: Modèle de Holt invalide pour les projections.")
        return None, None
    
    # Génération des prédictions
    try:
        forecast = model_results.forecast(forecast_periods)
        
        # Conversion en DataFrame pour faciliter la manipulation
        last_idx = time_series.index[-1]
        forecast_idx = pd.RangeIndex(start=last_idx + 1, stop=last_idx + forecast_periods + 1)
        forecast_df = pd.DataFrame(forecast, index=forecast_idx, columns=['Production au plant (g)'])
        
        # Affichage des prédictions
        print(f"\nPrévisions de Holt's Linear Trend pour {forecast_periods} années:")
        for i, (year, value) in enumerate(forecast_df.iterrows()):
            season = f"{year} - {year + 1}"
            print(f"  - Saison {season}: {value['Production au plant (g)']:.2f} g/plant")
        
        # Sauvegarde des prédictions
        os.makedirs('generated/data/projections', exist_ok=True)
        forecast_df['Saison'] = [f"{year} - {year + 1}" for year in forecast_df.index]
        forecast_df.to_csv('generated/data/projections/holts_trend_projections.csv')
        
        # Combiner les données historiques et les prédictions
        combined_df = pd.concat([time_series[['Production au plant (g)']], forecast_df[['Production au plant (g)']]])
        combined_df['Type'] = ['Historique'] * len(time_series) + ['Prévision'] * len(forecast_df)
        
        return combined_df, forecast_df
        
    except Exception as e:
        print(f"Erreur lors de la génération des prévisions: {e}")
        return None, None


def holts_method_by_parcel(data, forecast_periods=2, exponential=False):
    """
    Applique la méthode de Holt à chaque combinaison parcelle-espécie.
    
    Retourne:
    ---------
    DataFrame avec les prévisions pour chaque parcelle-espécie
    """
    # Identification des parcelles et espécies uniques
    parcels = data['Parcelle'].unique()
    species = data['Espécie'].unique()
    
    # Pour stocker les résultats
    forecast_results = []
    
    # Exploration des données par parcelle/espécie
    for parcel in parcels:
        for sp in species:
            # Filtre pour la parcelle/espécie spécifique
            filtered_data = data[(data['Parcelle'] == parcel) & (data['Espécie'] == sp)]
            
            if len(filtered_data) < 3:
                # Pas assez de données pour cette combinaison
                continue
            
            # Préparation des données
            try:
                time_series = prepare_time_series_data(filtered_data)
                
                if len(time_series) < 3:
                    continue
                
                # Extraction de la série temporelle
                y = time_series['Production au plant (g)'].dropna()
                
                if len(y) < 3:
                    continue
                
                # Création et ajustement du modèle
                model = Holt(y, exponential=exponential)
                fit_model = model.fit(optimized=True)
                
                # Génération des prévisions
                forecast = fit_model.forecast(forecast_periods)
                
                # Stockage des résultats
                for i, value in enumerate(forecast):
                    forecast_results.append({
                        'Parcelle': parcel,
                        'Espécie': sp,
                        'Période': i + 1,
                        'Prévision': value,
                        'Alpha': fit_model.params['smoothing_level'],
                        'Beta': fit_model.params['smoothing_trend'],
                        'AIC': fit_model.aic,
                        'Type': 'Exponentiel' if exponential else 'Linéaire'
                    })
                    
            except Exception as e:
                print(f"Erreur pour {parcel}/{sp}: {e}")
                continue
    
    # Création du DataFrame des prévisions
    if not forecast_results:
        print("Aucune prévision par parcelle disponible avec la méthode de Holt.")
        return None
    
    forecast_df = pd.DataFrame(forecast_results)
    
    # Sauvegarde des prévisions
    os.makedirs('generated/data/projections', exist_ok=True)
    forecast_df.to_csv('generated/data/projections/holts_by_parcel.csv', index=False)
    
    return forecast_df


def compare_holts_models(data, forecast_periods=3):
    """
    Compare les modèles de Holt linéaire et exponentiel.
    Utilise l'âge Brut comme variable d'âge.
    
    Retourne:
    ---------
    résultats des deux modèles et leurs métriques
    """
    # Préparation des données
    time_series = prepare_time_series_data(data)
    
    if len(time_series) < 3:
        print("Erreur: Données insuffisantes pour la comparaison des modèles de Holt.")
        return None, None, None
    
    # Extraction de la série temporelle
    y = time_series['Production au plant (g)'].dropna()
    
    if len(y) < 3:
        print("Erreur: Données valides insuffisantes pour la comparaison.")
        return None, None, None
    
    results = []
    model_fits = {}
    forecasts = {}
    
    # Modèles à comparer
    models = [
        ('Linéaire', False),
        ('Exponentiel', True)
    ]
    
    # Ajustement de chaque modèle
    for name, exponential in models:
        try:
            # Création et ajustement du modèle
            model = Holt(y, exponential=exponential, damped_trend=False)
            fit_model = model.fit(optimized=True)
            
            # Calcul des métriques
            mse = ((fit_model.fittedvalues - y) ** 2).mean()
            rmse = np.sqrt(mse)
            aic = fit_model.aic
            
            # Génération des prévisions
            forecast = fit_model.forecast(forecast_periods)
            
            # Stockage des résultats
            results.append({
                'Type': name,
                'Alpha': fit_model.params['smoothing_level'],
                'Beta': fit_model.params['smoothing_trend'],
                'RMSE': rmse,
                'AIC': aic
            })
            
            model_fits[name] = fit_model
            forecasts[name] = forecast
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement du modèle {name}: {e}")
            continue
    
    # Création du DataFrame des résultats
    if not results:
        print("Erreur: Aucun modèle n'a pu être ajusté.")
        return None, None, None
    
    results_df = pd.DataFrame(results)
    
    # Conversion des prévisions en DataFrame
    forecast_data = []
    
    last_idx = time_series.index[-1]
    forecast_idx = pd.RangeIndex(start=last_idx + 1, stop=last_idx + forecast_periods + 1)
    
    for name, forecast in forecasts.items():
        for i, (idx, value) in enumerate(zip(forecast_idx, forecast)):
            forecast_data.append({
                'Type': name,
                'Période': i + 1,
                'Année': idx,
                'Saison': f"{idx} - {idx + 1}",
                'Prévision': value
            })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Visualisation des résultats
    print("\nComparaison des modèles de Holt:")
    print(results_df[['Type', 'Alpha', 'Beta', 'RMSE', 'AIC']].to_string(index=False))
    
    print("\nComparaison des prévisions:")
    pivot_forecasts = forecast_df.pivot(index='Saison', columns='Type', values='Prévision')
    print(pivot_forecasts)
    
    # Sauvegarde des résultats
    os.makedirs('generated/data/projections', exist_ok=True)
    results_df.to_csv('generated/data/projections/holts_models_comparison.csv', index=False)
    forecast_df.to_csv('generated/data/projections/holts_forecasts_comparison.csv', index=False)
    
    return results_df, forecast_df, model_fits
