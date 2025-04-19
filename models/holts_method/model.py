#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from models.linear_trend.model import prepare_time_series_data


def build_holts_trend_model(data, alpha=None, beta=None, exponential=False):
    """
    Construit un modèle de Holt (tendance linéaire) pour la production de truffes.
    Utilise l'Âge Brut comme variable d'âge.
    
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
    print("\nConstruction du modèle de Holt (tendance linéaire)...")
    
    # Préparer les données pour l'analyse des séries temporelles
    time_series = prepare_time_series_data(data)
    
    # Créer et ajuster le modèle
    model = Holt(time_series['Production au plant (g)'], exponential=exponential)
    
    if alpha is None or beta is None:
        # Optimisation automatique des paramètres
        results = model.fit(optimized=True, use_brute=True)
        print(f"Paramètres optimisés: alpha={results.params['smoothing_level']:.4f}, beta={results.params['smoothing_trend']:.4f}")
    else:
        # Utilisation des paramètres fournis
        results = model.fit(smoothing_level=alpha, smoothing_trend=beta)
        print(f"Paramètres utilisés: alpha={alpha:.4f}, beta={beta:.4f}")
    
    # Afficher les métriques de qualité d'ajustement
    mse = ((results.fittedvalues - time_series['Production au plant (g)']) ** 2).mean()
    rmse = np.sqrt(mse)
    print(f"RMSE (erreur quadratique moyenne): {rmse:.4f}")
    print(f"AIC: {results.aic:.4f}")
    
    # Type de tendance utilisée
    trend_type = "exponentielle" if exponential else "linéaire"
    print(f"Type de tendance: {trend_type}")
    
    return results, time_series


def project_holts_trend(model_results, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modèle de Holt.
    
    Paramètres:
    -----------
    model_results : résultats du modèle de Holt
    time_series : série temporelle des données historiques
    forecast_periods : nombre de périodes à prévoir
    
    Retourne:
    ---------
    DataFrame avec les valeurs historiques et les prévisions
    """
    # Faire les prévisions
    forecast = model_results.forecast(steps=forecast_periods)
    
    # Créer un DataFrame pour les prévisions
    last_year = time_series.index[-1]
    forecast_years = range(last_year + 1, last_year + forecast_periods + 1)
    forecast_index = pd.Index(forecast_years, name='Year')
    
    forecast_df = pd.DataFrame({
        'Production au plant (g)': forecast,
        'Type': 'Forecast'
    }, index=forecast_index)
    
    # (Pas d'intervalle de confiance disponible pour HoltWintersResults dans statsmodels)
    # Ajouter une colonne de type aux données historiques
    historical = time_series.copy()
    historical['Type'] = 'Historical'
    
    # Combiner les données historiques et les prévisions
    combined = pd.concat([historical, forecast_df])
    
    print(f"\nPrévisions de Holt's Linear Trend pour {forecast_periods} années:")
    for year in forecast_df.index:
        season = f"{year} - {year+1}"
        print(f"  - Saison {season}: {forecast_df.loc[year, 'Production au plant (g)']:.2f} g/plant")
    
    return combined, forecast_df


def holts_method_by_parcel(data, forecast_periods=2, exponential=False):
    """
    Applique la méthode de Holt à chaque combinaison parcelle-espèce.
    
    Retourne:
    ---------
    DataFrame avec les prévisions pour chaque parcelle-espèce
    """
    forecasts = []
    
    # Grouper par parcelle et espèce
    groups = data.groupby(['Parcelle', 'Espèce'])
    
    print("\nAnalyse des tendances par parcelle avec la méthode de Holt:")
    trend_type = "exponentielle" if exponential else "linéaire"
    print(f"Type de tendance: {trend_type}")
    
    for (parcel, species), group_data in groups:
        # S'assurer qu'il y a au moins deux saisons de données
        if group_data['Saison'].nunique() < 2:
            continue
            
        print(f"\n  Analyse de {parcel}-{species}:")
            
        # Créer une série temporelle pour cette parcelle-espèce
        ts = group_data.groupby(['Saison'])['Production au plant (g)'].mean().reset_index()
        ts['Year'] = ts['Saison'].str.split(' - ').str[0].astype(int)
        ts = ts.sort_values('Year')
        ts.set_index('Year', inplace=True)
        
        # S'il y a assez de points pour la méthode de Holt
        if len(ts) >= 3:  # Holt nécessite au moins 3 points pour une tendance fiable
            try:
                # Ajuster le modèle
                model = Holt(ts['Production au plant (g)'], exponential=exponential)
                res = model.fit(optimized=True)
                
                # Faire les prévisions
                forecast = res.forecast(steps=forecast_periods)
                
                # Pour chaque année de prévision
                for i, pred in enumerate(forecast):
                    year = ts.index[-1] + i + 1
                    season = f"{year} - {year+1}"
                    
                    forecasts.append({
                        'Parcelle': parcel,
                        'Espèce': species,
                        'Year': year,
                        'Saison': season,
                        'Production Prévue (g/plant)': pred,
                        'Alpha': res.params['smoothing_level'],
                        'Beta': res.params['smoothing_trend']
                    })
                    
                print(f"    Prévision réussie, alpha={res.params['smoothing_level']:.4f}, beta={res.params['smoothing_trend']:.4f}")
                
            except Exception as e:
                print(f"    Erreur lors de l'ajustement du modèle: {e}")
    
    # Convertir en DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    
    return forecasts_df


def compare_holts_models(data, forecast_periods=3):
    """
    Compare les modèles de Holt linéaire et exponentiel.
    Utilise l'Âge Brut comme variable d'âge.
    
    Retourne:
    ---------
    résultats des deux modèles et leurs métriques
    """
    # Préparer les données
    time_series = prepare_time_series_data(data)
    
    print("\nComparaison des modèles de Holt linéaire et exponentiel:")
    
    # Ajuster le modèle linéaire
    linear_model = Holt(time_series['Production au plant (g)'], exponential=False)
    linear_results = linear_model.fit(optimized=True)
    
    # Ajuster le modèle exponentiel
    exp_model = Holt(time_series['Production au plant (g)'], exponential=True)
    exp_results = exp_model.fit(optimized=True)
    
    # Calculer les métriques pour le modèle linéaire
    linear_mse = ((linear_results.fittedvalues - time_series['Production au plant (g)']) ** 2).mean()
    linear_rmse = np.sqrt(linear_mse)
    
    # Calculer les métriques pour le modèle exponentiel
    exp_mse = ((exp_results.fittedvalues - time_series['Production au plant (g)']) ** 2).mean()
    exp_rmse = np.sqrt(exp_mse)
    
    print(f"Modèle linéaire: RMSE={linear_rmse:.4f}, AIC={linear_results.aic:.4f}")
    print(f"Modèle exponentiel: RMSE={exp_rmse:.4f}, AIC={exp_results.aic:.4f}")
    
    # Déterminer le meilleur modèle basé sur l'AIC
    best_model = "linéaire" if linear_results.aic < exp_results.aic else "exponentiel"
    print(f"Meilleur modèle (basé sur l'AIC): {best_model}")
    
    # Faire les prévisions avec le meilleur modèle
    if best_model == "linéaire":
        return linear_results, time_series, "linéaire"
    else:
        return exp_results, time_series, "exponentiel"
