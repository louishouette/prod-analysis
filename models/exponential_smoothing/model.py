#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from models.linear_trend.model import prepare_time_series_data

def build_exponential_smoothing_model(data, alpha=None):
    """
    Construit un modèle de lissage exponentiel simple pour la production de truffes.
    Utilise l'Âge Brut comme variable d'âge.
    """
    time_series = prepare_time_series_data(data)
    if time_series.empty or len(time_series) < 2:
        print("Erreur: Pas assez de données pour construire un modèle de lissage exponentiel")
        return None, time_series
    try:
        model = SimpleExpSmoothing(time_series['Production au plant (g)'])
        if alpha is not None:
            fit_model = model.fit(smoothing_level=alpha, optimized=False)
        else:
            fit_model = model.fit(optimized=True)
        return fit_model, time_series
    except Exception as e:
        print(f"Erreur lors de la construction du modèle de lissage exponentiel: {e}")
        return None, time_series

def project_exponential_smoothing(model, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modèle de lissage exponentiel.
    """
    if model is None or time_series.empty:
        return pd.DataFrame(), pd.DataFrame()
    try:
        forecast = model.forecast(steps=forecast_periods)
        last_period = time_series.index.max()
        forecast_index = range(last_period + 1, last_period + forecast_periods + 1)
        forecasts_df = pd.DataFrame({
            'forecast': forecast.values
        }, index=forecast_index)
        historical_df = pd.DataFrame({
            'Production au plant (g)': time_series['Production au plant (g)'],
            'Type': 'Historical'
        }, index=time_series.index)
        forecast_data = pd.DataFrame({
            'Production au plant (g)': forecast.values,
            'Type': 'Forecast'
        }, index=forecast_index)
        combined_data = pd.concat([historical_df, forecast_data])
        return combined_data, forecasts_df
    except Exception as e:
        print(f"Erreur lors de la projection avec le modèle de lissage exponentiel: {e}")
        return pd.DataFrame(), pd.DataFrame()

def exponential_smoothing_by_parcel(data, forecast_periods=3, min_periods=2):
    """
    Applique le lissage exponentiel pour chaque parcelle ayant suffisamment de données.
    """
    valid_data = data[
        (~data['Production au plant (g)'].isna()) &
        (data['Production au plant (g)'] > 0) &
        (data['Age'] > 0)
    ].copy()
    if valid_data.empty:
        return pd.DataFrame()
    parcel_data = valid_data.groupby(['Parcelle', 'Saison'])['Production au plant (g)'].mean().reset_index()
    parcels = parcel_data['Parcelle'].value_counts()
    parcels = parcels[parcels >= min_periods].index.tolist()
    if not parcels:
        return pd.DataFrame()
    seasons = sorted(parcel_data['Saison'].unique())
    next_seasons = [seasons[-1] + i + 1 for i in range(forecast_periods)]
    forecasts = []
    for parcel in parcels:
        parcel_ts = parcel_data[parcel_data['Parcelle'] == parcel]
        parcel_ts = parcel_ts.sort_values('Saison')
        if len(parcel_ts) >= min_periods:
            try:
                model = SimpleExpSmoothing(parcel_ts['Production au plant (g)'].values)
                fit_model = model.fit(optimized=True)
                forecast_values = fit_model.forecast(steps=forecast_periods)
                for i, season in enumerate(next_seasons):
                    forecasts.append({
                        'Parcelle': parcel,
                        'Saison': season,
                        'Type': 'Prévision',
                        'Production prévue (g/plant)': forecast_values[i],
                        'Alpha': fit_model.params['smoothing_level']
                    })
            except Exception:
                continue
    if forecasts:
        return pd.DataFrame(forecasts)
    else:
        return pd.DataFrame()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from models.linear_trend.model import prepare_time_series_data


def build_exponential_smoothing_model(data, alpha=None):
    """
    Construit un modèle de lissage exponentiel simple pour la production de truffes.
    Utilise l'Âge Brut comme variable d'âge.
    
    Paramètres:
    -----------
    data : DataFrame des données de production
    alpha : paramètre de lissage (None pour estimation automatique)
    
    Retourne:
    ---------
    modèle ajusté, données de série temporelle
    """
    # Préparer les données pour la modélisation de série temporelle
    time_series = prepare_time_series_data(data)
    
    if time_series.empty or len(time_series) < 2:
        print("Erreur: Pas assez de données pour construire un modèle de lissage exponentiel")
        return None, time_series
    
    print("\nConstruction du modèle de lissage exponentiel simple...")
    
    try:
        # Appliquer le modèle de lissage exponentiel simple
        model = SimpleExpSmoothing(time_series['Production au plant (g)'])
        
        # Ajuster le modèle avec le paramètre alpha spécifié ou laisser estimer automatiquement
        if alpha is not None:
            fit_model = model.fit(smoothing_level=alpha, optimized=False)
            print(f"Modèle ajusté avec alpha={alpha} (spécifié par l'utilisateur)")
        else:
            fit_model = model.fit(optimized=True)
            print(f"Modèle ajusté avec alpha={fit_model.params['smoothing_level']:.4f} (estimé automatiquement)")
        
        # Afficher les paramètres du modèle
        print(f"\nParamètres du modèle de lissage exponentiel simple:")
        print(f"Alpha (niveau): {fit_model.params['smoothing_level']:.4f}")
        print(f"AIC: {fit_model.aic:.2f}")
        print(f"BIC: {fit_model.bic:.2f}")
        
        return fit_model, time_series
        
    except Exception as e:
        print(f"Erreur lors de la construction du modèle de lissage exponentiel: {e}")
        return None, time_series


def project_exponential_smoothing(model, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modèle de lissage exponentiel.
    
    Paramètres:
    -----------
    model : modèle de lissage exponentiel ajusté
    time_series : DataFrame des données de série temporelle
    forecast_periods : nombre de périodes à prévoir
    
    Retourne:
    ---------
    données combinées (historiques + prévisions), prévisions
    """
    if model is None or time_series.empty:
        print("Erreur: Modèle de lissage exponentiel non disponible pour les projections")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"\nProjection de la production pour {forecast_periods} périodes futures...")
    
    try:
        # Générer les prévisions sans intervalles de confiance (non supporté par SimpleExpSmoothing)
        forecast = model.forecast(steps=forecast_periods)
        
        # Créer un DataFrame pour les prévisions
        last_period = time_series.index.max()
        forecast_index = range(last_period + 1, last_period + forecast_periods + 1)
        
        forecasts_df = pd.DataFrame({
            'forecast': forecast.values
        }, index=forecast_index)
        
        # Combiner les données historiques et les prévisions pour la visualisation
        historical_df = pd.DataFrame({
            'Production au plant (g)': time_series['Production au plant (g)'],
            'Type': 'Historical'
        }, index=time_series.index)
        
        forecast_data = pd.DataFrame({
            'Production au plant (g)': forecast.values,
            'Type': 'Forecast'
        }, index=forecast_index)
        
        combined_data = pd.concat([historical_df, forecast_data])
        
        print("Projections complétées avec succès.")
        print("\nPrévisions pour les périodes futures:")
        for idx, val in forecast.items():
            print(f"Période {idx}: {val:.2f} g/plant")
        
        return combined_data, forecasts_df
        
    except Exception as e:
        print(f"Erreur lors de la projection avec le modèle de lissage exponentiel: {e}")
        return pd.DataFrame(), pd.DataFrame()


def exponential_smoothing_by_parcel(data, forecast_periods=3, min_periods=2):
    """
    Applique le lissage exponentiel pour chaque parcelle ayant suffisamment de données.
    
    Paramètres:
    -----------
    data : DataFrame des données de production
    forecast_periods : nombre de périodes à prévoir
    min_periods : nombre minimal de périodes nécessaires pour la modélisation
    
    Retourne:
    ---------
    DataFrame avec les prévisions par parcelle
    """
    # Filtrer les données valides pour l'analyse par parcelle
    valid_data = data[
        (~data['Production au plant (g)'].isna()) & 
        (data['Production au plant (g)'] > 0) &
        (data['Age'] > 0)
    ].copy()
    
    if valid_data.empty:
        print("Erreur: Pas de données valides pour l'analyse par parcelle")
        return pd.DataFrame()
    
    # Regrouper par parcelle et saison pour obtenir la production moyenne par plant
    parcel_data = valid_data.groupby(['Parcelle', 'Saison'])['Production au plant (g)'].mean().reset_index()
    
    # Lister toutes les parcelles ayant suffisamment de données
    parcels = parcel_data['Parcelle'].value_counts()
    parcels = parcels[parcels >= min_periods].index.tolist()
    
    if not parcels:
        print(f"Erreur: Aucune parcelle n'a au moins {min_periods} périodes de données")
        return pd.DataFrame()
    
    print(f"\nApplication du lissage exponentiel sur {len(parcels)} parcelles...")
    
    # Lister les saisons disponibles et la prochaine saison à prévoir
    seasons = sorted(parcel_data['Saison'].unique())
    # Générer les prochaines saisons au format 'YYYY - YYYY+1'
    last_season = seasons[-1]
    try:
        last_year = int(str(last_season).split(' - ')[0])
    except Exception:
        last_year = int(last_season)
    next_seasons = [f"{last_year + i + 1} - {last_year + i + 2}" for i in range(forecast_periods)]
    
    # Stocker les résultats des prévisions
    forecasts = []
    
    # Pour chaque parcelle, construire et appliquer un modèle de lissage exponentiel
    for parcel in parcels:
        parcel_ts = parcel_data[parcel_data['Parcelle'] == parcel]
        parcel_ts = parcel_ts.sort_values('Saison')
        
        if len(parcel_ts) >= min_periods:
            try:
                # Appliquer le modèle de lissage exponentiel
                model = SimpleExpSmoothing(parcel_ts['Production au plant (g)'].values)
                fit_model = model.fit(optimized=True)
                
                # Générer les prévisions
                forecast_values = fit_model.forecast(steps=forecast_periods)
                
                # Ajouter les prévisions au résultat
                for i, season in enumerate(next_seasons):
                    forecasts.append({
                        'Parcelle': parcel,
                        'Saison': season,
                        'Type': 'Prévision',
                        'Production prévue (g/plant)': forecast_values[i],
                        'Alpha': fit_model.params['smoothing_level']
                    })
            except Exception as e:
                print(f"Erreur lors de la prévision pour la parcelle {parcel}: {e}")
    
    if forecasts:
        forecasts_df = pd.DataFrame(forecasts)
        print(f"Prévisions générées pour {len(parcels)} parcelles sur {len(next_seasons)} saisons.")
        return forecasts_df
    else:
        print("Aucune prévision n'a pu être générée.")
        return pd.DataFrame()
        ts = group_data.groupby(['Saison'])['Production au plant (g)'].mean().reset_index()
        ts['Year'] = ts['Saison'].str.split(' - ').str[0].astype(int)
        ts = ts.sort_values('Year')
        ts.set_index('Year', inplace=True)
        
        # S'il y a assez de points pour le lissage exponentiel
        if len(ts) >= 2:
            try:
                # Ajuster le modèle
                model = SimpleExpSmoothing(ts['Production au plant (g)'])
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
                        'Alpha': res.params['smoothing_level']
                    })
                    
                print(f"    Prévision réussie, alpha={res.params['smoothing_level']:.4f}")
                
            except Exception as e:
                print(f"    Erreur lors de l'ajustement du modèle: {e}")
    
    # Convertir en DataFrame
    forecasts_df = pd.DataFrame(forecasts)
    
    return forecasts_df
