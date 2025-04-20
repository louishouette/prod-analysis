#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os

# J'ai remarquu00e9 que le fichier original importait prepare_time_series_data de linear_trend, 
# mais avec notre nouvelle structure, nous pouvons l'importer directement.
from models.linear_trend_model import prepare_time_series_data

def build_exponential_smoothing_model(data, alpha=None):
    """
    Construit un modu00e8le de lissage exponentiel simple pour la production de truffes.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    
    Paramu00e8tres:
    -----------
    data : DataFrame des donnu00e9es de production
    alpha : paramu00e8tre de lissage (None pour estimation automatique)
    
    Retourne:
    ---------
    modu00e8le ajustu00e9, donnu00e9es de su00e9rie temporelle
    """
    print("\nConstruction du modu00e8le de lissage exponentiel simple...")
    
    time_series = prepare_time_series_data(data)
    import pandas as pd
    if not pd.api.types.is_integer_dtype(time_series.index):
        try:
            time_series.index = time_series.index.astype(int)
        except Exception:
            print("[AVERTISSEMENT] L'index de la su00e9rie temporelle n'est pas convertible en annu00e9es (int). Pru00e9visions non fiables.")
    
    # Forcer RangeIndex strictement consu00e9cutif (min u00e0 max)
    years = time_series.index.values
    if pd.api.types.is_integer_dtype(years):
        if len(years) > 1:
            full_range = pd.RangeIndex(start=years[0], stop=years[-1]+1)
            time_series = time_series.reindex(full_range)
    
    # Vu00e9rifier s'il y a assez de donnu00e9es
    if len(time_series) < 2:
        print("Erreur: Donnu00e9es insuffisantes pour le lissage exponentiel.")
        return None, time_series
    
    # Pru00e9paration des donnu00e9es de su00e9rie temporelle
    y = time_series['Production au plant (g)'].dropna()
    
    if len(y) < 2:
        print("Erreur: Donnu00e9es valides insuffisantes pour le lissage exponentiel.")
        return None, time_series
    
    # Construction et ajustement du modu00e8le
    model = SimpleExpSmoothing(y)
    
    if alpha is None:
        # Estimation automatique du paramu00e8tre
        fit_model = model.fit(optimized=True, remove_bias=True)
        print(f"Modu00e8le ajustu00e9 avec alpha={fit_model.params['smoothing_level']:.4f} (estimu00e9 automatiquement)")
    else:
        # Utilisation du paramu00e8tre spu00e9cifiu00e9
        fit_model = model.fit(smoothing_level=alpha, optimized=False, remove_bias=True)
        print(f"Modu00e8le ajustu00e9 avec alpha={alpha:.4f} (valeur fixe)")
    
    # Affichage des paramu00e8tres du modu00e8le
    print("\nParamu00e8tres du modu00e8le de lissage exponentiel simple:")
    print(f"Alpha (niveau): {fit_model.params['smoothing_level']:.4f}")
    print(f"AIC: {fit_model.aic:.2f}")
    print(f"BIC: {fit_model.bic:.2f}")
    
    return fit_model, time_series


def project_exponential_smoothing(model, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modu00e8le de lissage exponentiel.
    
    Paramu00e8tres:
    -----------
    model : modu00e8le de lissage exponentiel ajustu00e9
    time_series : DataFrame des donnu00e9es de su00e9rie temporelle
    forecast_periods : nombre de pu00e9riodes u00e0 pru00e9voir
    
    Retourne:
    ---------
    donnu00e9es combinu00e9es (historiques + pru00e9visions), pru00e9visions
    """
    if model is None:
        print("Erreur: Modu00e8le de lissage exponentiel invalide.")
        return None, None
    
    print(f"\nProjection de la production pour {forecast_periods} pu00e9riodes futures...")
    
    # Genu00e9ration des pru00e9visions
    try:
        forecast = model.forecast(forecast_periods)
        
        # Conversion en DataFrame pour faciliter la visualisation
        last_idx = time_series.index[-1]
        forecast_idx = pd.RangeIndex(start=last_idx + 1, stop=last_idx + forecast_periods + 1)
        forecast_df = pd.DataFrame(forecast, index=forecast_idx, columns=['Production au plant (g)'])
        
        # Affichage des pru00e9visions
        print("Projections complu00e9tu00e9es avec succu00e8s.\n")
        print("Pru00e9visions pour les pu00e9riodes futures:")
        for period, value in enumerate(forecast, 1):
            print(f"Pu00e9riode {period + 1}: {value:.2f} g/plant")
        
        # Sauvegarde des pru00e9visions
        os.makedirs('generated/data/projections', exist_ok=True)
        forecast_df.to_csv('generated/data/projections/exp_smoothing_projections.csv')
        
        # Combiner donnu00e9es historiques et pru00e9visions pour visualisation
        combined = pd.concat([time_series[['Production au plant (g)']], forecast_df])
        combined['Type'] = ['Historique'] * len(time_series) + ['Pru00e9vision'] * len(forecast_df)
        
        return combined, forecast_df
    
    except Exception as e:
        print(f"Erreur lors de la genu00e9ration des pru00e9visions: {e}")
        return None, None


def exponential_smoothing_by_parcel(data, forecast_periods=3, min_periods=2):
    """
    Applique le lissage exponentiel pour chaque parcelle ayant suffisamment de donnu00e9es.
    
    Paramu00e8tres:
    -----------
    data : DataFrame des donnu00e9es de production
    forecast_periods : nombre de pu00e9riodes u00e0 pru00e9voir
    min_periods : nombre minimal de pu00e9riodes nu00e9cessaires pour la modu00e9lisation
    
    Retourne:
    ---------
    DataFrame avec les prévisions par parcelle
    """
    # Identification des parcelles uniques
    parcels = data['Parcelle'].unique()
    
    # Stockage des résultats
    parcel_forecasts = {}
    parcel_models = {}
    
    # Exploration des données par parcelle
    for parcel in parcels:
        # Filtre pour la parcelle spécifique
        parcel_data = data[data['Parcelle'] == parcel]
        
        # Vérification du nombre d'observations
        seasons = parcel_data['Saison'].unique()
        
        if len(seasons) < min_periods:
            print(f"Parcelle {parcel}: Donnu00e9es insuffisantes ({len(seasons)} saisons, minimum {min_periods})")
            continue
        
        # Préparation des données temporelles
        try:
            time_series = prepare_time_series_data(parcel_data)
            
            if len(time_series) < min_periods:
                print(f"Parcelle {parcel}: Série temporelle insuffisante ({len(time_series)} points, minimum {min_periods})")
                continue
            
            # Modu00e8le de lissage exponentiel
            model = SimpleExpSmoothing(time_series['Production au plant (g)'].dropna())
            fit_model = model.fit(optimized=True)
            
            # Pru00e9vision
            forecast = fit_model.forecast(forecast_periods)
            
            # Stockage des ru00e9sultats
            parcel_forecasts[parcel] = forecast
            parcel_models[parcel] = fit_model
            
        except Exception as e:
            print(f"Erreur pour la parcelle {parcel}: {e}")
            continue
    
    # Création du DataFrame des prévisions
    if not parcel_forecasts:
        print("Aucune prévision par parcelle disponible.")
        return None
    
    # Structuration des résultats
    forecast_data = []
    
    for parcel, forecast in parcel_forecasts.items():
        model = parcel_models[parcel]
        alpha = model.params['smoothing_level']
        aic = model.aic
        
        for i, value in enumerate(forecast):
            forecast_data.append({
                'Parcelle': parcel,
                'Période': i + 1,
                'Prévision': value,
                'Alpha': alpha,
                'AIC': aic
            })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Sauvegarde des prévisions par parcelle
    os.makedirs('generated/data/projections', exist_ok=True)
    forecast_df.to_csv('generated/data/projections/exp_smoothing_by_parcel.csv', index=False)
    
    return forecast_df
