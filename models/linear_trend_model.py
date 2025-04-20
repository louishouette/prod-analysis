#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.deterministic import DeterministicProcess
import os


def prepare_time_series_data(data):
    """
    Prépare les données pour l'analyse des séries temporelles.
    Utilise l'âge Brut comme variable d'âge.
    """
    # Créer un tableau résumant la production moyenne par saison
    time_series = data.groupby(['Saison'])['Production au plant (g)'].mean().reset_index()
    # Extraire l'année de début de la saison et convertir en entier
    time_series['Year'] = time_series['Saison'].str.split(' - ').str[0].astype(int)
    # Trier par année
    time_series = time_series.sort_values('Year')
    # Définir l'année comme index (Int64Index)
    time_series.set_index('Year', inplace=True)
    # Nettoyage : supprimer les NaN
    time_series = time_series[~time_series['Production au plant (g)'].isna()]
    # Vérification taille
    if len(time_series) < 2:
        print("Warning: Insuffisamment de données pour la modélisation de tendance (minimum 2 années)")
    
    return time_series


def build_linear_trend_model(data):
    """
    Construit un modèle de tendance linéaire pour la production de truffes.
    Utilise l'âge Brut comme variable d'âge.
    """
    print("\nConstruction du modèle de tendance linéaire...")
    
    # Préparation des données
    time_series = prepare_time_series_data(data)
    
    if len(time_series) < 2:
        print("Erreur: Données insuffisantes pour construire un modèle de tendance linéaire.")
        return None, None
    
    # Création d'un processus déterministe avec tendance
    dp = DeterministicProcess(
        index=time_series.index,
        constant=True,      # inclure l'intercept
        order=1,           # ordre de tendance (1 = linéaire)
        drop=True
    )
    
    # Création de la matrice de conception (features)
    X = dp.in_sample()
    
    # Ajustement du modèle OLS
    y = time_series['Production au plant (g)']
    model = sm.OLS(y, X)
    model_results = model.fit()
    
    # Afficher la qualité de l'ajustement
    print(f"R² du modèle de tendance linéaire: {model_results.rsquared:.4f}")
    print(f"Pente de la tendance: {model_results.params[1]:.4f} g/an")
    
    return model_results, time_series


def project_linear_trend(model_results, time_series, forecast_years=3):
    """
    Projette la production future en utilisant le modèle de tendance linéaire.
    
    Paramètres:
    -----------
    model_results : résultats du modèle OLS
    time_series : série temporelle des données historiques
    forecast_years : nombre d'années à prédire
    
    Retourne:
    ---------
    DataFrame avec les valeurs historiques et les prédictions
    """
    if model_results is None or time_series is None or len(time_series) < 2:
        print("Erreur: Modèle ou données de tendance invalides pour les projections.")
        return None
    
    # Création de l'index de prédiction
    last_year = time_series.index[-1]
    forecast_index = pd.Index(range(last_year + 1, last_year + forecast_years + 1), name='Year')
    
    # Création du processus déterministe pour la prédiction
    dp = DeterministicProcess(
        index=forecast_index,
        constant=True,
        order=1,
        drop=True
    )
    
    # Création de la matrice de prédiction
    X_forecast = dp.out_of_sample(steps=forecast_years)
    
    # Génération des prédictions ponctuelles
    y_forecast = model_results.predict(X_forecast)
    
    # Calcul de l'intervalle de confiance (IC) à 95%
    y_forecast_ci = model_results.get_prediction(X_forecast).conf_int(alpha=0.05)
    
    # Création du DataFrame de prédiction
    forecast_df = pd.DataFrame({
        'Production au plant (g)': y_forecast,
        'Lower CI': y_forecast_ci[:, 0],
        'Upper CI': y_forecast_ci[:, 1]
    }, index=forecast_index)
    
    # Affichage des prédictions
    print("\nPrévisions de tendance linéaire pour", forecast_years, "années:")
    for i, (year, values) in enumerate(forecast_df.iterrows()):
        season = f"{year} - {year + 1}"
        print(f"  - Saison {season}: {values['Production au plant (g)']:.2f} g/plant (IC 95%: [{values['Lower CI']:.2f}, {values['Upper CI']:.2f}])")
    
    # Sauvegarde des projections
    os.makedirs('generated/data/projections', exist_ok=True)
    forecast_df['Saison'] = [f"{year} - {year + 1}" for year in forecast_df.index]
    forecast_df.to_csv('generated/data/projections/linear_trend_projections.csv')
    
    # Combiner données historiques et prédictions
    combined_df = pd.concat([time_series, forecast_df])
    combined_df['Type'] = ['Historique'] * len(time_series) + ['Prédiction'] * len(forecast_df)
    
    # Renvoyer deux valeurs pour rester compatible avec le code appelant
    return combined_df, forecast_df


def analyze_parcel_trends(data):
    """
    Analyse les tendances linéaires par parcelle et espèce.
    Retourne les parcelles avec les tendances les plus fortes.
    """
    # Préparation des données par parcelle et espèce
    parcels = data['Parcelle'].unique()
    species = data['Espèce'].unique()
    
    trend_results = []
    
    for parcel in parcels:
        for sp in species:
            # Filtrage des données
            parcel_data = data[(data['Parcelle'] == parcel) & (data['Espèce'] == sp)]
            
            if len(parcel_data) < 2:
                continue  # Pas assez de données pour une tendance
            
            # Préparation des données de tendance
            try:
                time_series = prepare_time_series_data(parcel_data)
                
                if len(time_series) < 2:
                    continue
                
                # Ajustement du modèle
                dp = DeterministicProcess(
                    index=time_series.index,
                    constant=True,
                    order=1,
                    drop=True
                )
                
                X = dp.in_sample()
                y = time_series['Production au plant (g)']
                model = sm.OLS(y, X)
                model_results = model.fit()
                
                # Enregistrement des résultats
                trend_results.append({
                    'Parcelle': parcel,
                    'Espèce': sp,
                    'Pente': model_results.params[1],
                    'R²': model_results.rsquared,
                    'P-value': model_results.pvalues[1],
                    'Observations': len(time_series)
                })
                
            except Exception as e:
                print(f"Erreur lors de l'analyse de tendance pour {parcel}/{sp}: {e}")
                continue
    
    # Conversion en DataFrame et tri par pente
    trend_df = pd.DataFrame(trend_results)
    
    if not trend_df.empty:
        trend_df = trend_df.sort_values('Pente', ascending=False)
    
    return trend_df
