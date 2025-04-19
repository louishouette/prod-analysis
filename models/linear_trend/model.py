#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.deterministic import DeterministicProcess


def prepare_time_series_data(data):
    """
    Prépare les données pour l'analyse des séries temporelles.
    Utilise l'Âge Brut comme variable d'âge.
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
        print("[AVERTISSEMENT] Série temporelle trop courte pour modélisation (moins de 2 points).")
    # Vérifier variation
    if time_series['Production au plant (g)'].nunique() == 1:
        print("[AVERTISSEMENT] Série temporelle constante (pas de variation de production).")
    return time_series


def build_linear_trend_model(data):
    """
    Construit un modèle de tendance linéaire pour la production de truffes.
    Utilise l'Âge Brut comme variable d'âge.
    """
    print("\nConstruction du modèle de tendance linéaire...")
    
    # Préparer les données pour l'analyse des séries temporelles
    time_series = prepare_time_series_data(data)
    
    # Créer un processus déterministe pour la tendance
    dp = DeterministicProcess(
        index=time_series.index,
        constant=True,   # Intercept (constant)
        order=1,         # Tendance linéaire (ordre 1)
        drop=True        # Supprimer la première colonne (redondante avec constante)
    )
    
    # Créer les variables exogènes (features)
    X = dp.in_sample()
    
    # Ajuster le modèle
    model = sm.OLS(time_series['Production au plant (g)'], X)
    results = model.fit()
    
    # Afficher un résumé des résultats
    print(f"R² du modèle de tendance linéaire: {results.rsquared:.4f}")
    print(f"Pente de la tendance: {results.params[1]:.4f} g/an")
    
    return results, time_series


def project_linear_trend(model_results, time_series, forecast_years=3):
    """
    Projette la production future en utilisant le modèle de tendance linéaire.
    
    Paramètres:
    -----------
    model_results : résultats du modèle OLS
    time_series : série temporelle des données historiques
    forecast_years : nombre d'années à prévoir
    
    Retourne:
    ---------
    DataFrame avec les valeurs historiques et les prévisions
    """
    # Créer un index pour les années de prévision
    last_year = time_series.index[-1]
    future_index = pd.Index(range(last_year + 1, last_year + forecast_years + 1), name='Year')
    
    # Créer un processus déterministe pour les prévisions
    dp = DeterministicProcess(
        index=future_index,
        constant=True,
        order=1,
        drop=True
    )
    
    # Créer les variables exogènes pour les prévisions
    X_forecast = dp.out_of_sample(steps=forecast_years)
    
    # Faire les prévisions
    y_forecast = model_results.predict(X_forecast)
    
    # Créer un DataFrame pour les prévisions
    forecast = pd.DataFrame({
        'Production au plant (g)': y_forecast,
        'Type': 'Forecast'
    }, index=future_index)
    
    # Ajouter une colonne de type aux données historiques
    historical = time_series.copy()
    historical['Type'] = 'Historical'
    
    # Combiner les données historiques et les prévisions
    combined = pd.concat([historical, forecast])
    
    # Calculer les intervalles de confiance pour les prévisions
    pred = model_results.get_prediction(X_forecast)
    pred_int = pred.summary_frame(alpha=0.05)  # 95% d'intervalle de confiance
    
    # Ajouter les intervalles de confiance
    forecast['lower'] = pred_int['mean_ci_lower']
    forecast['upper'] = pred_int['mean_ci_upper']
    
    print(f"\nPrévisions de tendance linéaire pour {forecast_years} années:")
    for year in forecast.index:
        season = f"{year} - {year+1}"
        print(f"  - Saison {season}: {forecast.loc[year, 'Production au plant (g)']:.2f} g/plant " +
              f"(IC 95%: [{forecast.loc[year, 'lower']:.2f}, {forecast.loc[year, 'upper']:.2f}])")
    
    return combined, forecast


def analyze_parcel_trends(data):
    """
    Analyse les tendances linéaires par parcelle et espèce.
    Retourne les parcelles avec les tendances les plus fortes.
    """
    results = []
    
    # Grouper les données par parcelle et espèce
    groups = data.groupby(['Parcelle', 'Espèce'])
    
    for (parcel, species), group_data in groups:
        # S'assurer qu'il y a au moins deux saisons de données
        if group_data['Saison'].nunique() < 2:
            continue
        # Créer une série temporelle pour cette parcelle-espèce
        ts = group_data.groupby(['Saison'])['Production au plant (g)'].mean().reset_index()
        ts['Year'] = ts['Saison'].str.split(' - ').str[0].astype(int)
        ts = ts.sort_values('Year')
        statut = None
        # S'il y a assez de points pour une régression linéaire
        if len(ts) < 3:
            statut = 'Série trop courte'
            pente, r2, pval = float('nan'), float('nan'), float('nan')
        elif ts['Production au plant (g)'].nunique() == 1:
            statut = 'Constante'
            pente, r2, pval = 0.0, float('nan'), float('nan')
        else:
            # Ajuster une régression linéaire simple
            X = sm.add_constant(ts['Year'])
            model = sm.OLS(ts['Production au plant (g)'], X)
            res = model.fit()
            pente = res.params[1]
            r2 = res.rsquared
            pval = res.pvalues[1]
            if not (pd.isna(pente) or pd.isna(r2) or pd.isna(pval)):
                statut = 'OK'
            else:
                statut = 'Non significatif'
        # Stocker les résultats
        results.append({
            'Parcelle': parcel,
            'Espèce': species,
            'Pente': pente,
            'R²': r2,
            'P-value': pval,
            'Statut': statut
        })
    
    # Convertir en DataFrame et trier par pente (tendance)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Pente', ascending=False)
    
    return results_df
