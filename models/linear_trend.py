#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.deterministic import DeterministicProcess


def prepare_time_series_data(data):
    """
    Pru00e9pare les donnu00e9es pour l'analyse des su00e9ries temporelles.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    """
    # Cru00e9er un tableau ru00e9sumant la production moyenne par saison
    time_series = data.groupby(['Saison'])['Production au plant (g)'].mean().reset_index()
    # Extraire l'annu00e9e de du00e9but de la saison et convertir en entier
    time_series['Year'] = time_series['Saison'].str.split(' - ').str[0].astype(int)
    # Trier par annu00e9e
    time_series = time_series.sort_values('Year')
    # Cru00e9er un index temporel
    time_series['Time_Index'] = range(len(time_series))
    # Du00e9finir l'index temporel pour les analyses de su00e9ries temporelles
    time_series = time_series.set_index('Time_Index')
    
    return time_series


def build_linear_trend_model(data):
    """
    Construit un modu00e8le de tendance linu00e9aire pour la production de truffes.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    """
    print("\nConstruction du modu00e8le de tendance linu00e9aire...")
    
    # Pru00e9parer les donnu00e9es temporelles
    time_series = prepare_time_series_data(data)
    
    if len(time_series) < 2:
        print("  Avertissement: Nombre insuffisant de points temporels pour le modu00e8le linu00e9aire.")
        return None, time_series
    
    # Cru00e9er un processus du00e9terministe pour la tendance
    dp = DeterministicProcess(index=time_series.index, constant=True, order=1)
    X = dp.in_sample()
    
    # Ajuster le modu00e8le de tendance linu00e9aire
    y = time_series['Production au plant (g)'].values
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Afficher un ru00e9sumu00e9 du modu00e8le
    print(f"  Coefficient de tendance: {results.params[1]:.2f} (g/an)")
    print(f"  Ru00b2: {results.rsquared:.3f}")
    
    return results, time_series


def project_linear_trend(model_results, time_series, forecast_years=3):
    """
    Projette la production future en utilisant le modu00e8le de tendance linu00e9aire.
    
    Paramu00e8tres:
    -----------
    model_results : ru00e9sultats du modu00e8le OLS
    time_series : su00e9rie temporelle des donnu00e9es historiques
    forecast_years : nombre d'annu00e9es u00e0 pru00e9voir
    
    Retourne:
    ---------
    DataFrame avec les valeurs historiques et les pru00e9visions
    """
    if model_results is None:
        print("  Avertissement: Modu00e8le linu00e9aire non disponible pour les projections.")
        return None, None
    
    print(f"\nProjection de la tendance linu00e9aire pour {forecast_years} annu00e9es...")
    
    # Extraire les annu00e9es historiques et les donnu00e9es de production
    historical_years = time_series['Year'].values
    historical_production = time_series['Production au plant (g)'].values
    
    # Cru00e9er l'index pour la pru00e9vision
    last_index = max(time_series.index)
    forecast_index = pd.RangeIndex(start=last_index + 1, stop=last_index + forecast_years + 1)
    
    # Gu00e9nu00e9rer le processus du00e9terministe pour la pru00e9vision
    dp = DeterministicProcess(index=pd.RangeIndex(start=0, stop=last_index + forecast_years + 1), constant=True, order=1)
    X_forecast = dp.out_of_sample(steps=forecast_years)
    
    # Pru00e9voir la production future
    forecast = model_results.predict(X_forecast)
    
    # Extrapoler les annu00e9es futures
    last_year = historical_years[-1]
    future_years = [last_year + i + 1 for i in range(forecast_years)]
    
    # Combiner historique et pru00e9visions dans un seul DataFrame
    combined_df = pd.DataFrame({
        'Year': np.concatenate([historical_years, future_years]),
        'Production': np.concatenate([historical_production, forecast]),
        'Type': ['Historical'] * len(historical_years) + ['Forecast'] * forecast_years
    })
    
    # Calculer l'augmentation pru00e9vue
    if len(historical_production) > 0:
        baseline = historical_production[-1]
        percentage_increase = ((forecast[-1] / baseline) - 1) * 100 if baseline > 0 else float('inf')
        print(f"  Production de base (dernier point): {baseline:.2f} g/plant")
        print(f"  Production pru00e9vue (dans {forecast_years} ans): {forecast[-1]:.2f} g/plant")
        print(f"  Augmentation projetu00e9e: {percentage_increase:.1f}%")
    
    return combined_df, forecast


def analyze_parcel_trends(data):
    """
    Analyse les tendances linu00e9aires par parcelle et espu00e8ce.
    Retourne les parcelles avec les tendances les plus fortes.
    """
    print("\nAnalyse des tendances par parcelle...")
    
    # Ru00e9cupu00e9rer les parcelles uniques
    parcels = data['Parcelle'].unique()
    
    # Stocker les ru00e9sultats d'analyse
    parcel_trends = []
    
    for parcel in parcels:
        parcel_data = data[data['Parcelle'] == parcel]
        
        # S'assurer qu'il y a au moins deux points temporels
        if len(parcel_data['Saison'].unique()) < 2:
            continue
        
        # Pru00e9parer les donnu00e9es temporelles pour cette parcelle
        time_series = prepare_time_series_data(parcel_data)
        
        # Ajuster un modu00e8le linu00e9aire simple
        dp = DeterministicProcess(index=time_series.index, constant=True, order=1)
        X = dp.in_sample()
        y = time_series['Production au plant (g)'].values
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Ru00e9cupu00e9rer les informations sur l'espu00e8ce
        species = parcel_data['Espu00e8ce'].iloc[0] if not parcel_data.empty else 'Inconnue'
        
        # Ajouter les ru00e9sultats u00e0 la liste
        parcel_trends.append({
            'Parcelle': parcel,
            'Espu00e8ce': species,
            'Coefficient': results.params[1],
            'P_Value': results.pvalues[1],
            'R_Squared': results.rsquared
        })
    
    # Convertir en DataFrame et trier par coefficient de tendance
    trends_df = pd.DataFrame(parcel_trends)
    if not trends_df.empty:
        trends_df = trends_df.sort_values('Coefficient', ascending=False)
        
        # Afficher les parcelles avec les tendances les plus fortes
        print("\n  Parcelles avec les tendances positives les plus fortes :")
        for i, row in trends_df.head(3).iterrows():
            print(f"    {row['Parcelle']} ({row['Espu00e8ce']}): {row['Coefficient']:.2f} g/an, Ru00b2 = {row['R_Squared']:.3f}")
        
        print("\n  Parcelles avec les tendances nu00e9gatives les plus fortes :")
        for i, row in trends_df.tail(3).iloc[::-1].iterrows():
            print(f"    {row['Parcelle']} ({row['Espu00e8ce']}): {row['Coefficient']:.2f} g/an, Ru00b2 = {row['R_Squared']:.3f}")
    else:
        print("  Aucune tendance par parcelle disponible pour l'analyse.")
    
    return trends_df
