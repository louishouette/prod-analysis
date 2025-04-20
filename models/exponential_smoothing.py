#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from models.linear_trend import prepare_time_series_data

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
    
    # Pru00e9parer les donnu00e9es de su00e9rie temporelle
    time_series = prepare_time_series_data(data)
    
    # Vu00e9rifier qu'il y a suffisamment de donnu00e9es
    if len(time_series) < 2:
        print("  Avertissement: Nombre insuffisant de points temporels pour le lissage exponentiel.")
        return None, time_series
    
    # Forcer un index temporel consou00e9cutif
    if not pd.api.types.is_integer_dtype(time_series.index):
        try:
            time_series.index = time_series.index.astype(int)
        except Exception:
            print("[AVERTISSEMENT] L'index de la su00e9rie temporelle n'est pas convertible en annu00e9es (int). Pru00e9visions non fiables.")
    
    # Forcer RangeIndex strictement consu00e9cutif (min u00e0 max)
    time_series = time_series.reset_index(drop=True)
    
    # Donnu00e9es de production pour le modu00e8le
    y = time_series['Production au plant (g)'].values
    
    try:
        # Ajuster le modu00e8le de lissage exponentiel
        if alpha is None:
            model = SimpleExpSmoothing(y).fit(optimized=True)
            print(f"  Paramu00e8tre alpha optimisu00e9: {model.params['smoothing_level']:.3f}")
        else:
            model = SimpleExpSmoothing(y).fit(smoothing_level=alpha)
            print(f"  Paramu00e8tre alpha spu00e9cifiu00e9: {alpha:.3f}")
        
        return model, time_series
    except Exception as e:
        print(f"  Erreur lors de l'ajustement du modu00e8le de lissage exponentiel: {e}")
        return None, time_series

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
        print("  Avertissement: Modu00e8le de lissage exponentiel non disponible pour les projections.")
        return None, None
    
    print(f"\nProjection du lissage exponentiel pour {forecast_periods} pu00e9riodes...")
    
    try:
        # Extraire les annu00e9es historiques et les donnu00e9es de production
        historical_years = time_series['Year'].values
        historical_production = time_series['Production au plant (g)'].values
        
        # Gu00e9nu00e9rer les pru00e9visions
        forecast = model.forecast(forecast_periods).values
        
        # Extrapoler les annu00e9es futures
        last_year = historical_years[-1]
        future_years = [last_year + i + 1 for i in range(forecast_periods)]
        
        # Combiner historique et pru00e9visions dans un seul DataFrame
        combined_df = pd.DataFrame({
            'Year': np.concatenate([historical_years, future_years]),
            'Production': np.concatenate([historical_production, forecast]),
            'Type': ['Historical'] * len(historical_years) + ['Forecast'] * forecast_periods
        })
        
        # Calculer l'augmentation pru00e9vue
        if len(historical_production) > 0:
            baseline = historical_production[-1]
            percentage_increase = ((forecast[-1] / baseline) - 1) * 100 if baseline > 0 else float('inf')
            print(f"  Production de base (dernier point): {baseline:.2f} g/plant")
            print(f"  Production pru00e9vue (dans {forecast_periods} ans): {forecast[-1]:.2f} g/plant")
            print(f"  Variation projetu00e9e: {percentage_increase:.1f}%")
        
        return combined_df, forecast
    
    except Exception as e:
        print(f"  Erreur lors de la projection du lissage exponentiel: {e}")
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
    DataFrame avec les pru00e9visions par parcelle
    """
    print("\nAnalyse par lissage exponentiel par parcelle...")
    
    # Ru00e9cupu00e9rer les parcelles uniques
    parcels = data['Parcelle'].unique()
    
    # Stocker les projections par parcelle
    parcel_projections = []
    
    # Ru00e9cupu00e9rer la saison actuelle et la saison cible
    latest_season = data['Saison'].max()
    latest_year = int(latest_season.split(' - ')[0])
    target_season = f"{latest_year + 1} - {latest_year + 2}"  # Saison suivante
    
    for parcel in parcels:
        parcel_data = data[data['Parcelle'] == parcel]
        
        # Vu00e9rifier qu'il y a suffisamment de points temporels
        unique_seasons = parcel_data['Saison'].unique()
        if len(unique_seasons) < min_periods:
            # print(f"  Parcelle {parcel}: donnu00e9es insuffisantes ({len(unique_seasons)} saisons)")
            continue
        
        try:
            # Pru00e9parer les donnu00e9es de su00e9rie temporelle pour cette parcelle
            time_series = prepare_time_series_data(parcel_data)
            
            # Ajuster le modu00e8le
            model = SimpleExpSmoothing(time_series['Production au plant (g)'].values).fit(optimized=True)
            
            # Gu00e9nu00e9rer des pru00e9visions
            forecast = model.forecast(forecast_periods).values
            
            # Extraire les informations sur la parcelle
            species = parcel_data['Espu00e8ce'].iloc[0] if not parcel_data.empty else 'Inconnue'
            current_age = parcel_data[parcel_data['Saison'] == latest_season]['Age'].iloc[0] if not parcel_data[parcel_data['Saison'] == latest_season].empty else 0
            plants = parcel_data[parcel_data['Saison'] == latest_season]['Plants'].iloc[0] if not parcel_data[parcel_data['Saison'] == latest_season].empty else 0
            
            # Ru00e9cu00e9pu00e9rer la derniu00e8re observation
            latest_obs = time_series['Production au plant (g)'].iloc[-1] if not time_series.empty else 0
            
            # Calculer la projection pour la saison suivante (premier point de pru00e9vision)
            next_projection = forecast[0] if len(forecast) > 0 else 0
            
            # Stocker les ru00e9sultats
            projection = {
                'Parcelle': parcel,
                'Espu00e8ce': species,
                'Age': current_age + 1,  # u00c2ge pour la saison suivante
                'Saison': target_season,
                'Latest_Observation': latest_obs,
                'Projection': next_projection,
                'Change_Pct': ((next_projection / latest_obs) - 1) * 100 if latest_obs > 0 else float('inf'),
                'Plants': plants,
                'Total_Projection': next_projection * plants
            }
            
            parcel_projections.append(projection)
            
        except Exception as e:
            # print(f"  Erreur pour la parcelle {parcel}: {e}")
            continue
    
    # Convertir en DataFrame
    projections_df = pd.DataFrame(parcel_projections)
    
    # Afficher un ru00e9sumu00e9 des ru00e9sultats
    if not projections_df.empty:
        # Trouver les parcelles avec les plus grandes augmentations projetu00e9es
        top_increasing = projections_df.sort_values('Change_Pct', ascending=False).head(3)
        
        print(f"\n  Pru00e9visions pour la saison {target_season} (lissage exponentiel):")
        print(f"  Parcelles avec les plus fortes augmentations projetu00e9es :")
        for _, row in top_increasing.iterrows():
            print(f"    {row['Parcelle']} ({row['Espu00e8ce']}): {row['Latest_Observation']:.1f} g u2192 {row['Projection']:.1f} g ({row['Change_Pct']:.1f}%)")
        
        # Calculer la production totale projetu00e9e
        total_projected = projections_df['Total_Projection'].sum()
        total_plants = projections_df['Plants'].sum()
        avg_projection = total_projected / total_plants if total_plants > 0 else 0
        
        print(f"\n  Production totale projetu00e9e: {total_projected:.1f} g ({avg_projection:.1f} g/plant en moyenne)")
    else:
        print("  Aucune projection par parcelle disponible.")
    
    return projections_df
