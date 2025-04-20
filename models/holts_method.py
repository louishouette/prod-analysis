#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from models.linear_trend import prepare_time_series_data


def build_holts_trend_model(data, alpha=None, beta=None, exponential=False):
    """
    Construit un modu00e8le de Holt (tendance linu00e9aire) pour la production de truffes.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    
    Paramu00e8tres:
    -----------
    data : DataFrame des donnu00e9es de production
    alpha : paramu00e8tre de lissage pour le niveau (None pour estimation automatique)
    beta : paramu00e8tre de lissage pour la tendance (None pour estimation automatique)
    exponential : si True, utilise une tendance exponentielle plutu00f4t que linu00e9aire
    
    Retourne:
    ---------
    modu00e8le ajustu00e9, donnu00e9es de su00e9rie temporelle
    """
    print(f"\nConstruction du modu00e8le de Holt ({'exponentiel' if exponential else 'linu00e9aire'})...")
    
    # Pru00e9parer les donnu00e9es de su00e9rie temporelle
    time_series = prepare_time_series_data(data)
    
    # Vu00e9rifier qu'il y a suffisamment de donnu00e9es
    if len(time_series) < 3:  # Holt a besoin d'au moins 3 points pour estimer niveau + tendance
        print("  Avertissement: Nombre insuffisant de points temporels pour le modu00e8le de Holt.")
        return None, time_series
    
    # Forcer un index temporel consu00e9cutif
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
        # Ajuster le modu00e8le de Holt avec tendance
        if alpha is None and beta is None:
            # Optimisation automatique des paramu00e8tres
            model = Holt(y, exponential=exponential).fit(optimized=True)
            print(f"  Paramu00e8tres optimisu00e9s: alpha={model.params['smoothing_level']:.3f}, beta={model.params['smoothing_trend']:.3f}")
        else:
            # Utilisation des paramu00e8tres fournis
            alpha_val = alpha if alpha is not None else 0.8
            beta_val = beta if beta is not None else 0.2
            model = Holt(y, exponential=exponential).fit(smoothing_level=alpha_val, smoothing_trend=beta_val)
            print(f"  Paramu00e8tres spu00e9cifiu00e9s: alpha={alpha_val:.3f}, beta={beta_val:.3f}")
        
        return model, time_series
    except Exception as e:
        print(f"  Erreur lors de l'ajustement du modu00e8le de Holt: {e}")
        return None, time_series


def project_holts_trend(model_results, time_series, forecast_periods=3):
    """
    Projette la production future en utilisant le modu00e8le de Holt.
    
    Paramu00e8tres:
    -----------
    model_results : ru00e9sultats du modu00e8le de Holt
    time_series : su00e9rie temporelle des donnu00e9es historiques
    forecast_periods : nombre de pu00e9riodes u00e0 pru00e9voir
    
    Retourne:
    ---------
    DataFrame avec les valeurs historiques et les pru00e9visions
    """
    if model_results is None:
        print("  Avertissement: Modu00e8le de Holt non disponible pour les projections.")
        return None, None
    
    print(f"\nProjection selon la mu00e9thode de Holt pour {forecast_periods} pu00e9riodes...")
    
    try:
        # Extraire les annu00e9es historiques et les donnu00e9es de production
        historical_years = time_series['Year'].values
        historical_production = time_series['Production au plant (g)'].values
        
        # Gu00e9nu00e9rer les pru00e9visions
        forecast = model_results.forecast(forecast_periods).values
        
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
        print(f"  Erreur lors de la projection du modu00e8le de Holt: {e}")
        return None, None


def holts_method_by_parcel(data, forecast_periods=2, exponential=False):
    """
    Applique la mu00e9thode de Holt u00e0 chaque combinaison parcelle-espu00e8ce.
    
    Retourne:
    ---------
    DataFrame avec les pru00e9visions pour chaque parcelle-espu00e8ce
    """
    print("\nAnalyse par mu00e9thode de Holt par parcelle...")
    
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
        
        # Vu00e9rifier qu'il y a suffisamment de points temporels (au moins 3 pour Holt)
        unique_seasons = parcel_data['Saison'].unique()
        if len(unique_seasons) < 3:
            continue
        
        try:
            # Pru00e9parer les donnu00e9es de su00e9rie temporelle pour cette parcelle
            time_series = prepare_time_series_data(parcel_data)
            
            # Ajuster le modu00e8le
            model = Holt(
                time_series['Production au plant (g)'].values, 
                exponential=exponential
            ).fit(optimized=True)
            
            # Gu00e9nu00e9rer des pru00e9visions
            forecast = model.forecast(forecast_periods).values
            
            # Extraire les informations sur la parcelle
            species = parcel_data['Espu00e8ce'].iloc[0] if not parcel_data.empty else 'Inconnue'
            current_age = parcel_data[parcel_data['Saison'] == latest_season]['Age'].iloc[0] if not parcel_data[parcel_data['Saison'] == latest_season].empty else 0
            plants = parcel_data[parcel_data['Saison'] == latest_season]['Plants'].iloc[0] if not parcel_data[parcel_data['Saison'] == latest_season].empty else 0
            
            # Ru00e9cupu00e9rer la derniu00e8re observation
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
            continue
    
    # Convertir en DataFrame
    projections_df = pd.DataFrame(parcel_projections)
    
    # Afficher un ru00e9sumu00e9 des ru00e9sultats
    if not projections_df.empty:
        # Calculer la production totale projetu00e9e
        total_projected = projections_df['Total_Projection'].sum()
        total_plants = projections_df['Plants'].sum()
        avg_projection = total_projected / total_plants if total_plants > 0 else 0
        
        print(f"\n  Pru00e9visions pour la saison {target_season} (mu00e9thode de Holt {('exponentielle' if exponential else 'linu00e9aire')}):")
        print(f"  Production totale projetu00e9e: {total_projected:.1f} g ({avg_projection:.1f} g/plant en moyenne)")
    else:
        print("  Aucune projection par parcelle disponible.")
    
    return projections_df


def compare_holts_models(data, forecast_periods=3):
    """
    Compare les modu00e8les de Holt linu00e9aire et exponentiel.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    
    Retourne:
    ---------
    ru00e9sultats des deux modu00e8les et leurs mu00e9triques
    """
    print("\nComparaison des variantes du modu00e8le de Holt...")
    
    # Construire le modu00e8le de Holt linu00e9aire
    linear_model, time_series = build_holts_trend_model(data, exponential=False)
    
    # Construire le modu00e8le de Holt exponentiel
    exp_model, _ = build_holts_trend_model(data, exponential=True)
    
    # Vu00e9rifier si les modu00e8les ont u00e9tu00e9 construits avec succu00e8s
    if linear_model is None and exp_model is None:
        print("  Aucun modu00e8le de Holt n'a pu u00eatre construit. Comparu00e9 annulu00e9e.")
        return None, None, None, None
    
    # Projeter les deux modu00e8les
    linear_df, linear_forecast = project_holts_trend(linear_model, time_series, forecast_periods) if linear_model else (None, None)
    exp_df, exp_forecast = project_holts_trend(exp_model, time_series, forecast_periods) if exp_model else (None, None)
    
    # Comparaison des mu00e9triques si les deux modu00e8les sont disponibles
    if linear_model is not None and exp_model is not None:
        # Calculer les erreurs de pru00e9vision in-sample
        y_actual = time_series['Production au plant (g)'].values
        linear_fitted = linear_model.fittedvalues
        exp_fitted = exp_model.fittedvalues
        
        # MSE (Mean Squared Error)
        linear_mse = np.mean((y_actual[1:] - linear_fitted) ** 2)  # Ignorer le premier point pour u00e9quilibrer
        exp_mse = np.mean((y_actual[1:] - exp_fitted) ** 2)
        
        # MAPE (Mean Absolute Percentage Error)
        linear_mape = np.mean(np.abs((y_actual[1:] - linear_fitted) / y_actual[1:])) * 100 if np.all(y_actual[1:] != 0) else np.inf
        exp_mape = np.mean(np.abs((y_actual[1:] - exp_fitted) / y_actual[1:])) * 100 if np.all(y_actual[1:] != 0) else np.inf
        
        print("\n  Comparaison des mu00e9triques d'erreur in-sample:")
        print(f"  MSE - Linu00e9aire: {linear_mse:.2f}, Exponentiel: {exp_mse:.2f}")
        if not np.isinf(linear_mape) and not np.isinf(exp_mape):
            print(f"  MAPE - Linu00e9aire: {linear_mape:.2f}%, Exponentiel: {exp_mape:.2f}%")
        
        # Identifier le modu00e8le pru00e9fu00e9ru00e9 basu00e9 sur MSE
        preferred_model = "linu00e9aire" if linear_mse <= exp_mse else "exponentiel"
        print(f"\n  Modu00e8le pru00e9fu00e9ru00e9 basu00e9 sur MSE: {preferred_model}")
        
        # Calculer la divergence des pru00e9visions
        if linear_forecast is not None and exp_forecast is not None:
            final_diff_pct = abs((linear_forecast[-1] - exp_forecast[-1]) / linear_forecast[-1] * 100) if linear_forecast[-1] != 0 else np.inf
            print(f"  Divergence entre les pru00e9visions finales: {final_diff_pct:.1f}%")
    
    # Visualiser les ru00e9sultats
    try:
        if linear_df is not None and exp_df is not None:
            plt.figure(figsize=(12, 6))
            
            # Donnu00e9es historiques
            historical_mask = linear_df['Type'] == 'Historical'
            plt.plot(linear_df.loc[historical_mask, 'Year'], linear_df.loc[historical_mask, 'Production'], 
                    'ko-', label='Donnu00e9es historiques')
            
            # Pru00e9visions linu00e9aires
            forecast_mask = linear_df['Type'] == 'Forecast'
            plt.plot(linear_df.loc[forecast_mask, 'Year'], linear_df.loc[forecast_mask, 'Production'], 
                    'b-', label='Pru00e9vision Holt linu00e9aire')
            
            # Pru00e9visions exponentielles
            forecast_mask = exp_df['Type'] == 'Forecast'
            plt.plot(exp_df.loc[forecast_mask, 'Year'], exp_df.loc[forecast_mask, 'Production'], 
                    'r-', label='Pru00e9vision Holt exponentielle')
            
            plt.title('Comparaison des modu00e8les de Holt linu00e9aire et exponentiel')
            plt.xlabel('Annu00e9e')
            plt.ylabel('Production moyenne (g/plant)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('generated/plots/time_series/holts_methods_comparison.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"  Erreur lors de la visualisation des modu00e8les de Holt: {e}")
    
    return linear_model, exp_model, linear_forecast, exp_forecast
