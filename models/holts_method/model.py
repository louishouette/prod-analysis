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
    
    import pandas as pd
    if not pd.api.types.is_integer_dtype(time_series.index):
        try:
            time_series.index = time_series.index.astype(int)
        except Exception:
            print("[AVERTISSEMENT] L'index de la série temporelle n'est pas convertible en années (int). Prévisions non fiables.")
    if time_series.empty or len(time_series) < 2:
        print("Erreur: Pas assez de données pour construire un modèle de Holt")
        return None, time_series
    
    # Créer et ajuster le modèle
    model = Holt(time_series['Production au plant (g)'], exponential=exponential)
    
    if alpha is None or beta is None:
        # Optimisation automatique des paramètres
        try:
            results = model.fit(optimized=True, use_brute=True)
            print(f"Paramètres optimisés: alpha={results.params['smoothing_level']:.4f}, beta={results.params['smoothing_trend']:.4f}")
        except Exception as e:
            print(f"Erreur lors de la construction du modèle de Holt: {e}")
            return None, time_series
    else:
        # Utilisation des paramètres fournis
        try:
            results = model.fit(smoothing_level=alpha, smoothing_trend=beta)
            print(f"Paramètres utilisés: alpha={alpha:.4f}, beta={beta:.4f}")
        except Exception as e:
            print(f"Erreur lors de la construction du modèle de Holt: {e}")
            return None, time_series
    
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
    # S'assurer que l'index est bien d'entiers (année)
    import pandas as pd
    if not pd.api.types.is_integer_dtype(time_series.index):
        try:
            time_series.index = time_series.index.astype(int)
        except Exception:
            print("[AVERTISSEMENT] L'index de la série temporelle n'est pas convertible en années (int). Prévisions non fiables.")
    # Faire les prévisions
    forecast = model_results.forecast(steps=forecast_periods)
    # Créer un DataFrame pour les prévisions avec un index d'entiers
    last_year = int(time_series.index[-1])
    forecast_years = range(last_year + 1, last_year + forecast_periods + 1)
    forecast_index = pd.Index(forecast_years, dtype=int, name='Year')
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
        ts = group_data.groupby('Saison')['Production au plant (g)'].mean().reset_index()
        ts['Year'] = ts['Saison'].str.split(' - ').str[0].astype(int)
        ts = ts.sort_values('Year')
        ts.set_index('Year', inplace=True)
        ts = ts[~ts['Production au plant (g)'].isna()]
        
        import pandas as pd
        if not pd.api.types.is_integer_dtype(ts.index):
            try:
                ts.index = ts.index.astype(int)
            except Exception:
                print(f"[AVERTISSEMENT] Parcelle {parcel}, Espèce {species} : index non convertible en années (int). Prévisions non fiables.")
        if len(ts) < 3:
            print(f"[AVERTISSEMENT] Parcelle {parcel} : série trop courte pour modélisation (moins de 3 points).")
            continue
        if ts['Production au plant (g)'].nunique() == 1:
            print(f"[AVERTISSEMENT] Parcelle {parcel} : série constante (pas de variation de production).")
            continue
        
        # S'il y a assez de points pour la méthode de Holt
        try:
            # Ajuster le modèle
            model = Holt(ts['Production au plant (g)'], exponential=exponential)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                res = model.fit(optimized=True)
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
    # S'assurer que l'index est bien d'entiers (année)
    import pandas as pd
    if not pd.api.types.is_integer_dtype(time_series.index):
        try:
            time_series.index = time_series.index.astype(int)
        except Exception:
            print("[AVERTISSEMENT] L'index de la série temporelle n'est pas convertible en années (int). Prévisions non fiables.")
    print("\nComparaison des modèles de Holt linéaire et exponentiel:")
    # Ajuster le modèle linéaire
    import pandas as pd
    ts_linear = time_series.copy()
    import pandas as pd
    # Préparer série pour Holt avec RangeIndex strictement consécutif
    years = ts_linear.index.values
    if pd.api.types.is_integer_dtype(years) and len(years) > 1:
        full_range = pd.RangeIndex(start=years[0], stop=years[-1]+1)
        ts_linear = ts_linear.reindex(full_range)
        ts_linear.index.name = 'Year'
        n_missing = ts_linear['Production au plant (g)'].isna().sum()
        if n_missing > 0:
            print(f"[INFO] {n_missing} années manquantes (linéaire). Interpolation automatique...")
            ts_linear['Production au plant (g)'] = ts_linear['Production au plant (g)'].interpolate(method='linear')
            still_missing = ts_linear['Production au plant (g)'].isna().sum()
            if still_missing > 0:
                print(f"[AVERTISSEMENT] {still_missing} années restent manquantes après interpolation (linéaire). Ignorées.")
            ts_linear = ts_linear[~ts_linear['Production au plant (g)'].isna()]
    if ts_linear.empty or len(ts_linear) < 2:
        print("[ERREUR] Série trop courte après nettoyage/interpolation pour Holt linéaire. Fit impossible.")
        linear_results = None
    else:
        prod_series = pd.Series(
            ts_linear['Production au plant (g)'].values,
            index=pd.RangeIndex(start=ts_linear.index.min(), stop=ts_linear.index.max()+1),
            name='Production au plant (g)'
        )
        linear_model = Holt(prod_series, exponential=False)
        linear_results = linear_model.fit(optimized=True)

    # Ajuster le modèle exponentiel
    ts_exp = time_series.copy()
    years = ts_exp.index.values
    if pd.api.types.is_integer_dtype(years) and len(years) > 1:
        full_range = pd.RangeIndex(start=years[0], stop=years[-1]+1)
        ts_exp = ts_exp.reindex(full_range)
        ts_exp.index.name = 'Year'
        n_missing = ts_exp['Production au plant (g)'].isna().sum()
        if n_missing > 0:
            print(f"[INFO] {n_missing} années manquantes (exponentiel). Interpolation automatique...")
            ts_exp['Production au plant (g)'] = ts_exp['Production au plant (g)'].interpolate(method='linear')
            still_missing = ts_exp['Production au plant (g)'].isna().sum()
            if still_missing > 0:
                print(f"[AVERTISSEMENT] {still_missing} années restent manquantes après interpolation (exponentiel). Ignorées.")
            ts_exp = ts_exp[~ts_exp['Production au plant (g)'].isna()]
    if ts_exp.empty or len(ts_exp) < 2:
        print("[ERREUR] Série trop courte après nettoyage/interpolation pour Holt exponentiel. Fit impossible.")
        exp_results = None
    else:
        prod_series_exp = pd.Series(
            ts_exp['Production au plant (g)'].values,
            index=pd.RangeIndex(start=ts_exp.index.min(), stop=ts_exp.index.max()+1),
            name='Production au plant (g)'
        )
        exp_model = Holt(prod_series_exp, exponential=True)
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
