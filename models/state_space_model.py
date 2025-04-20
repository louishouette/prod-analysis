#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.structural import UnobservedComponents
from models.shared.functions import gompertz

def build_state_space_model(data, rampup_data, gamma_fixed, A_fixed, beta_fixed):
    """
    Construction et ajustement du modèle hiérarchique bayésien à espace d'états.
    Utilise l'âge Brut comme variable d'âge.
    """
    print("\nConstruction du modèle hiérarchique bayésien à espace d'états...")
    
    # Création de la fonction cible basée sur les paramètres Gompertz
    def age_curve(age):
        return gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    
    # Préparation des données pour la modélisation à espace d'états
    valid_data = data.copy()
    
    print(f"Données du modèle à espace d'états: {len(valid_data)} observations valides")
    
    # Groupement par saison, parcelle, espèce et âge
    grouped_data = valid_data.groupby(['Saison', 'Parcelle', 'Espece', 'Age'])
    
    # Agrégation des données pour chaque groupe
    aggregated_data = grouped_data.agg({
        'Plants': 'sum',
        'Plants Productifs': 'sum',
        'Production au plant (g)': 'mean',
        'Poids produit (g)': 'sum'
    }).reset_index()
    
    # Calcul du ratio de production par rapport à la courbe de montée en production
    ratios = []
    model_data = []
    
    # Pour chaque saison, parcelle, espèce, âge
    for idx, row in aggregated_data.iterrows():
        age = row['Age']
        expected_production = age_curve(age)
        actual_production = row['Production au plant (g)']
        
        if expected_production > 0:
            ratio = actual_production / expected_production
        else:
            ratio = 0.0
        
        # Sauvegarde du ratio et des données associées
        ratios.append(ratio)
        model_data.append({
            'Saison': row['Saison'],
            'Parcelle': row['Parcelle'], 
            'Espèce': row['Espèce'],
            'Age': age,
            'Plants': row['Plants'],
            'Plants Productifs': row['Plants Productifs'],
            'Production au plant (g)': actual_production,
            'Production attendue (g)': expected_production,
            'Ratio': ratio
        })
    
    # Conversion en DataFrame
    ratio_df = pd.DataFrame(model_data)
    
    # Groupement par parcelle et espèce
    parcel_species_groups = ratio_df.groupby(['Parcelle', 'Espèce'])
    
    # Création des modèles à espace d'états pour chaque parcelle et espèce
    state_space_models = {}
    forecasts = {}
    
    for (parcel, species), group in parcel_species_groups:
        # Extraction des ratios de production pour cette parcelle/espèce
        try:
            grp_ratios = group['Ratio'].values
            
            if len(grp_ratios) < 2:
                print(f"Pas assez de données pour {parcel}/{species}, ignorant...")
                continue
            
            # Création du modèle à espace d'états
            model = UnobservedComponents(
                grp_ratios,
                level='local linear trend',
                irregular=True
            )
            
            # Ajustement du modèle
            res = model.fit(disp=False)
            
            # Prédiction pour la saison suivante
            forecast = res.forecast(steps=1)[0]
            
            # Sauvegarde du modèle et de la prédiction
            state_space_models[(parcel, species)] = res
            forecasts[(parcel, species)] = forecast
            
        except Exception as e:
            print(f"Erreur lors de la modélisation à espace d'états pour {parcel}/{species}: {e}")
            continue
    
    # Conversion des prédictions en DataFrame pour faciliter l'analyse
    forecast_data = [{
        'Parcelle': parcel,
        'Espèce': species,
        'Ratio prédit': forecast
    } for (parcel, species), forecast in forecasts.items()]
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Sauvegarde des ratios pour analyse ultérieure
    os.makedirs('generated/data/projections', exist_ok=True)
    ratio_df.to_csv('generated/data/projections/state_space_production_ratios.csv', index=False)
    print("Ratios de production du modèle à espace d'états sauvegardés dans 'generated/data/projections/state_space_production_ratios.csv'")
    
    return state_space_models, forecast_df, ratio_df

def project_state_space_production(parcel_species_forecasts, data, gamma_fixed, A_fixed, beta_fixed, next_season=2025):
    """
    Projette la production future en utilisant le modèle à espace d'états.
    Utilise l'âge Brut comme variable d'âge.
    """
    # Vérification si nous avons les données pour faire des projections
    if parcel_species_forecasts is None or parcel_species_forecasts.empty:
        print("Pas de prévisions disponibles pour le modèle à espace d'états.")
        return None
    
    # Préparation des données de base pour la projection
    next_season_str = f"{next_season} - {next_season + 1}"
    valid_data = data.copy()
    
    # Identifier la dernière saison dans les données
    latest_season = valid_data['Saison'].unique()[-1]
    latest_data = valid_data[valid_data['Saison'] == latest_season].copy()
    
    # Créer les données pour la prochaine saison
    next_season_data = latest_data.copy()
    next_season_data['Saison'] = next_season_str
    next_season_data['Age'] = next_season_data['Age'] + 1  # Incrémentation de l'âge
    
    # Calculer la production attendue avec le modèle Gompertz
    next_season_data['Expected_Production'] = next_season_data['Age'].apply(
        lambda age: gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    )
    
    # Appliquer les ratios de prévision du modèle à espace d'états
    # Créer un dictionnaire pour la recherche rapide des ratios prévus
    forecast_dict = {}
    for _, row in parcel_species_forecasts.iterrows():
        forecast_dict[(row['Parcelle'], row['Espèce'])] = row['Ratio prédit']
    
    # Appliquer les ratios pour calculer la production prévue
    forecasted_production = []
    for idx, row in next_season_data.iterrows():
        parcel = row['Parcelle']
        species = row['Espèce']
        expected_prod = row['Expected_Production']
        
        # Obtenir le ratio prevu pour cette parcelle/espèce
        ratio = forecast_dict.get((parcel, species), 1.0)  # 1.0 est la valeur par défaut si pas de prévision
        
        # Calculer la production prévue
        state_space_forecast = expected_prod * ratio
        
        # Calculer la production totale prévue
        total_expected_prod = expected_prod * row['Plants']
        total_state_space_forecast = state_space_forecast * row['Plants']
        
        # Ajouter à la liste avec toutes les informations
        next_season_data.at[idx, 'State_Space_Forecast'] = state_space_forecast
        next_season_data.at[idx, 'Total_Expected_Production'] = total_expected_prod
        next_season_data.at[idx, 'Total_State_Space_Forecast'] = total_state_space_forecast
    
    # Arrondir les valeurs numériques pour une meilleure lisibilité
    numeric_columns = [
        'Expected_Production', 'State_Space_Forecast',
        'Total_Expected_Production', 'Total_State_Space_Forecast'
    ]
    
    for col in numeric_columns:
        if col in next_season_data.columns:
            next_season_data[col] = next_season_data[col].round(6)
    
    # Sauvegarde des projections pour analyse ultérieure
    os.makedirs('generated/data/projections', exist_ok=True)
    next_season_data.to_csv(
        f'generated/data/projections/state_space_projections_{next_season}.csv',
        index=False
    )
    
    print(f"Projections du modèle à espace d'états sauvegardées dans 'generated/data/projections/state_space_projections_{next_season}.csv'")
    
    # Renvoyer à la fois le dataframe de projection et le next_season_data pour rester compatible
    # avec le code qui attend deux valeurs
    return next_season_data, next_season_data
    
