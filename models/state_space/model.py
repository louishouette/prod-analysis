#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from models.gompertz.model import gompertz

def build_state_space_model(data, rampup_data, gamma_fixed, A_fixed, beta_fixed):
    """
    Construction et ajustement du modèle hiérarchique bayésien à espace d'états.
    Utilise l'Âge Brut comme variable d'âge.
    """
    print("\nConstruction du modèle hiérarchique bayésien à espace d'états...")
    
    # Création de la fonction cible basée sur les paramètres Gompertz
    def age_curve(age):
        return gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    
    # Préparation des données pour la modélisation à espace d'états
    valid_data = data.copy()
    
    print(f"Données du modèle à espace d'états: {len(valid_data)} observations valides")
    
    # Groupement par saison, parcelle, espèce et âge
    grouped = valid_data.groupby(['Saison', 'Parcelle', 'Espèce', 'Age'])
    prod_time_series = grouped['Production au plant (g)'].mean().reset_index()
    
    # Conversion de la saison en année pour le tri chronologique
    prod_time_series['Year'] = prod_time_series['Saison'].str.split(' - ').str[0].astype(int)
    
    # Tri par parcelle, espèce, année et âge
    prod_time_series = prod_time_series.sort_values(['Parcelle', 'Espèce', 'Year', 'Age'])
    
    # Calcul de la production attendue basée sur la courbe d'âge
    prod_time_series['Expected_Production'] = prod_time_series['Age'].apply(age_curve)
    
    # Calcul du ratio de déviation: réel / attendu
    prod_time_series['Deviation_Ratio'] = prod_time_series['Production au plant (g)'] / prod_time_series['Expected_Production']
    
    # Remplacement des valeurs infinies et NaN par 0 ou 1 selon le cas
    prod_time_series['Deviation_Ratio'] = prod_time_series['Deviation_Ratio'].replace([np.inf, -np.inf], np.nan)
    prod_time_series['Deviation_Ratio'] = prod_time_series['Deviation_Ratio'].fillna(1.0)  # Déviation neutre si incertain
    
    # Transformation logarithmique du ratio de déviation pour un meilleur ajustement (plus proche d'une distribution normale)
    prod_time_series['Log_Deviation'] = np.log(prod_time_series['Deviation_Ratio'])
    
    # Configuration des modèles à espace d'états pour chaque combinaison parcelle-espèce
    parcel_species_models = {}
    parcel_species_forecasts = {}
    
    # Obtention des combinaisons parcelle-espèce avec suffisamment de points de données
    parcel_species_combos = prod_time_series.groupby(['Parcelle', 'Espèce']).size()
    valid_combos = parcel_species_combos[parcel_species_combos >= 2].index.tolist()
    
    print(f"Ajustement des modèles à espace d'états pour {len(valid_combos)} combinaisons parcelle-espèce...")
    
    for parcel, species in valid_combos:
        # Filtrage des données pour cette combinaison parcelle-espèce
        combo_data = prod_time_series[(prod_time_series['Parcelle'] == parcel) & 
                                     (prod_time_series['Espèce'] == species)]
        
        if len(combo_data) < 2:
            continue  # Ignorer si pas assez de points temporels
            
        # Création d'un modèle pour la déviation logarithmique avec composante de niveau local
        try:
            model = UnobservedComponents(
                combo_data['Log_Deviation'], 
                level='local level', 
                stochastic_level=True
            )
            
            # Ajustement du modèle
            model_fit = model.fit(disp=False)
            
            # Stockage du modèle ajusté
            parcel_species_models[(parcel, species)] = model_fit
            
            # Prévision d'un pas en avant (saison suivante)
            forecast = model_fit.forecast(steps=1)
            forecast_value = np.exp(forecast[0])  # Reconversion depuis l'échelle logarithmique
            
            # Vérification que la valeur prévue est raisonnable (pas extrême)
            if 0.1 <= forecast_value <= 5.0:  # Limite à une plage raisonnable
                parcel_species_forecasts[(parcel, species)] = forecast_value
                print(f"  - {parcel}-{species}: Modèle ajusté avec succès, déviation prévue: {forecast_value:.2f}")
            else:
                print(f"  - {parcel}-{species}: Valeur prévue {forecast_value:.2f} hors plage raisonnable, utilisation de la valeur par défaut")
                # Utilisation d'une valeur par défaut plus raisonnable basée sur la performance récente
                recent_deviation = combo_data.iloc[-1]['Deviation_Ratio']
                parcel_species_forecasts[(parcel, species)] = recent_deviation
            
        except Exception as e:
            print(f"  - {parcel}-{species}: Impossible d'ajuster le modèle: {e}")
            # Utilisation du ratio de déviation le plus récent comme prévision
            if not combo_data.empty:
                recent_deviation = combo_data.iloc[-1]['Deviation_Ratio']
                parcel_species_forecasts[(parcel, species)] = recent_deviation
                print(f"    Utilisation du ratio de déviation le plus récent: {recent_deviation:.2f}")
            else:
                # Par défaut à neutre (1.0) si nous n'avons pas de données
                parcel_species_forecasts[(parcel, species)] = 1.0
    
    return parcel_species_models, parcel_species_forecasts, prod_time_series

def project_state_space_production(parcel_species_forecasts, data, gamma_fixed, A_fixed, beta_fixed, next_season=2025):
    """
    Projette la production future en utilisant le modèle à espace d'états.
    Utilise l'Âge Brut comme variable d'âge.
    """
    # Création de la fonction d'âge pour la production attendue
    def age_curve(age):
        return gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    
    # Obtention des données les plus récentes pour projeter vers l'avant
    latest_season = data['Saison'].str.split(' - ').str[0].astype(int).max()
    next_season_name = f"{next_season} - {next_season+1}"
    
    latest_data = data[data['Saison'].str.contains(str(latest_season))].copy()
    
    # Préparation des données de projection: incrémentation de l'âge de 1 pour la saison suivante
    proj_data = latest_data.copy()
    proj_data['Saison'] = next_season_name
    proj_data['Age'] = proj_data['Age'] + 1
    
    # Calcul de la production attendue de base en fonction de l'âge
    proj_data['Expected_Production'] = proj_data['Age'].apply(age_curve)
    
    # Limitation de la production attendue à des valeurs réalistes (max 100g par plante)
    proj_data['Expected_Production'] = np.minimum(proj_data['Expected_Production'], 100.0)
    
    print(f"Plage de production attendue: {proj_data['Expected_Production'].min():.1f}g à {proj_data['Expected_Production'].max():.1f}g par plante")
    
    # Application des prévisions du modèle à espace d'états comme facteurs d'ajustement
    proj_data['State_Space_Forecast'] = np.nan
    
    for idx, row in proj_data.iterrows():
        parcel = row['Parcelle']
        species = row['Espèce']
        expected_prod = row['Expected_Production']
        
        # Application du ratio de déviation prévu s'il est disponible (avec des limites raisonnables)
        deviation_ratio = parcel_species_forecasts.get((parcel, species), 1.0)
        
        # Limitation des ratios de déviation à des plages raisonnables (0.1 à 2.0)
        if deviation_ratio < 0.1:
            deviation_ratio = 0.1
            print(f"  - Limitation du ratio de déviation faible pour {parcel}-{species} à 0.1")
        elif deviation_ratio > 2.0:
            deviation_ratio = 2.0
            print(f"  - Limitation du ratio de déviation élevé pour {parcel}-{species} à 2.0")
            
        proj_data.at[idx, 'State_Space_Forecast'] = expected_prod * deviation_ratio
    
    # Calcul de la production totale attendue par parcelle-espèce (en grammes)
    proj_data['Total_Expected_Production'] = proj_data['Expected_Production'] * proj_data['Plants']
    proj_data['Total_State_Space_Forecast'] = proj_data['State_Space_Forecast'] * proj_data['Plants']
    
    # Vérification des valeurs déraisonnables dans la production totale
    if proj_data['Total_Expected_Production'].max() > 10000 or proj_data['Total_State_Space_Forecast'].max() > 10000:
        print("AVERTISSEMENT: Certaines valeurs de production totale sont inhabituellement élevées. Limitation à des valeurs raisonnables.")
        # Limitation de la production totale à 10kg par combinaison parcelle-espèce
        proj_data['Total_Expected_Production'] = np.minimum(proj_data['Total_Expected_Production'], 10000.0)
        proj_data['Total_State_Space_Forecast'] = np.minimum(proj_data['Total_State_Space_Forecast'], 10000.0)
    
    # Création de projections par âge
    age_projections = {}
    ages = np.arange(1, 13)  # Âges 1-12
    
    for parcel in proj_data['Parcelle'].unique():
        parcel_data = proj_data[proj_data['Parcelle'] == parcel]
        age_proj = np.zeros(len(ages))
        
        for i, age in enumerate(ages):
            age_rows = parcel_data[parcel_data['Age'] == age]
            if not age_rows.empty:
                # Utilisation de la prévision à espace d'états si disponible, sinon utilisation de la production attendue
                if np.isnan(age_rows['State_Space_Forecast']).all():
                    age_proj[i] = age_rows['Expected_Production'].mean()
                else:
                    age_proj[i] = age_rows['State_Space_Forecast'].mean()
            else:
                # Si pas de données pour cet âge, utilisation de la courbe d'âge
                age_proj[i] = age_curve(age)
        
        age_projections[parcel] = age_proj
    
    # Conversion en DataFrame
    ss_proj_df = pd.DataFrame(age_projections, index=ages)
    ss_proj_df.index.name = 'Age'
    
    return ss_proj_df, proj_data
