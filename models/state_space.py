#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from models.gompertz import gompertz

def build_state_space_model(data, rampup_data, gamma_fixed, A_fixed, beta_fixed):
    """
    Construction et ajustement du modu00e8le hiu00e9rarchique bayu00e9sien u00e0 espace d'u00e9tats.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    """
    print("\nConstruction du modu00e8le hiu00e9rarchique bayu00e9sien u00e0 espace d'u00e9tats...")
    
    # Cru00e9ation de la fonction cible basu00e9e sur les paramu00e8tres Gompertz
    def age_curve(age):
        return gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    
    # Pru00e9paration des donnu00e9es pour la modu00e9lisation u00e0 espace d'u00e9tats
    parcel_species_forecasts = {}
    unique_parcels = data['Parcelle'].unique()
    
    for parcel in unique_parcels:
        print(f"  Modu00e9lisation de la parcelle: {parcel}")
        parcel_data = data[data['Parcelle'] == parcel].copy()
        
        if len(parcel_data) <= 1:
            print(f"  Avertissement: Donnu00e9es insuffisantes pour la parcelle {parcel}. Utilisation de la courbe de ru00e9fu00e9rence.")
            continue
        
        # Calcul des u00e9carts par rapport u00e0 la courbe de ru00e9fu00e9rence
        parcel_data['Reference'] = parcel_data['Age'].apply(age_curve)
        parcel_data['Deviation'] = parcel_data['Production au plant (g)'] / parcel_data['Reference']
        
        # Remplacement des valeurs infinies ou NaN par 1 (ratio neutre)
        parcel_data['Deviation'] = parcel_data['Deviation'].replace([np.inf, -np.inf, np.nan], 1.0)
        
        # Calcul de la du00e9viation moyenne pour les donnu00e9es ru00e9centes (dernier point disponible)
        if not parcel_data.empty:
            recent_deviation = parcel_data.iloc[-1]['Deviation']
        else:
            recent_deviation = 1.0
        
        # Si la du00e9viation est aberrante, utiliser une valeur raisonnable
        if recent_deviation > 5.0 or recent_deviation < 0.2:
            print(f"  Avertissement: Du00e9viation aberrante pour {parcel} ({recent_deviation:.2f}). Ru00e9gularisation...")
            recent_deviation = np.clip(recent_deviation, 0.2, 5.0)
        
        # Modu00e9lisation des su00e9ries temporelles
        try:
            # Pru00e9paration de la su00e9rie temporelle
            if len(parcel_data) >= 3:
                # Modu00e8le structurel avec niveau et pente locaux
                model = UnobservedComponents(
                    parcel_data['Deviation'], 
                    level='local level', 
                    stochastic_level=True
                )
                results = model.fit(disp=False)
                
                # Ru00e9cupu00e9ration des u00e9tats estimu00e9s et pru00e9vision
                pred = results.get_prediction(start=0, end=len(parcel_data))
                forecast = pred.predicted_mean.iloc[-1]
                
                # Extraction de la tendance pour la derniu00e8re observation
                trend_states = results.states.filtered.iloc[-1, 0] if hasattr(results.states, 'filtered') else forecast
                
                # Calcul du facteur de correction final
                correction_factor = trend_states
            else:
                # Utiliser la moyenne des du00e9viations comme facteur de correction
                correction_factor = parcel_data['Deviation'].mean()
                print(f"  Nombre d'observations insuffisant pour {parcel}. Utilisation de la du00e9viation moyenne: {correction_factor:.2f}.")
            
            # Ru00e9gularisation du facteur de correction
            correction_factor = np.clip(correction_factor, 0.5, 2.0)
            
            # Cru00e9er l'objet de pru00e9vision pour cette parcelle
            species = parcel_data['Espu00e8ce'].iloc[0] if not parcel_data.empty else "Inconnu"
            current_age = parcel_data['Age'].max() if not parcel_data.empty else 0
            num_plants = parcel_data['Plants'].iloc[0] if not parcel_data.empty else 0
            
            parcel_forecast = {
                'Parcelle': parcel,
                'Espu00e8ce': species,
                'Age': current_age,
                'Plants': num_plants,
                'Correction': correction_factor,
                'Observation': parcel_data['Production au plant (g)'].iloc[-1] if not parcel_data.empty else 0,
                'Reference': age_curve(current_age)
            }
            
            parcel_species_forecasts[parcel] = parcel_forecast
            
        except Exception as e:
            print(f"  Erreur lors de la modu00e9lisation de {parcel}: {e}")
            # Assurer une valeur par du00e9faut mu00eame en cas d'erreur
            species = parcel_data['Espu00e8ce'].iloc[0] if not parcel_data.empty else "Inconnu"
            current_age = parcel_data['Age'].max() if not parcel_data.empty else 0
            num_plants = parcel_data['Plants'].iloc[0] if not parcel_data.empty else 0
            
            parcel_forecast = {
                'Parcelle': parcel,
                'Espu00e8ce': species,
                'Age': current_age,
                'Plants': num_plants,
                'Correction': 1.0,  # valeur neutre par du00e9faut
                'Observation': parcel_data['Production au plant (g)'].iloc[-1] if not parcel_data.empty else 0,
                'Reference': age_curve(current_age)
            }
            
            parcel_species_forecasts[parcel] = parcel_forecast
    
    return parcel_species_forecasts

def project_state_space_production(parcel_species_forecasts, data, gamma_fixed, A_fixed, beta_fixed, next_season=2025):
    """
    Projette la production future en utilisant le modu00e8le u00e0 espace d'u00e9tats.
    Utilise l'u00c2ge Brut comme variable d'u00e2ge.
    """
    print("\nProjection de la production future (modu00e8le u00e0 espace d'u00e9tats)...")
    
    # Cru00e9ation de la fonction cible basu00e9e sur les paramu00e8tres Gompertz
    def age_curve(age):
        return gompertz(age, A_fixed, beta_fixed, gamma_fixed)
    
    # Pru00e9paration du tableau de projections
    projections = []
    max_age = 12  # u00c2ge maximal pour les projections
    
    # Ru00e9cupu00e9rer la saison actuelle
    current_season = data['Saison'].max()
    
    # Identification des parcelles actuelles
    current_data = data[data['Saison'] == current_season].copy()
    
    # Projection pour chaque parcelle
    for _, parcel_row in current_data.iterrows():
        parcel = parcel_row['Parcelle']
        current_age = parcel_row['Age']
        next_age = current_age + 1  # u00c2ge pour la saison suivante
        num_plants = parcel_row['Plants']
        species = parcel_row['Espu00e8ce']
        
        # Ru00e9cupu00e9rer le facteur de correction de la parcelle
        correction_factor = 1.0  # Valeur par du00e9faut
        if parcel in parcel_species_forecasts:
            correction_factor = parcel_species_forecasts[parcel]['Correction']
        
        # Calcul des projections pour les prochaines annu00e9es
        for projection_age in range(next_age, min(next_age + 5, max_age + 1)):  # Limiter u00e0 max_age
            # Calcul de la production projetu00e9e pour cet u00e2ge
            reference_production = age_curve(projection_age)
            state_space_projection = reference_production * correction_factor
            
            # Ajout u00e0 la liste des projections
            projection = {
                'Saison': f"{next_season + projection_age - next_age}-{next_season + projection_age - next_age + 1}",
                'Parcelle': parcel,
                'Espu00e8ce': species,
                'Age': projection_age,
                'Plants': num_plants,
                'Reference': reference_production,
                'State_Space_Forecast': state_space_projection,
                'Total_Reference': reference_production * num_plants,
                'Total_State_Space_Forecast': state_space_projection * num_plants
            }
            projections.append(projection)
    
    # Conversion en DataFrame pour faciliter l'analyse
    projections_df = pd.DataFrame(projections)
    
    # Calcul des statistiques par saison
    if not projections_df.empty:
        season_stats = projections_df.groupby('Saison').agg({
            'Total_Reference': 'sum',
            'Total_State_Space_Forecast': 'sum',
            'Plants': 'sum'
        }).reset_index()
        
        # Calcul des moyennes par plant
        season_stats['Avg_Reference_Per_Plant'] = season_stats['Total_Reference'] / season_stats['Plants']
        season_stats['Avg_State_Space_Per_Plant'] = season_stats['Total_State_Space_Forecast'] / season_stats['Plants']
        
        print("\nProjections par saison (grammes):")
        for _, row in season_stats.iterrows():
            print(f"  Saison {row['Saison']}: {row['Total_State_Space_Forecast']:.1f}g ({row['Avg_State_Space_Per_Plant']:.1f}g/plant)")
        
        # Prendre la premiu00e8re saison comme projection principale
        next_season_forecast = projections_df[projections_df['Saison'] == projections_df['Saison'].min()].copy()
    else:
        next_season_forecast = pd.DataFrame()
        print("  Aucune donnu00e9e disponible pour la projection.")
    
    return projections_df, next_season_forecast
