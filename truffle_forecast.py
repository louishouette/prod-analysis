#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os

# Define file paths
SEASON_2023_2024_PATH = 'season_2023-2024.csv'
SEASON_2024_2025_PATH = 'season_2024-2025.csv'
RAMP_UP_PATH = 'ramp-up.csv'

# Utility function to parse French decimal format (comma as decimal separator)
def parse_french_decimal(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return value

# Load the dataset with correct encoding and separator
def load_data(file_path):
    try:
        # Using semicolon as separator and handling French decimal format (comma)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
        # Convert numeric columns with comma decimal separator
        numeric_cols = ['Taux de Productivité (%)', 'Poids produit (g)', 
                        'Poids moyen (g)', 'Production au plant (g)']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_french_decimal)
                
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Main function to analyze and forecast production
def forecast_production():
    print("\n===== TRUFFLE PRODUCTION FORECAST FOR 2024-2025 SEASON =====\n")
    
    # Load data
    print("Loading data...")
    season_2023_2024 = load_data(SEASON_2023_2024_PATH)
    season_2024_2025 = load_data(SEASON_2024_2025_PATH)
    ramp_up = load_data(RAMP_UP_PATH)
    
    if season_2023_2024 is None or season_2024_2025 is None or ramp_up is None:
        print("Failed to load required data files.")
        return
    
    # Combine seasons for analysis
    season_2023_2024['Season'] = '2023-2024'
    season_2024_2025['Season'] = '2024-2025'
    
    # Combine datasets for analysis
    combined_data = pd.concat([season_2023_2024, season_2024_2025])
    
    # Basic statistical analysis of production
    print("\n--- Basic Statistical Analysis ---")
    
    # Calculate total production per season
    total_production_by_season = combined_data.groupby('Season')['Poids produit (g)'].sum()
    print(f"\nTotal Production by Season (g):")
    for season, production in total_production_by_season.items():
        print(f"{season}: {production:.2f}g")
    
    # Calculate average production per plant by season
    season_plant_production = combined_data.groupby('Season').\
        apply(lambda x: x['Poids produit (g)'].sum() / x['Plants'].sum())
    print(f"\nAverage Production per Plant by Season (g):")
    for season, avg_prod in season_plant_production.items():
        print(f"{season}: {avg_prod:.2f}g/plant")
    
    # Analyze production by age
    print("\n--- Production Analysis by Age ---")
    age_production = combined_data.groupby(['Season', 'Age']).\
        apply(lambda x: {
            'total_plants': x['Plants'].sum(),
            'productive_plants': x['Plants Productifs'].sum(),
            'productivity_rate': x['Plants Productifs'].sum() / x['Plants'].sum() * 100 if x['Plants'].sum() > 0 else 0,
            'total_production': x['Poids produit (g)'].sum(),
            'avg_production_per_plant': x['Poids produit (g)'].sum() / x['Plants'].sum() if x['Plants'].sum() > 0 else 0
        }).apply(pd.Series)
    
    # Display production by age
    for season in sorted(age_production.index.get_level_values(0).unique()):
        print(f"\n{season} Production by Age:")
        season_data = age_production.loc[season]
        for age, row in season_data.iterrows():
            print(f"Age {age}: {row['avg_production_per_plant']:.2f}g/plant "
                  f"(Productivity Rate: {row['productivity_rate']:.2f}%)")
    
    # Analyze production by species
    print("\n--- Production Analysis by Species ---")
    species_production = combined_data.groupby(['Season', 'Espèce']).\
        apply(lambda x: {
            'total_plants': x['Plants'].sum(),
            'productive_plants': x['Plants Productifs'].sum(),
            'productivity_rate': x['Plants Productifs'].sum() / x['Plants'].sum() * 100 if x['Plants'].sum() > 0 else 0,
            'total_production': x['Poids produit (g)'].sum(),
            'avg_production_per_plant': x['Poids produit (g)'].sum() / x['Plants'].sum() if x['Plants'].sum() > 0 else 0
        }).apply(pd.Series)
    
    # Display production by species
    for season in sorted(species_production.index.get_level_values(0).unique()):
        print(f"\n{season} Production by Species:")
        season_data = species_production.loc[season]
        for species, row in season_data.iterrows():
            if row['total_plants'] > 0:
                print(f"{species}: {row['avg_production_per_plant']:.2f}g/plant "
                      f"(Plants: {row['total_plants']}, "
                      f"Productivity Rate: {row['productivity_rate']:.2f}%)")
    
    # Build prediction model for 2024-2025 season
    print("\n--- Building Forecast Model ---")
    
    # Prepare data for next season - start with the last season's plants
    forecast_base = season_2024_2025.copy()
    
    # Increment age by 1 for all plants for the next season
    forecast_base['Age'] = forecast_base['Age'] + 1
    forecast_base['Age Brut'] = forecast_base['Age Brut'] + 1
    
    # Function to predict productivity rate and production per plant based on historical data
    def predict_plant_productivity(row, age_data, species_data, ramp_up_data):
        age = row['Age']
        species = row['Espèce']
        
        # Get age-based expectation from ramp-up data
        target_production = 0
        if age in ramp_up_data['Age'].values:
            target_production = ramp_up_data[ramp_up_data['Age'] == age]['Production au plant (g)'].values[0]
        
        # Get historical productivity rate for this age and species
        hist_prod_rate = 0
        hist_prod_per_plant = 0
        
        if age in age_data.index:
            hist_prod_rate = age_data.loc[age, 'productivity_rate']
            hist_prod_per_plant = age_data.loc[age, 'avg_production_per_plant']
        
        species_prod_rate = 0
        if species in species_data.index:
            species_prod_rate = species_data.loc[species, 'productivity_rate']
        
        # Calculate expected productivity rate using historical data and ramp-up target
        # Use a weighted average of historical age-based and species-based productivity
        if hist_prod_rate > 0:
            expected_prod_rate = 0.7 * hist_prod_rate + 0.3 * species_prod_rate
        else:
            expected_prod_rate = species_prod_rate
        
        # For production per plant, use a blend of historical data and ramp-up target
        if hist_prod_per_plant > 0:
            expected_production = 0.6 * hist_prod_per_plant + 0.4 * target_production
        else:
            expected_production = 0.8 * target_production
        
        # Apply a growth factor based on the trend between 2023-2024 and 2024-2025 seasons
        growth_factor = 1.2  # Assuming 20% growth based on observed data trend
        
        return expected_prod_rate * growth_factor, expected_production * growth_factor
    
    # Create production forecast
    print("Generating forecast...")
    
    # Get the data for the most recent season for reference
    latest_age_production = age_production.loc["2024-2025"]
    latest_species_production = species_production.loc["2024-2025"]
    
    # Apply prediction function to each row
    forecast_results = []
    
    for _, row in forecast_base.iterrows():
        prod_rate, prod_per_plant = predict_plant_productivity(
            row, latest_age_production, latest_species_production, ramp_up)
        
        # Calculate expected productive plants based on productivity rate
        expected_productive_plants = int(row['Plants'] * (prod_rate / 100))
        # Calculate expected total production
        expected_total_production = row['Plants'] * prod_per_plant
        
        forecast_results.append({
            'Parcelle': row['Parcelle'],
            'Espèce': row['Espèce'],
            'Plants': row['Plants'],
            'Age': row['Age'],
            'Expected Productivity Rate (%)': prod_rate,
            'Expected Productive Plants': expected_productive_plants,
            'Expected Production per Plant (g)': prod_per_plant,
            'Expected Total Production (g)': expected_total_production
        })
    
    # Convert forecast results to DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    
    # Summarize results
    print("\n--- Forecast Summary for 2025-2026 Season ---")
    
    print(f"\nTotal Expected Production: {forecast_df['Expected Total Production (g)'].sum():.2f}g")
    print(f"Average Expected Production per Plant: {forecast_df['Expected Total Production (g)'].sum() / forecast_df['Plants'].sum():.2f}g/plant")
    
    # Summarize by parcelle
    parcelle_summary = forecast_df.groupby('Parcelle').agg({
        'Plants': 'sum',
        'Expected Productive Plants': 'sum',
        'Expected Total Production (g)': 'sum'
    })
    parcelle_summary['Expected Productivity Rate (%)'] = (parcelle_summary['Expected Productive Plants'] / parcelle_summary['Plants']) * 100
    parcelle_summary['Expected Production per Plant (g)'] = parcelle_summary['Expected Total Production (g)'] / parcelle_summary['Plants']
    
    print("\nExpected Production by Parcelle:")
    for parcelle, row in parcelle_summary.iterrows():
        print(f"Parcelle {parcelle}: {row['Expected Total Production (g)']:.2f}g "
              f"(Plants: {row['Plants']}, "
              f"Productivity Rate: {row['Expected Productivity Rate (%)']:.2f}%, "
              f"Per Plant: {row['Expected Production per Plant (g)']:.2f}g)")
    
    # Summarize by species
    species_summary = forecast_df.groupby('Espèce').agg({
        'Plants': 'sum',
        'Expected Productive Plants': 'sum',
        'Expected Total Production (g)': 'sum'
    })
    species_summary['Expected Productivity Rate (%)'] = (species_summary['Expected Productive Plants'] / species_summary['Plants']) * 100
    species_summary['Expected Production per Plant (g)'] = species_summary['Expected Total Production (g)'] / species_summary['Plants']
    
    print("\nExpected Production by Species:")
    for species, row in species_summary.iterrows():
        print(f"{species}: {row['Expected Total Production (g)']:.2f}g "
              f"(Plants: {row['Plants']}, "
              f"Productivity Rate: {row['Expected Productivity Rate (%)']:.2f}%, "
              f"Per Plant: {row['Expected Production per Plant (g)']:.2f}g)")
    
    # Save forecast to CSV
    output_file = 'forecast_2025-2026.csv'
    forecast_df.to_csv(output_file, sep=';', index=False)
    print(f"\nDetailed forecast saved to {output_file}")
    
    # Generate some visualizations
    create_visualizations(combined_data, forecast_df)

# Function to create visualizations
def create_visualizations(historical_data, forecast_data):
    print("\n--- Generating Visualizations ---")
    
    # Set up the output directory for plots
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Production by Season
    plt.figure(figsize=(10, 6))
    season_production = historical_data.groupby('Season')['Poids produit (g)'].sum()
    
    # Add forecast data
    seasons = list(season_production.index) + ['2025-2026']
    productions = list(season_production.values) + [forecast_data['Expected Total Production (g)'].sum()]
    
    plt.bar(seasons, productions, color=['blue', 'green', 'red'])
    plt.title('Total Truffle Production by Season')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'production_by_season.png'))
    
    # 2. Production by Age
    plt.figure(figsize=(12, 6))
    
    # Historical age production
    age_prod = historical_data.groupby('Age')['Production au plant (g)'].mean()
    
    # Forecast age production
    forecast_age_prod = forecast_data.groupby('Age')['Expected Production per Plant (g)'].mean()
    
    # Ramp-up target
    ramp_up = load_data(RAMP_UP_PATH)
    
    # Plot historical and forecast
    ages = sorted(set(list(age_prod.index) + list(forecast_age_prod.index)))
    
    plt.plot(age_prod.index, age_prod.values, 'o-', label='Historical Average')
    plt.plot(forecast_age_prod.index, forecast_age_prod.values, 's-', label='Forecast 2025-2026')
    plt.plot(ramp_up['Age'], ramp_up['Production au plant (g)'], '*--', label='Target Ramp-up')
    
    plt.title('Average Production per Plant by Age')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Production per Plant (g)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'production_by_age.png'))
    
    # 3. Production by Species
    plt.figure(figsize=(12, 6))
    
    # Historical species production
    species_prod = historical_data.groupby(['Season', 'Espèce'])['Poids produit (g)'].sum().unstack(fill_value=0)
    
    # Forecast species production
    forecast_species_prod = forecast_data.groupby('Espèce')['Expected Total Production (g)'].sum()
    
    # Add forecast to the historical data for plotting
    all_species = set(species_prod.columns) | set(forecast_species_prod.index)
    
    # Plot for each season including forecast
    seasons = list(species_prod.index) + ['2025-2026']
    species_data = {}
    
    # Prepare data for stacked bar chart
    for species in all_species:
        species_data[species] = []
        for season in species_prod.index:
            species_data[species].append(species_prod.loc[season, species] if species in species_prod.columns else 0)
        # Add forecast data
        species_data[species].append(forecast_species_prod[species] if species in forecast_species_prod.index else 0)
    
    # Plot stacked bar chart
    bottom = np.zeros(len(seasons))
    for species, data in species_data.items():
        plt.bar(seasons, data, bottom=bottom, label=species)
        bottom += np.array(data)
    
    plt.title('Truffle Production by Species and Season')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'production_by_species.png'))
    
    # 4. Productivity Rate by Age
    plt.figure(figsize=(10, 6))
    
    # Calculate productivity rate by age
    prod_rate_by_age = historical_data.groupby(['Season', 'Age']).apply(
        lambda x: x['Plants Productifs'].sum() / x['Plants'].sum() * 100 if x['Plants'].sum() > 0 else 0
    ).unstack(fill_value=0)
    
    # Forecast productivity rate by age
    forecast_prod_rate = forecast_data.groupby('Age')['Expected Productivity Rate (%)'].mean()
    
    # Plot historical productivity rates
    for season in prod_rate_by_age.index:
        plt.plot(prod_rate_by_age.columns, prod_rate_by_age.loc[season], 'o-', label=f'Season {season}')
    
    # Plot forecast productivity rate
    plt.plot(forecast_prod_rate.index, forecast_prod_rate.values, 's-', label='Forecast 2025-2026')
    
    plt.title('Productivity Rate by Age')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Productivity Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'productivity_rate_by_age.png'))
    
    print(f"Visualizations saved to {plots_dir}/ directory")

# Run the analysis
if __name__ == "__main__":
    forecast_production()
