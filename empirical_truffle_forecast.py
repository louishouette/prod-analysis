#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    print("\n===== EMPIRICAL TRUFFLE PRODUCTION FORECAST FOR 2025-2026 SEASON =====\n")
    
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
    season_plant_production = combined_data.groupby('Season').apply(lambda x: x['Poids produit (g)'].sum() / x['Plants'].sum())
    print(f"\nAverage Production per Plant by Season (g):")
    for season, avg_prod in season_plant_production.items():
        print(f"{season}: {avg_prod:.2f}g/plant")
    
    # Calculate growth rate between seasons
    growth_rate = season_plant_production['2024-2025'] / season_plant_production['2023-2024']
    print(f"Growth rate from 2023-2024 to 2024-2025: {growth_rate:.2f}x")
    
    # Analyze production by age
    print("\n--- Production Analysis by Age ---")
    age_production = combined_data.groupby(['Season', 'Age']).apply(lambda x: {
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
    
    # Calculate growth rates by age
    print("\n--- Growth Rates by Age ---")
    ages_2023_2024 = set(age_production.loc['2023-2024'].index)
    ages_2024_2025 = set(age_production.loc['2024-2025'].index)
    
    common_ages = ages_2023_2024.intersection(ages_2024_2025)
    age_growth_rates = {}
    age_productivity_growth = {}
    
    for age in common_ages:
        prod_2023_2024 = age_production.loc['2023-2024'].loc[age, 'avg_production_per_plant']
        prod_2024_2025 = age_production.loc['2024-2025'].loc[age, 'avg_production_per_plant']
        
        # Calculate growth rate for production per plant
        if prod_2023_2024 > 0:
            growth = prod_2024_2025 / prod_2023_2024
        else:
            # If previous production was zero, use overall growth rate
            growth = growth_rate
        
        # Calculate growth rate for productivity rate
        prod_rate_2023_2024 = age_production.loc['2023-2024'].loc[age, 'productivity_rate']
        prod_rate_2024_2025 = age_production.loc['2024-2025'].loc[age, 'productivity_rate']
        
        if prod_rate_2023_2024 > 0:
            prod_growth = prod_rate_2024_2025 / prod_rate_2023_2024
        else:
            # If previous productivity was zero, use a conservative value
            prod_growth = 1.5
        
        age_growth_rates[age] = growth
        age_productivity_growth[age] = prod_growth
        
        print(f"Age {age}: Production growth = {growth:.2f}x, Productivity growth = {prod_growth:.2f}x")
    
    # Analyze production by species
    print("\n--- Production Analysis by Species ---")
    species_production = combined_data.groupby(['Season', 'Espèce']).apply(lambda x: {
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
    
    # Calculate growth rates by species
    print("\n--- Growth Rates by Species ---")
    species_2023_2024 = set(species_production.xs('2023-2024', level=0).index)
    species_2024_2025 = set(species_production.xs('2024-2025', level=0).index)
    
    common_species = species_2023_2024.intersection(species_2024_2025)
    species_growth_rates = {}
    species_productivity_growth = {}
    
    for species in common_species:
        prod_2023_2024 = species_production.loc[('2023-2024', species), 'avg_production_per_plant']
        prod_2024_2025 = species_production.loc[('2024-2025', species), 'avg_production_per_plant']
        
        # Calculate growth rate for production per plant
        if prod_2023_2024 > 0:
            growth = prod_2024_2025 / prod_2023_2024
        else:
            # If previous production was zero, use overall growth rate
            growth = growth_rate
        
        # Calculate growth rate for productivity rate
        prod_rate_2023_2024 = species_production.loc[('2023-2024', species), 'productivity_rate']
        prod_rate_2024_2025 = species_production.loc[('2024-2025', species), 'productivity_rate']
        
        if prod_rate_2023_2024 > 0:
            prod_growth = prod_rate_2024_2025 / prod_rate_2023_2024
        else:
            # If previous productivity was zero, use a conservative value
            prod_growth = 1.5
        
        species_growth_rates[species] = growth
        species_productivity_growth[species] = prod_growth
        
        print(f"{species}: Production growth = {growth:.2f}x, Productivity growth = {prod_growth:.2f}x")
    
    # Build prediction model for 2025-2026 season based purely on historical data
    print("\n--- Building Empirical Forecast Model ---")
    
    # Prepare data for next season - start with the last season's plants
    forecast_base = season_2024_2025.copy()
    
    # Increment age by 1 for all plants for the next season
    forecast_base['Age'] = forecast_base['Age'] + 1
    forecast_base['Age Brut'] = forecast_base['Age Brut'] + 1
    
    # Function to predict productivity and production based on historical patterns
    def predict_empirical_productivity(row, latest_age_data, latest_species_data, age_growth_rates, species_growth_rates, age_productivity_growth, species_productivity_growth):
        age = row['Age']
        species = row['Espèce']
        last_age = age - 1  # The age in the previous season
        
        # Get historical productivity rate and production per plant for this species
        species_productivity = 0
        species_production = 0
        
        if species in latest_species_data.index:
            species_productivity = latest_species_data.loc[species, 'productivity_rate']
            species_production = latest_species_data.loc[species, 'avg_production_per_plant']
        
        # Get historical productivity rate and production per plant for this age group
        age_productivity = 0
        age_production = 0
        
        if last_age in latest_age_data.index:
            age_productivity = latest_age_data.loc[last_age, 'productivity_rate']
            age_production = latest_age_data.loc[last_age, 'avg_production_per_plant']
        
        # Apply historical growth rates based on age and species
        age_growth = age_growth_rates.get(last_age, growth_rate)
        species_growth = species_growth_rates.get(species, growth_rate)
        
        # Apply historical productivity growth rates
        age_prod_growth = age_productivity_growth.get(last_age, 1.5)
        species_prod_growth = species_productivity_growth.get(species, 1.5)
        
        # Calculate expected production using weighted average of age and species factors
        # Give more weight to age-based factors as they have stronger correlation with production
        if age_production > 0:
            expected_production = (0.7 * (age_production * age_growth) + 
                                 0.3 * (species_production * species_growth))
        else:
            # If no age-specific data, rely more on species data
            expected_production = species_production * species_growth
        
        # Calculate expected productivity rate using weighted average
        if age_productivity > 0:
            expected_productivity = (0.7 * (age_productivity * age_prod_growth) + 
                                   0.3 * (species_productivity * species_prod_growth))
        else:
            # If no age-specific data, rely more on species data
            expected_productivity = species_productivity * species_prod_growth
        
        # For new trees that have no production history, use conservative estimates
        if age <= 2 and expected_production == 0:
            expected_production = 0.1  # Very small initial production
            expected_productivity = 0.2  # Very small initial productivity
        
        # For trees aging into productive years, ensure minimum reasonable production
        if age >= 5 and expected_production < 1.0:
            expected_production = max(expected_production, 1.0)
        
        # Cap growth to prevent unrealistic projections
        max_prod_by_age = {
            1: 0.2,
            2: 0.5,
            3: 1.5,
            4: 3.0,
            5: 5.0,
            6: 8.0,
            7: 12.0,
            8: 18.0,
        }
        
        # Apply ceiling to production values based on age
        age_ceiling = max_prod_by_age.get(age, 18.0)
        expected_production = min(expected_production, age_ceiling)
        
        # For early-producing species, boost values slightly
        early_producers = ['Chêne chevelu', "Pin d'Alep", 'Chêne vert']
        if species in early_producers and age >= 4:
            expected_production *= 1.2
            expected_productivity *= 1.2
        
        return expected_productivity, expected_production
    
    # Create production forecast
    print("Generating empirical forecast...")
    
    # Get the data for the most recent season for reference
    latest_age_production = age_production.loc["2024-2025"]
    latest_species_production = species_production.loc["2024-2025"]
    
    # Apply prediction function to each row
    forecast_results = []
    
    for _, row in forecast_base.iterrows():
        prod_rate, prod_per_plant = predict_empirical_productivity(
            row, latest_age_production, latest_species_production,
            age_growth_rates, species_growth_rates,
            age_productivity_growth, species_productivity_growth)
        
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
    print("\n--- Empirical Forecast Summary for 2025-2026 Season ---")
    
    print(f"\nTotal Expected Production: {forecast_df['Expected Total Production (g)'].sum():.2f}g")
    print(f"Average Expected Production per Plant: {forecast_df['Expected Total Production (g)'].sum() / forecast_df['Plants'].sum():.2f}g/plant")
    print(f"Overall growth from previous season: {forecast_df['Expected Total Production (g)'].sum() / total_production_by_season['2024-2025']:.2f}x")
    
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
    output_file = 'empirical_forecast_2025-2026.csv'
    forecast_df.to_csv(output_file, sep=';', index=False)
    print(f"\nDetailed forecast saved to {output_file}")
    
    # Generate some visualizations
    create_visualizations(combined_data, forecast_df, ramp_up)

# Function to create visualizations
def create_visualizations(historical_data, forecast_data, ramp_up):
    print("\n--- Generating Visualizations ---")
    
    # Set up the output directory for plots
    plots_dir = 'empirical_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Production by Season
    plt.figure(figsize=(10, 6))
    season_production = historical_data.groupby('Season')['Poids produit (g)'].sum()
    
    # Add forecast data
    seasons = list(season_production.index) + ['2025-2026']
    productions = list(season_production.values) + [forecast_data['Expected Total Production (g)'].sum()]
    
    plt.bar(seasons, productions, color=['blue', 'green', 'red'])
    plt.title('Total Truffle Production by Season (Empirical Forecast)')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'production_by_season.png'))
    
    # 2. Production by Age with ramp-up target for reference
    plt.figure(figsize=(12, 6))
    
    # Plot historical, forecast, and ramp-up target
    age_prod = historical_data.groupby(['Season', 'Age'])['Production au plant (g)'].mean().unstack(0)
    forecast_age_prod = forecast_data.groupby('Age')['Expected Production per Plant (g)'].mean()
    
    # Get all ages for the x-axis
    all_ages = sorted(set(ramp_up['Age']) | 
                     set(age_prod.index if isinstance(age_prod, pd.DataFrame) else []) | 
                     set(forecast_age_prod.index))
    
    # Plot the data
    markers = ['o', 's', '^', 'd', 'x']
    
    if isinstance(age_prod, pd.DataFrame):
        for i, season in enumerate(age_prod.columns):
            plt.plot(age_prod.index, age_prod[season].values, 
                     marker=markers[i % len(markers)], linestyle='-', 
                     label=f'Actual {season}')
    
    plt.plot(forecast_age_prod.index, forecast_age_prod.values, 
             marker='*', linestyle='-', linewidth=2,
             label='Forecast 2025-2026')
    
    # Plot the ramp-up target curve as a reference
    plt.plot(ramp_up['Age'], ramp_up['Production au plant (g)'], 
             marker='o', linestyle='--', alpha=0.5,
             label='Ramp-up Target (Reference Only)')
    
    plt.title('Average Production per Plant by Age')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Production per Plant (g)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'production_by_age.png'))
    
    # 3. Production trend over time with forecast
    plt.figure(figsize=(12, 6))
    
    # Create synthetic years for x-axis
    x_years = [2024, 2025, 2026]  # Representing the end year of each season
    y_production = [season_production['2023-2024'], 
                    season_production['2024-2025'],
                    forecast_data['Expected Total Production (g)'].sum()]
    
    # Plot total production trend
    plt.plot(x_years, y_production, 'o-', linewidth=2)
    
    # Add a polynomial trend line
    z = np.polyfit(x_years, y_production, 2)
    p = np.poly1d(z)
    
    # Generate points for trend line
    x_trend = np.linspace(min(x_years), max(x_years) + 1, 100)
    y_trend = p(x_trend)
    
    plt.plot(x_trend, y_trend, '--', alpha=0.7, color='gray')
    
    # Add data labels
    for i, (x, y) in enumerate(zip(x_years, y_production)):
        season = ['2023-2024', '2024-2025', '2025-2026'][i]
        plt.annotate(f"{season}\n{y:.0f}g", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
    plt.title('Truffle Production Trend and Forecast')
    plt.xlabel('Season End Year')
    plt.ylabel('Total Production (g)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'production_trend.png'))
    
    # 4. Production by Species
    plt.figure(figsize=(14, 7))
    
    # Historical species production
    species_prod = historical_data.groupby(['Season', 'Espèce'])['Poids produit (g)'].sum().unstack(fill_value=0)
    
    # Forecast species production
    forecast_species_prod = forecast_data.groupby('Espèce')['Expected Total Production (g)'].sum()
    
    # Choose top species by total production
    all_species = set(species_prod.columns if isinstance(species_prod, pd.DataFrame) else [])
    all_species.update(forecast_species_prod.index)
    
    # Filter to just top 5 species by total production for readability
    if len(all_species) > 5:
        top_species = forecast_species_prod.nlargest(5).index
    else:
        top_species = all_species
    
    # Prepare data for stacked bar chart
    seasons = ['2023-2024', '2024-2025', '2025-2026']
    species_data = {}
    
    for species in top_species:
        species_data[species] = [
            species_prod.loc['2023-2024', species] if isinstance(species_prod, pd.DataFrame) and species in species_prod.columns else 0,
            species_prod.loc['2024-2025', species] if isinstance(species_prod, pd.DataFrame) and species in species_prod.columns else 0,
            forecast_species_prod[species] if species in forecast_species_prod.index else 0
        ]
    
    # Plot stacked bar chart
    bottom = np.zeros(len(seasons))
    
    for species, data in species_data.items():
        plt.bar(seasons, data, bottom=bottom, label=species)
        bottom += np.array(data)
    
    plt.title('Truffle Production by Top Species (Empirical Forecast)')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'production_by_top_species.png'))
    
    # 5. Productivity Rate by Age
    plt.figure(figsize=(10, 6))
    
    # Calculate productivity rate by age
    prod_rate_by_age = historical_data.groupby(['Season', 'Age']).apply(
        lambda x: x['Plants Productifs'].sum() / x['Plants'].sum() * 100 if x['Plants'].sum() > 0 else 0
    ).unstack(fill_value=0)
    
    # Forecast productivity rate by age
    forecast_prod_rate = forecast_data.groupby('Age')['Expected Productivity Rate (%)'].mean()
    
    # Plot historical productivity rates
    if isinstance(prod_rate_by_age, pd.DataFrame):
        for i, season in enumerate(prod_rate_by_age.columns):
            plt.plot(prod_rate_by_age.index, prod_rate_by_age[season], 
                     marker=markers[i % len(markers)], linestyle='-', 
                     label=f'Actual {season}')
    
    # Plot forecast productivity rate
    plt.plot(forecast_prod_rate.index, forecast_prod_rate.values, 
             marker='*', linestyle='-', linewidth=2,
             label='Forecast 2025-2026')
    
    plt.title('Productivity Rate by Age (Empirical Forecast)')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Productivity Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'productivity_rate_by_age.png'))
    
    print(f"Visualizations saved to {plots_dir}/ directory")

# Run the analysis
if __name__ == "__main__":
    forecast_production()
