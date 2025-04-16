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
    print("\n===== ADJUSTED TRUFFLE PRODUCTION FORECAST FOR 2025-2026 SEASON =====\n")
    
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
    
    # Create age shift map to model the delay in production compared to ramp-up targets
    # This maps the actual production age to the corresponding ramp-up target age
    # For example, if trees at age 6 perform like the ramp-up target for age 3, we map 6->3
    age_shift_map = {
        1: 0,  # Age 1 trees perform below any target in ramp-up
        2: 0,  # Age 2 trees perform below any target in ramp-up
        3: 1,  # Age 3 trees perform closer to age 1 target  
        4: 2,  # Age 4 trees perform closer to age 2 target
        5: 3,  # Age 5 trees perform closer to age 3 target
        6: 3,  # Age 6 trees perform closer to age 3 target
        7: 4   # Age 7 trees perform closer to age 4 target
    }
    
    # Calculate the ratio between actual production and ramp-up targets
    # This will be used to adjust the expected production
    age_adjustment_ratio = {}
    ramp_up_dict = dict(zip(ramp_up['Age'], ramp_up['Production au plant (g)']))
    
    for season in ['2024-2025']:  # Use only the most recent season for calibration
        season_data = age_production.loc[season]
        for age, row in season_data.iterrows():
            target_age = age_shift_map.get(age, 0)
            target_production = ramp_up_dict.get(target_age, 0) if target_age > 0 else 0
            
            if target_production > 0:
                # Calculate ratio between actual and target
                ratio = row['avg_production_per_plant'] / target_production
            else:
                ratio = 0.3  # Default conservative ratio for young trees
                
            age_adjustment_ratio[age] = ratio
            print(f"Age {age} trees producing at {ratio:.2f}x the target for age {target_age}")
    
    # Calculate average growth factor between seasons for each age group
    print("\n--- Growth Factors Between Seasons ---")
    growth_factors = {}
    
    # First get the same age groups in both seasons
    ages_2023_2024 = set(age_production.loc['2023-2024'].index)
    ages_2024_2025 = set(age_production.loc['2024-2025'].index)
    common_ages = ages_2023_2024.intersection(ages_2024_2025)
    
    for age in common_ages:
        prod_2023_2024 = age_production.loc['2023-2024'].loc[age, 'avg_production_per_plant']
        prod_2024_2025 = age_production.loc['2024-2025'].loc[age, 'avg_production_per_plant']
        
        if prod_2023_2024 > 0:
            growth = prod_2024_2025 / prod_2023_2024
        else:
            growth = 1.0  # Default to no growth if previous production was zero
            
        growth_factors[age] = min(growth, 2.0)  # Cap growth at 2x to be conservative
        print(f"Age {age}: {growth_factors[age]:.2f}x growth")
    
    # Build prediction model for 2025-2026 season
    print("\n--- Building Forecast Model ---")
    
    # Prepare data for next season - start with the last season's plants
    forecast_base = season_2024_2025.copy()
    
    # Increment age by 1 for all plants for the next season
    forecast_base['Age'] = forecast_base['Age'] + 1
    forecast_base['Age Brut'] = forecast_base['Age Brut'] + 1
    
    # Function to predict productivity rate and production per plant based on historical data
    def predict_plant_productivity(row, age_data, species_data, ramp_up_data, age_shift_map, age_adjustment_ratio, growth_factors):
        age = row['Age']
        species = row['Espèce']
        
        # Get historical productivity rate for this age and species
        hist_prod_rate = 0
        hist_prod_per_plant = 0
        
        if (age-1) in age_data.index:  # Look at performance of this age group in the previous season
            last_age = age-1
            hist_prod_rate = age_data.loc[last_age, 'productivity_rate']
            hist_prod_per_plant = age_data.loc[last_age, 'avg_production_per_plant']
            
            # Apply a modest growth factor based on empirical data
            growth = growth_factors.get(last_age, 1.2)  # Use 1.2 as default growth if not available
            hist_prod_rate *= growth
            hist_prod_per_plant *= growth
        
        species_prod_rate = 0
        if species in species_data.index:
            species_prod_rate = species_data.loc[species, 'productivity_rate']
        
        # Get adjusted target production based on the age shift map
        target_age = age_shift_map.get(age, 0)
        ramp_up_production = 0
        if target_age > 0 and target_age in ramp_up_data['Age'].values:
            ramp_up_production = ramp_up_data[ramp_up_data['Age'] == target_age]['Production au plant (g)'].values[0]
        
        # Adjust ramp-up target based on observed performance
        adjustment = age_adjustment_ratio.get(age-1, 0.3)  # Use calibration from the previous age
        adjusted_target = ramp_up_production * adjustment
        
        # Calculate expected productivity rate using historical data and species characteristics
        if hist_prod_rate > 0:
            expected_prod_rate = 0.8 * hist_prod_rate + 0.2 * species_prod_rate
        else:
            expected_prod_rate = 0.5 * species_prod_rate  # Be conservative if no history
        
        # For production per plant, use a blend of historical data and adjusted target
        if hist_prod_per_plant > 0:
            expected_production = 0.7 * hist_prod_per_plant + 0.3 * adjusted_target
        else:
            expected_production = 0.5 * adjusted_target  # Be conservative if no history
        
        # Ensure we don't go below zero
        expected_prod_rate = max(0, expected_prod_rate)
        expected_production = max(0, expected_production)
        
        # Apply a more conservative growth factor for the forecast
        growth_factor = 1.1  # 10% growth is more conservative
        
        # Cap the production based on age to be more realistic
        max_prod_by_age = {
            1: 0.5,  # Very young trees have minimal production
            2: 1.0,
            3: 2.0,
            4: 4.0,
            5: 6.0,
            6: 10.0,
            7: 15.0
        }
        
        max_production = max_prod_by_age.get(age, 15.0)
        expected_production = min(expected_production, max_production)
        
        return expected_prod_rate, expected_production
    
    # Create production forecast
    print("Generating adjusted forecast...")
    
    # Get the data for the most recent season for reference
    latest_age_production = age_production.loc["2024-2025"]
    latest_species_production = species_production.loc["2024-2025"]
    
    # Apply prediction function to each row
    forecast_results = []
    
    for _, row in forecast_base.iterrows():
        prod_rate, prod_per_plant = predict_plant_productivity(
            row, latest_age_production, latest_species_production, ramp_up,
            age_shift_map, age_adjustment_ratio, growth_factors)
        
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
    print("\n--- Adjusted Forecast Summary for 2025-2026 Season ---")
    
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
    output_file = 'adjusted_forecast_2025-2026.csv'
    forecast_df.to_csv(output_file, sep=';', index=False)
    print(f"\nDetailed forecast saved to {output_file}")
    
    # Generate some visualizations
    create_visualizations(combined_data, forecast_df, ramp_up, age_shift_map)

# Function to create visualizations
def create_visualizations(historical_data, forecast_data, ramp_up, age_shift_map):
    print("\n--- Generating Visualizations ---")
    
    # Set up the output directory for plots
    plots_dir = 'adjusted_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Production by Season
    plt.figure(figsize=(10, 6))
    season_production = historical_data.groupby('Season')['Poids produit (g)'].sum()
    
    # Add forecast data
    seasons = list(season_production.index) + ['2025-2026']
    productions = list(season_production.values) + [forecast_data['Expected Total Production (g)'].sum()]
    
    plt.bar(seasons, productions, color=['blue', 'green', 'red'])
    plt.title('Total Truffle Production by Season (Adjusted Forecast)')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'production_by_season.png'))
    
    # 2. Production by Age with ramp-up target and adjusted expectations
    plt.figure(figsize=(12, 6))
    
    # Historical age production
    age_prod = historical_data.groupby('Age')['Production au plant (g)'].mean()
    
    # Forecast age production
    forecast_age_prod = forecast_data.groupby('Age')['Expected Production per Plant (g)'].mean()
    
    # Plot historical, forecast, and ramp-up target
    plt.plot(age_prod.index, age_prod.values, 'o-', label='Historical Average')
    plt.plot(forecast_age_prod.index, forecast_age_prod.values, 's-', label='Adjusted Forecast 2025-2026')
    
    # Plot the actual ramp-up curve
    plt.plot(ramp_up['Age'], ramp_up['Production au plant (g)'], '*--', label='Target Ramp-up')
    
    # Create the shifted ramp-up curve to show the delay
    max_age = max(forecast_age_prod.index)
    shifted_ages = []
    shifted_values = []
    
    for age in range(1, max_age + 1):
        target_age = age_shift_map.get(age, 0)
        if target_age > 0:
            shifted_ages.append(age)
            target_val = ramp_up[ramp_up['Age'] == target_age]['Production au plant (g)'].values[0] if target_age in ramp_up['Age'].values else 0
            shifted_values.append(target_val)
    
    plt.plot(shifted_ages, shifted_values, 'x-.', label='Shifted Ramp-up (Adjusted for Time Delay)')
    
    plt.title('Average Production per Plant by Age (with Time-Adjusted Target)')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Production per Plant (g)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'production_by_age_with_shift.png'))
    
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
    
    plt.title('Truffle Production by Species and Season (Adjusted Forecast)')
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
    plt.plot(forecast_prod_rate.index, forecast_prod_rate.values, 's-', label='Adjusted Forecast 2025-2026')
    
    plt.title('Productivity Rate by Age (Adjusted Forecast)')
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Productivity Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'productivity_rate_by_age.png'))
    
    # 5. Comparison of Historical vs Target vs Forecast 
    plt.figure(figsize=(12, 6))
    
    # Historical data for recent season
    recent_season = historical_data[historical_data['Season'] == '2024-2025']
    recent_age_prod = recent_season.groupby('Age')['Production au plant (g)'].mean()
    
    # Bar chart comparing target, actual, and forecast
    ages = sorted(set(ramp_up['Age']) | set(recent_age_prod.index) | set(forecast_age_prod.index))
    
    # Process data for bar chart
    bar_data = []
    
    for age in ages:
        target_val = ramp_up[ramp_up['Age'] == age]['Production au plant (g)'].values[0] if age in ramp_up['Age'].values else 0
        actual_val = recent_age_prod[age] if age in recent_age_prod.index else 0
        forecast_val = forecast_age_prod[age] if age in forecast_age_prod.index else 0
        
        bar_data.append({
            'Age': age,
            'Target': target_val,
            'Actual 2024-2025': actual_val,
            'Forecast 2025-2026': forecast_val
        })
    
    # Convert to DataFrame for easy plotting
    bar_df = pd.DataFrame(bar_data)
    bar_df = bar_df.sort_values('Age')
    
    # Plot bar chart
    bar_width = 0.25
    x = np.arange(len(bar_df))
    
    plt.bar(x - bar_width, bar_df['Target'], bar_width, label='Ramp-up Target', alpha=0.7)
    plt.bar(x, bar_df['Actual 2024-2025'], bar_width, label='Actual 2024-2025', alpha=0.7)
    plt.bar(x + bar_width, bar_df['Forecast 2025-2026'], bar_width, label='Forecast 2025-2026', alpha=0.7)
    
    plt.xlabel('Age of Plants (years)')
    plt.ylabel('Production per Plant (g)')
    plt.title('Comparison of Target, Actual, and Forecast Production by Age')
    plt.xticks(x, bar_df['Age'])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'target_actual_forecast_comparison.png'))
    
    print(f"Visualizations saved to {plots_dir}/ directory")

# Run the analysis
if __name__ == "__main__":
    forecast_production()
