#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
SEASON_2023_2024_PATH = 'season_2023-2024.csv'
SEASON_2024_2025_PATH = 'season_2024-2025.csv'
EMPIRICAL_FORECAST_PATH = 'empirical_forecast_2025-2026.csv'

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
        numeric_cols = ['Taux de Productivitu00e9 (%)', 'Poids produit (g)', 
                        'Poids moyen (g)', 'Production au plant (g)']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_french_decimal)
                
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Main function to create improved graphs
def create_improved_graphs():
    print("\n===== Creating Improved Productivity Rate by Age Graph =====\n")
    
    # Load historical data
    season_2023_2024 = load_data(SEASON_2023_2024_PATH)
    season_2024_2025 = load_data(SEASON_2024_2025_PATH)
    
    # Load forecast data
    try:
        forecast_data = pd.read_csv(EMPIRICAL_FORECAST_PATH, sep=';')
    except Exception as e:
        print(f"Error loading forecast data: {e}")
        return
        
    # Add season labels
    season_2023_2024['Season'] = '2023-2024'
    season_2024_2025['Season'] = '2024-2025'
    
    # Combine datasets for analysis
    historical_data = pd.concat([season_2023_2024, season_2024_2025])
    
    # Create output directory
    plots_dir = 'improved_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create a clearer productivity rate by age chart
    plt.figure(figsize=(12, 8))
    
    # Calculate productivity rate by age (% of plants producing for each age)
    productivity_by_age = {}
    
    # For 2023-2024 season
    productivity_2023_2024 = {}
    for age, group in season_2023_2024.groupby('Age'):
        total_plants = group['Plants'].sum()
        productive_plants = group['Plants Productifs'].sum()
        productivity_rate = (productive_plants / total_plants * 100) if total_plants > 0 else 0
        productivity_2023_2024[age] = productivity_rate
    
    # For 2024-2025 season
    productivity_2024_2025 = {}
    for age, group in season_2024_2025.groupby('Age'):
        total_plants = group['Plants'].sum()
        productive_plants = group['Plants Productifs'].sum()
        productivity_rate = (productive_plants / total_plants * 100) if total_plants > 0 else 0
        productivity_2024_2025[age] = productivity_rate
    
    # For forecast 2025-2026
    productivity_forecast = {}
    for age, group in forecast_data.groupby('Age'):
        productivity_rate = group['Expected Productivity Rate (%)'].mean()
        productivity_forecast[age] = productivity_rate
    
    # Get all ages across all seasons
    all_ages = sorted(set(list(productivity_2023_2024.keys()) + 
                       list(productivity_2024_2025.keys()) + 
                       list(productivity_forecast.keys())))
    
    # Prepare data for bar chart
    x = np.arange(len(all_ages))  # the label locations
    width = 0.25  # the width of the bars
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars for each season
    season1_values = [productivity_2023_2024.get(age, 0) for age in all_ages]
    season2_values = [productivity_2024_2025.get(age, 0) for age in all_ages]
    forecast_values = [productivity_forecast.get(age, 0) for age in all_ages]
    
    # Plot the bars
    rects1 = ax.bar(x - width, season1_values, width, label='2023-2024', color='skyblue')
    rects2 = ax.bar(x, season2_values, width, label='2024-2025', color='orange')
    rects3 = ax.bar(x + width, forecast_values, width, label='2025-2026 (Forecast)', color='green')
    
    # Add labels, title and legend
    ax.set_xlabel('Tree Age (years)', fontsize=14)
    ax.set_ylabel('Productivity Rate (%)', fontsize=14)
    ax.set_title('Productivity Rate by Tree Age Across Seasons', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_ages)
    
    # Add value labels on top of each bar
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.5:  # Only label bars with visible height
                ax.annotate(f'{height:.1f}%',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'improved_productivity_rate_by_age.png'), dpi=300)
    print(f"Improved productivity rate chart saved to {plots_dir}/improved_productivity_rate_by_age.png")
    
    # Create line graph version for comparison
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each season
    plt.plot(all_ages, season1_values, 'o-', linewidth=2, label='2023-2024', color='blue')
    plt.plot(all_ages, season2_values, 's-', linewidth=2, label='2024-2025', color='orange')
    plt.plot(all_ages, forecast_values, '^-', linewidth=2, label='2025-2026 (Forecast)', color='green')
    
    # Add labels for data points
    for i, age in enumerate(all_ages):
        # 2023-2024 values
        if season1_values[i] > 0.5:
            plt.annotate(f'{season1_values[i]:.1f}%', 
                        (age, season1_values[i]), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        color='blue',
                        fontsize=9)
        
        # 2024-2025 values
        if season2_values[i] > 0.5:
            plt.annotate(f'{season2_values[i]:.1f}%', 
                        (age, season2_values[i]), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        color='orange',
                        fontsize=9)
        
        # Forecast values
        if forecast_values[i] > 0.5:
            plt.annotate(f'{forecast_values[i]:.1f}%', 
                        (age, forecast_values[i]), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        color='green',
                        fontsize=9)
    
    # Add labels, title and customize appearance
    plt.title('Productivity Rate by Tree Age (Line Chart)', fontsize=16)
    plt.xlabel('Tree Age (years)', fontsize=14)
    plt.ylabel('Productivity Rate (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(all_ages)
    
    # Show 0 to max+10% on y-axis for better visualization
    max_value = max(max(season1_values), max(season2_values), max(forecast_values))
    plt.ylim(0, max_value * 1.1)
    
    # Save the line chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'productivity_rate_line_chart.png'), dpi=300)
    print(f"Line chart version saved to {plots_dir}/productivity_rate_line_chart.png")
    
    # Also create a combined production and productivity visualization
    plt.figure(figsize=(16, 10))
    
    # Create figure with 2 y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Calculate production per plant by age
    production_2023_2024 = {}
    for age, group in season_2023_2024.groupby('Age'):
        total_production = group['Poids produit (g)'].sum()
        total_plants = group['Plants'].sum()
        production_per_plant = total_production / total_plants if total_plants > 0 else 0
        production_2023_2024[age] = production_per_plant
    
    production_2024_2025 = {}
    for age, group in season_2024_2025.groupby('Age'):
        total_production = group['Poids produit (g)'].sum()
        total_plants = group['Plants'].sum()
        production_per_plant = total_production / total_plants if total_plants > 0 else 0
        production_2024_2025[age] = production_per_plant
    
    production_forecast = {}
    for age, group in forecast_data.groupby('Age'):
        production_per_plant = group['Expected Production per Plant (g)'].mean()
        production_forecast[age] = production_per_plant
    
    # Prepare production data
    prod_season1 = [production_2023_2024.get(age, 0) for age in all_ages]
    prod_season2 = [production_2024_2025.get(age, 0) for age in all_ages]
    prod_forecast = [production_forecast.get(age, 0) for age in all_ages]
    
    # Plot production lines
    line1 = ax1.plot(all_ages, prod_season1, 'o-', linewidth=2, label='Production 2023-2024', color='blue')
    line2 = ax1.plot(all_ages, prod_season2, 's-', linewidth=2, label='Production 2024-2025', color='orange')
    line3 = ax1.plot(all_ages, prod_forecast, '^-', linewidth=2, label='Production 2025-2026 (Forecast)', color='green')
    
    # Plot productivity as bars
    bar1 = ax2.bar(np.array(all_ages) - 0.2, season1_values, 0.15, alpha=0.3, label='Productivity % 2023-2024', color='blue')
    bar2 = ax2.bar(np.array(all_ages), season2_values, 0.15, alpha=0.3, label='Productivity % 2024-2025', color='orange')
    bar3 = ax2.bar(np.array(all_ages) + 0.2, forecast_values, 0.15, alpha=0.3, label='Productivity % 2025-2026', color='green')
    
    # Add labels and customize appearance
    ax1.set_xlabel('Tree Age (years)', fontsize=14)
    ax1.set_ylabel('Production per Plant (g)', fontsize=14, color='black')
    ax2.set_ylabel('Productivity Rate (%)', fontsize=14, color='gray')
    plt.title('Production per Plant & Productivity Rate by Tree Age', fontsize=16)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    # Add grid and set limits
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(all_ages)
    
    # Save the combined chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_production_productivity.png'), dpi=300)
    print(f"Combined production and productivity chart saved to {plots_dir}/combined_production_productivity.png")
    
    print("\nAll improved visualizations created successfully.")

# Run the visualization script
if __name__ == "__main__":
    create_improved_graphs()
