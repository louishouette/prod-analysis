#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Function to load and preprocess season data
def load_season_data(file_path, season_name):
    # Load data with semicolon separator
    df = pd.read_csv(file_path, sep=';')
    
    # Fix potential decimal comma formatting (French format)
    numeric_cols = ['Taux de Productivité (%)', 'Poids produit (g)', 'Poids moyen (g)', 'Production au plant (g)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Add season column
    df['Season'] = season_name
    
    return df

# Function to load ramp-up targets
def load_rampup_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    # Fix potential decimal comma formatting
    df['Production au plant (g)'] = df['Production au plant (g)'].astype(str).str.replace(',', '.').astype(float)
    return df

# Load data
s2023_2024 = load_season_data('season_2023-2024.csv', '2023-2024')
s2024_2025 = load_season_data('season_2024-2025.csv', '2024-2025')
rampup = load_rampup_data('ramp-up.csv')

# Combine season data
combined_data = pd.concat([s2023_2024, s2024_2025], ignore_index=True)

# Define KPIs to analyze
kpis = [
    'Plants Productifs',
    'Taux de Productivité (%)',
    'Poids produit (g)',
    'Nombre de truffe',
    'Poids moyen (g)',
    'Production au plant (g)'
]

# Create a directory for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Function to create plots for each KPI by parcel, season and age
def plot_kpi_by_parcel_season_age(data, kpi):
    # Group by Parcelle, Season and Age, calculating mean of the KPI
    grouped = data.groupby(['Parcelle', 'Season', 'Age'])[kpi].mean().reset_index()
    
    # Get unique parcels and sort them
    parcels = sorted(data['Parcelle'].unique())
    
    # Create a figure with subplots for each parcel
    n_parcels = len(parcels)
    fig, axes = plt.subplots(n_parcels, 1, figsize=(12, n_parcels * 4), sharex=True)
    
    # If only one parcel, axes is not an array
    if n_parcels == 1:
        axes = [axes]
    
    for i, parcel in enumerate(parcels):
        ax = axes[i]
        parcel_data = grouped[grouped['Parcelle'] == parcel]
        
        # Plot each season
        for season in ['2023-2024', '2024-2025']:
            season_data = parcel_data[parcel_data['Season'] == season]
            if not season_data.empty:
                ax.plot(season_data['Age'], season_data[kpi], marker='o', 
                        linestyle='-', label=f'Season {season}')
        
        # Add ramp-up target for 'Production au plant (g)' KPI
        if kpi == 'Production au plant (g)' and not rampup.empty:
            ax.plot(rampup['Age'], rampup[kpi], marker='x', linestyle='--', 
                   color='red', label='Target Ramp-up')
        
        ax.set_title(f'Parcel {parcel} - {kpi}')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(kpi)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use integer ticks for Age
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add data points annotations
        for idx, row in season_data.iterrows():
            ax.annotate(f"{row[kpi]:.2f}", 
                     (row['Age'], row[kpi]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')
    
    plt.tight_layout()
    plt.savefig(f'plots/{kpi.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")}.png')
    plt.close()

# Alternative visualization: Create heatmaps for KPIs by age and parcel for each season
def plot_kpi_heatmaps(data, kpi):
    for season in ['2023-2024', '2024-2025']:
        season_data = data[data['Season'] == season]
        
        # Pivot data to create a matrix suitable for heatmap
        pivot_data = season_data.pivot_table(
            index='Parcelle', 
            columns='Age',
            values=kpi,
            aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title(f'{kpi} by Parcel and Age - Season {season}')
        plt.ylabel('Parcel')
        plt.xlabel('Age (years)')
        plt.tight_layout()
        plt.savefig(f'plots/heatmap_{season}_{kpi.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")}.png')
        plt.close()

# Create plots for each KPI
for kpi in kpis:
    print(f"Plotting {kpi}...")
    plot_kpi_by_parcel_season_age(combined_data, kpi)
    plot_kpi_heatmaps(combined_data, kpi)

# Create a summary plot for Production au plant by Age, comparing all parcels against target
def plot_production_summary(data):
    kpi = 'Production au plant (g)'
    
    # Calculate average production per plant by age across all parcels and seasons
    avg_by_age = data.groupby(['Age'])[kpi].mean().reset_index()
    avg_by_age_season = data.groupby(['Age', 'Season'])[kpi].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Plot average by age
    plt.plot(avg_by_age['Age'], avg_by_age[kpi], 
             marker='o', linestyle='-', linewidth=2, 
             color='blue', label='Average all parcels')
    
    # Plot average by age and season
    for season in ['2023-2024', '2024-2025']:
        season_data = avg_by_age_season[avg_by_age_season['Season'] == season]
        plt.plot(season_data['Age'], season_data[kpi], 
                 marker='o', linestyle='--', 
                 label=f'Season {season}')
    
    # Plot ramp-up target
    plt.plot(rampup['Age'], rampup[kpi], 
             marker='x', linestyle='-.', linewidth=2, 
             color='red', label='Target Ramp-up')
    
    plt.title('Average Production per Plant by Age vs Target')
    plt.xlabel('Age (years)')
    plt.ylabel(kpi)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add annotations
    for _, row in avg_by_age.iterrows():
        plt.annotate(f"{row[kpi]:.2f}", 
                 (row['Age'], row[kpi]),
                 textcoords="offset points",
                 xytext=(0, 5),
                 ha='center')
    
    # Add annotations for ramp-up target
    for _, row in rampup.iterrows():
        plt.annotate(f"{row[kpi]:.1f}", 
                 (row['Age'], row[kpi]),
                 textcoords="offset points",
                 xytext=(0, -15),
                 ha='center',
                 color='red')
    
    plt.tight_layout()
    plt.savefig('plots/production_summary.png')
    plt.close()

# Create a plot showing productivity rate by species, age and season
def plot_species_comparison(data):
    kpi = 'Taux de Productivité (%)'
    
    # Group by Species, Season and Age
    grouped = data.groupby(['Espèce', 'Season', 'Age'])[kpi].mean().reset_index()
    
    # Get unique species
    species = sorted(data['Espèce'].unique())
    
    plt.figure(figsize=(14, 8))
    
    for i, specie in enumerate(species):
        specie_data = grouped[grouped['Espèce'] == specie]
        
        for j, season in enumerate(['2023-2024', '2024-2025']):
            season_data = specie_data[specie_data['Season'] == season]
            if not season_data.empty:
                plt.plot(season_data['Age'], season_data[kpi], 
                        marker='o', linestyle=['-', '--'][j], 
                        color=plt.cm.tab10(i), 
                        label=f'{specie} - {season}')
    
    plt.title('Productivity Rate by Species, Age and Season')
    plt.xlabel('Age (years)')
    plt.ylabel(kpi)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('plots/species_productivity_comparison.png')
    plt.close()

# Execute the summary plots
plot_production_summary(combined_data)
plot_species_comparison(combined_data)

print("All plots have been saved to the 'plots' directory.")
