#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler

# Set plot style for more appealing visuals
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette for seasons
season_colors = {
    '2023-2024': '#1f77b4',  # Blue
    '2024-2025': '#ff7f0e',  # Orange
    'Target': '#d62728'       # Red
}

# Set font properties
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

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
    fig, axes = plt.subplots(n_parcels, 1, figsize=(12, n_parcels * 4), sharex=True, dpi=100)
    fig.suptitle(f'{kpi} by Parcel and Tree Age', fontsize=18, y=0.92)
    
    # If only one parcel, axes is not an array
    if n_parcels == 1:
        axes = [axes]
    
    for i, parcel in enumerate(parcels):
        ax = axes[i]
        parcel_data = grouped[grouped['Parcelle'] == parcel]
        
        # Plot each season with custom colors
        for season in ['2023-2024', '2024-2025']:
            season_data = parcel_data[parcel_data['Season'] == season]
            if not season_data.empty:
                ax.plot(season_data['Age'], season_data[kpi], marker='o', 
                        linestyle='-', linewidth=2.5, 
                        color=season_colors[season], 
                        label=f'Season {season}')
        
        # Add ramp-up target for 'Production au plant (g)' KPI
        if kpi == 'Production au plant (g)' and not rampup.empty:
            ax.plot(rampup['Age'], rampup[kpi], marker='x', linestyle='--', 
                   linewidth=2, color=season_colors['Target'], 
                   label='Target Ramp-up')
        
        ax.set_title(f'Parcel {parcel}', fontweight='bold')
        ax.set_xlabel('Age (years)', fontweight='bold')
        ax.set_ylabel(kpi, fontweight='bold')
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Use integer ticks for Age
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        # Add data points annotations with improved styling
        for idx, row in season_data.iterrows():
            ax.annotate(f"{row[kpi]:.2f}", 
                     (row['Age'], row[kpi]),
                     textcoords="offset points",
                     xytext=(0, 7),
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'plots/{kpi.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")}.png')
    plt.close()

# Function removed as per requirements

# Create plots for each KPI
for kpi in kpis:
    print(f"Plotting {kpi}...")
    plot_kpi_by_parcel_season_age(combined_data, kpi)

# Create a summary plot for Production au plant by Age, comparing all parcels against target
def plot_production_summary(data):
    kpi = 'Production au plant (g)'
    
    # Calculate average production per plant by age across all parcels and seasons
    avg_by_age = data.groupby(['Age'])[kpi].mean().reset_index()
    avg_by_age_season = data.groupby(['Age', 'Season'])[kpi].mean().reset_index()
    
    plt.figure(figsize=(14, 10), dpi=100)
    
    # Create a light color background
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Plot average by age
    plt.plot(avg_by_age['Age'], avg_by_age[kpi], 
             marker='o', markersize=10, linestyle='-', linewidth=3, 
             color='#2c3e50', label='Average all parcels')
    
    # Plot average by age and season with custom colors
    for season in ['2023-2024', '2024-2025']:
        season_data = avg_by_age_season[avg_by_age_season['Season'] == season]
        plt.plot(season_data['Age'], season_data[kpi], 
                 marker='o', markersize=8, linestyle='--', linewidth=2.5,
                 color=season_colors[season],
                 label=f'Season {season}')
    
    # Plot ramp-up target
    plt.plot(rampup['Age'], rampup[kpi], 
             marker='x', markersize=10, linestyle='-.', linewidth=2.5, 
             color=season_colors['Target'], label='Target Ramp-up')
    
    plt.title('Average Production per Plant by Age vs Target', fontweight='bold', pad=20)
    plt.xlabel('Age (years)', fontweight='bold', labelpad=10)
    plt.ylabel(kpi, fontweight='bold', labelpad=10)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a border around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    
    # Add annotations with improved styling
    for _, row in avg_by_age.iterrows():
        plt.annotate(f"{row[kpi]:.2f}", 
                 (row['Age'], row[kpi]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 fontsize=11,
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add annotations for ramp-up target
    for _, row in rampup.iterrows():
        plt.annotate(f"{row[kpi]:.1f}", 
                 (row['Age'], row[kpi]),
                 textcoords="offset points",
                 xytext=(0, -20),
                 ha='center',
                 fontsize=11,
                 fontweight='bold',
                 color=season_colors['Target'],
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))
    
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
    
    # Get all unique species and create a color map dynamically
    species_colors = {}
    color_sets = [
        ['#3498db', '#9b59b6'],  # Blue, Purple for 2023-2024 and 2024-2025
        ['#2ecc71', '#f1c40f'],  # Green, Yellow for 2023-2024 and 2024-2025
        ['#e74c3c', '#8e44ad'],  # Red, Purple for 2023-2024 and 2024-2025
        ['#f39c12', '#16a085'],  # Orange, Teal for 2023-2024 and 2024-2025
        ['#2980b9', '#c0392b']   # Dark Blue, Dark Red for 2023-2024 and 2024-2025
    ]
    
    for i, specie in enumerate(species):
        # Cycle through color sets if there are more species than color sets
        species_colors[specie] = color_sets[i % len(color_sets)]
    
    plt.figure(figsize=(14, 10), dpi=100)
    
    # Create a light color background
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    for i, specie in enumerate(species):
        specie_data = grouped[grouped['Espèce'] == specie]
        
        for j, season in enumerate(['2023-2024', '2024-2025']):
            season_data = specie_data[specie_data['Season'] == season]
            if not season_data.empty:
                line = plt.plot(season_data['Age'], season_data[kpi], 
                        marker='o', markersize=8, linestyle=['-', '--'][j], linewidth=2.5,
                        color=species_colors[specie][j], 
                        label=f'{specie} - {season}')
                
                # Add annotations
                for _, row in season_data.iterrows():
                    plt.annotate(f"{row[kpi]:.2f}", 
                             (row['Age'], row[kpi]),
                             textcoords="offset points",
                             xytext=(0, 8),
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.title('Productivity Rate by Species, Age and Season', fontweight='bold', pad=20)
    plt.xlabel('Age (years)', fontweight='bold', labelpad=10)
    plt.ylabel(kpi, fontweight='bold', labelpad=10)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a border around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    plt.tight_layout()
    plt.savefig('plots/species_productivity_comparison.png')
    plt.close()

# Execute the summary plots
plot_production_summary(combined_data)
plot_species_comparison(combined_data)

print("All plots have been saved to the 'plots' directory.")
