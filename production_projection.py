#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set plot style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Custom color palette for seasons
season_colors = {
    '2023-2024': '#1f77b4',  # Blue
    '2024-2025': '#ff7f0e',  # Orange
    '2025-2026': '#2ca02c'    # Green (projection)
}

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

# Calculate total production by parcel and season
production_by_parcel = combined_data.groupby(['Parcelle', 'Season'])['Poids produit (g)'].sum().reset_index()
production_by_parcel = production_by_parcel.rename(columns={'Poids produit (g)': 'Total Production (g)'})

# Create wide format data to make projection calculations easier
pivot_production = production_by_parcel.pivot(index='Parcelle', columns='Season', values='Total Production (g)').reset_index()
pivot_production = pivot_production.rename_axis(None, axis=1).fillna(0)

# Function to project next season based on previous two seasons
def project_next_season(current, previous):
    if previous == 0 and current == 0:
        return 0
    elif previous == 0:
        # If started producing only in the current season, assume 50% growth
        return current * 1.5
    else:
        # Calculate growth rate between previous and current seasons
        growth_rate = current / previous
        # Apply growth rate to current season to get projection
        # Add some moderation to avoid extreme projections
        if growth_rate > 3:
            growth_rate = 3  # Cap extreme growth rates
        return current * growth_rate

# Add projection for 2025-2026 season
pivot_production['2025-2026'] = pivot_production.apply(
    lambda row: project_next_season(row['2024-2025'], row['2023-2024']), 
    axis=1
)

# Convert back to long format for plotting
projected_data = pd.melt(
    pivot_production,
    id_vars=['Parcelle'],
    value_vars=['2023-2024', '2024-2025', '2025-2026'],
    var_name='Season',
    value_name='Total Production (g)'
)

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('projections'):
    os.makedirs('projections')

# Plot total production by parcel and season with projections
def plot_total_production_with_projection():
    # Get unique parcels and sort them
    parcels = sorted(projected_data['Parcelle'].unique())
    
    # Create a figure with subplots for each parcel
    n_parcels = len(parcels)
    fig, axes = plt.subplots(n_parcels, 1, figsize=(12, n_parcels * 4), sharex=True, dpi=100)
    fig.suptitle('Truffle Production by Parcel with 2025-2026 Projection', fontsize=18, y=0.92)
    
    # If only one parcel, axes is not an array
    if n_parcels == 1:
        axes = [axes]
    
    season_list = ['2023-2024', '2024-2025', '2025-2026']
    
    # Format function for thousands separator
    def format_with_comma(x, pos):
        return f'{x:,.0f}'.replace(',', ' ')
    
    for i, parcel in enumerate(parcels):
        ax = axes[i]
        parcel_data = projected_data[projected_data['Parcelle'] == parcel]
        
        # Plot bars for each season
        bars = ax.bar(
            season_list,
            parcel_data['Total Production (g)'],
            color=[season_colors[s] for s in season_list],
            width=0.6
        )
        
        # No hatching for projection bar, just different color
        
        # Add data labels on top of each bar with thousand separator
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 10,
                    f'{height:,.0f}g'.replace(',', ' '),
                    ha='center', va='bottom',
                    fontweight='bold'
                )
        
        # Add a subtle line connecting the points to show the trend
        ax.plot(
            season_list,
            parcel_data['Total Production (g)'],
            'o-', color='#444444', alpha=0.6, linewidth=1.5
        )
        
        ax.set_title(f'Parcel {parcel}', fontweight='bold')
        ax.set_ylabel('Total Production (g)', fontweight='bold')
        
        # Customize the plot appearance
        ax.set_facecolor('#f8f9fa')  # Light background
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis with thousand separator
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(format_with_comma))
    
    # Add a legend to explain the seasons
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=season_colors['2023-2024'], label='Season 2023-2024'),
        Patch(facecolor=season_colors['2024-2025'], label='Season 2024-2025'),
        Patch(facecolor=season_colors['2025-2026'], label='Projected Season 2025-2026')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.91))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('projections/total_production_by_parcel.png')
    plt.close()

# Plot overall production projection
def plot_total_production_summary():
    # Calculate totals by season
    season_totals = projected_data.groupby('Season')['Total Production (g)'].sum().reset_index()
    
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Format function for thousands separator
    def format_with_comma(x, pos):
        return f'{x:,.0f}'.replace(',', ' ')
    
    # Plot with bars and line
    bars = plt.bar(
        season_totals['Season'],
        season_totals['Total Production (g)'],
        color=[season_colors[s] for s in season_totals['Season']],
        width=0.6
    )
    
    # No hatching for projection bar, just different color
    
    # Add data labels on top of each bar with thousand separator
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 100,
            f'{height:,.0f}g'.replace(',', ' '),
            ha='center', va='bottom',
            fontweight='bold', fontsize=14
        )
    
    # Add a trend line
    plt.plot(
        season_totals['Season'],
        season_totals['Total Production (g)'],
        'o-', color='#444444', alpha=0.6, linewidth=2
    )
    
    # Add percentage growth labels
    for i in range(1, len(season_totals)):
        current = season_totals['Total Production (g)'].iloc[i]
        previous = season_totals['Total Production (g)'].iloc[i-1]
        if previous > 0:
            growth_pct = ((current - previous) / previous) * 100
            plt.text(
                i,  # x position
                (current + previous) / 2,  # y position
                f'+{growth_pct:.1f}%',
                ha='center', va='center',
                fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    # Add a title and labels
    plt.title('Total Truffle Production with 2025-2026 Projection', fontweight='bold', pad=20, fontsize=16)
    plt.ylabel('Total Production (g)', fontweight='bold', labelpad=10, fontsize=14)
    plt.xlabel('Season', fontweight='bold', labelpad=10, fontsize=14)
    
    # Add a legend with clean look
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=season_colors['2023-2024'], label='Actual 2023-2024'),
        Patch(facecolor=season_colors['2024-2025'], label='Actual 2024-2025'),
        Patch(facecolor=season_colors['2025-2026'], label='Projected 2025-2026')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Customize the plot appearance
    plt.grid(axis='y', alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')  # Light background
    
    # Format y-axis with thousand separator
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_comma))
    
    plt.tight_layout()
    plt.savefig('projections/total_production_summary.png')
    plt.close()

# Plot top 5 parcels contributing to production
def plot_top_5_parcels():
    # Get the top 5 parcels by projected production in 2025-2026
    top_parcels = pivot_production.sort_values('2025-2026', ascending=False).head(5)['Parcelle'].values
    
    # Filter data for top 5 parcels
    top_parcels_data = projected_data[projected_data['Parcelle'].isin(top_parcels)]
    
    # Create the figure first
    plt.figure(figsize=(14, 10), dpi=100)
    
    # Use a regular barplot instead of catplot for better control
    ax = sns.barplot(
        data=top_parcels_data,
        x='Parcelle', y='Total Production (g)',
        hue='Season',
        palette=season_colors
    )
    
    # Format y-axis with thousand separator
    def format_with_comma(x, pos):
        return f'{x:,.0f}'.replace(',', ' ')
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_with_comma))
    
    # Add title and labels
    plt.title('Top 5 Producing Parcels with 2025-2026 Projection', fontweight='bold', pad=20, fontsize=16)
    plt.ylabel('Total Production (g)', fontweight='bold', labelpad=10, fontsize=14)
    plt.xlabel('Parcel', fontweight='bold', labelpad=10, fontsize=14)
    
    # Move legend to avoid overlap (place at top-left corner)
    plt.legend(title='Season', bbox_to_anchor=(0, 1.02), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('projections/top_5_parcels.png')
    plt.close()

# Create a summary of the projections (Excel or CSV)
def create_excel_summary():
    # Create a DataFrame for output with totals
    excel_data = pivot_production.copy()
    
    # Add growth rates
    excel_data['Growth Rate 23-24 to 24-25'] = excel_data.apply(
        lambda row: ((row['2024-2025'] / row['2023-2024']) - 1) * 100 if row['2023-2024'] > 0 else 'N/A',
        axis=1
    )
    excel_data['Growth Rate 24-25 to 25-26 (Projected)'] = excel_data.apply(
        lambda row: ((row['2025-2026'] / row['2024-2025']) - 1) * 100 if row['2024-2025'] > 0 else 'N/A',
        axis=1
    )
    
    # Add totals row
    totals = pd.DataFrame([{
        'Parcelle': 'TOTAL',
        '2023-2024': excel_data['2023-2024'].sum(),
        '2024-2025': excel_data['2024-2025'].sum(),
        '2025-2026': excel_data['2025-2026'].sum(),
        'Growth Rate 23-24 to 24-25': ((excel_data['2024-2025'].sum() / excel_data['2023-2024'].sum()) - 1) * 100 if excel_data['2023-2024'].sum() > 0 else 'N/A',
        'Growth Rate 24-25 to 25-26 (Projected)': ((excel_data['2025-2026'].sum() / excel_data['2024-2025'].sum()) - 1) * 100 if excel_data['2024-2025'].sum() > 0 else 'N/A'
    }])
    
    excel_data = pd.concat([excel_data, totals], ignore_index=True)
    
    # Try to save as Excel first, fall back to CSV if openpyxl is not available
    try:
        excel_data.to_excel('projections/production_projections.xlsx', index=False)
        print("Excel summary saved to projections/production_projections.xlsx")
    except ImportError:
        # Fall back to CSV if openpyxl is not available
        excel_data.to_csv('projections/production_projections.csv', index=False, sep=';')
        print("CSV summary saved to projections/production_projections.csv (Excel export requires openpyxl package)")
    
    # Print the table to console for immediate viewing
    print("\nProjected Production by Parcel (in grams):")
    print(excel_data[['Parcelle', '2023-2024', '2024-2025', '2025-2026']].to_string(index=False))
    
    # Print overall projection
    total_projection = excel_data.loc[excel_data['Parcelle'] == 'TOTAL', '2025-2026'].iloc[0]
    print(f"\nTotal projected production for 2025-2026 season: {total_projection:.0f} grams")

# Execute the plots and summary
plot_total_production_with_projection()
plot_total_production_summary()

# Try to execute the catplot if seaborn is available
try:
    # First check if our version of seaborn supports catplot
    if hasattr(sns, 'catplot'):
        plot_top_5_parcels()
    else:
        print("Warning: Your version of seaborn doesn't support catplot. Skipping top 5 parcels visualization.")
except Exception as e:
    print(f"Error creating top 5 parcels plot: {e}")

# Create Excel summary
create_excel_summary()

print("\nAll projections have been saved to the 'projections' directory.")
