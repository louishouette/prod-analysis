#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy import optimize
import os

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

# Custom color palette for quartiles
quartile_colors = {
    'Q1 (0-25%)': '#3498db',    # Blue
    'Q2 (25-50%)': '#2ecc71',   # Green
    'Q3 (50-75%)': '#f39c12',   # Orange
    'Q4 (75-100%)': '#e74c3c',  # Red
    'All Parcels': '#9b59b6'     # Purple
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

# Define KPIs to analyze
kpis = [
    'Plants Productifs',
    'Taux de Productivité (%)',
    'Poids produit (g)',
    'Nombre de truffe',
    'Poids moyen (g)',
    'Production au plant (g)'
]

# Create a directory for quartile plots if it doesn't exist
if not os.path.exists('quartile_projections'):
    os.makedirs('quartile_projections')

# Linear function for trendline fitting (simpler for 2 data points)
def linear_func(x, a, b):
    return a * x + b

# Function to fit a linear trend and predict next value (more appropriate for 2 data points)
def fit_trend_curve(x_data, y_data, predict_x):
    try:
        # With only 2 data points, we can do a simple projection
        if len(x_data) == 2 and len(y_data) == 2:
            # Calculate growth rate between the two points
            growth_rate = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
            
            # Use a modified logarithmic growth to avoid unrealistic projections
            # for large growth rates - this assumes growth will slow down
            if growth_rate > 0:
                # Calculate projected value using diminishing growth
                base_projection = y_data[1] + growth_rate
                predicted_y = y_data[1] + growth_rate * (np.log((predict_x - x_data[1]) + 1) / np.log(2))
            else:
                # For negative growth, continue the trend linearly
                predicted_y = y_data[1] + growth_rate
            
            # Generate values for trendline plotting
            x_range = np.linspace(min(x_data), predict_x, 100)
            
            # Create a smooth transition curve between last known point and projection
            y_range = []
            for xi in x_range:
                if xi <= x_data[1]:
                    # Linear between known points
                    ratio = (xi - x_data[0]) / (x_data[1] - x_data[0]) if x_data[1] != x_data[0] else 0
                    yi = y_data[0] + ratio * (y_data[1] - y_data[0])
                else:
                    # Modified logarithmic growth after last known point
                    if growth_rate > 0:
                        yi = y_data[1] + growth_rate * (np.log((xi - x_data[1]) + 1) / np.log(2))
                    else:
                        yi = y_data[1] + growth_rate * (xi - x_data[1])
                y_range.append(yi)
            
            # Ensure predicted value is non-negative
            predicted_y = max(0, predicted_y)
            
            return predicted_y, x_range, np.array(y_range), None
        else:
            # Use linear function if we have more data points in the future
            params, _ = optimize.curve_fit(linear_func, x_data, y_data)
            predicted_y = linear_func(predict_x, *params)
            x_range = np.linspace(min(x_data), predict_x, 100)
            y_range = linear_func(x_range, *params)
            return predicted_y, x_range, y_range, params
    except Exception as e:
        print(f"Warning: Could not fit trend curve: {e}")
        return None, None, None, None

# Function to plot KPI by quartiles with trend projection
def plot_kpi_by_quartiles(kpi):
    print(f"Processing {kpi}...")
    
    # Step 1: Calculate quartiles for each season based on the KPI
    quartile_data = []
    
    # Get all parcels for consistent assignment
    all_parcels = sorted(combined_data['Parcelle'].unique())
    
    # Create a tracking dictionary for quartile assignment consistency across seasons
    parcel_quartile_map = {}
    
    # First, get the quartile assignments from the most recent season
    recent_season = '2024-2025'
    recent_data = combined_data[combined_data['Season'] == recent_season]
    
    if not recent_data.empty:
        # Group by parcel and calculate average KPI
        recent_avg = recent_data.groupby('Parcelle')[kpi].mean().reset_index()
        
        if not recent_avg.empty and recent_avg[kpi].notna().any():
            # Filter out NaN values for quartile calculation
            valid_avg = recent_avg[recent_avg[kpi].notna()]
            
            if len(valid_avg) >= 4:  # Need at least 4 data points for quartiles
                quartiles = valid_avg[kpi].quantile([0.25, 0.5, 0.75]).values
                
                # Assign quartile groups to parcels
                recent_avg['Quartile'] = pd.cut(
                    recent_avg[kpi], 
                    bins=[-float('inf'), quartiles[0], quartiles[1], quartiles[2], float('inf')],
                    labels=['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
                )
                
                # Store the quartile assignments
                for _, row in recent_avg.iterrows():
                    if pd.notna(row['Quartile']):
                        parcel_quartile_map[row['Parcelle']] = row['Quartile']
            else:
                # If fewer than 4 data points, assign evenly
                values = valid_avg[kpi].values
                sorted_indices = np.argsort(values)
                quartile_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
                
                n = len(valid_avg)
                for i, idx in enumerate(sorted_indices):
                    quartile_idx = min(3, int(4 * i / n))
                    parcel = valid_avg.iloc[idx]['Parcelle']
                    parcel_quartile_map[parcel] = quartile_labels[quartile_idx]
    
    # Now process each season with consistent quartile assignments
    for season in ['2023-2024', '2024-2025']:
        season_data = combined_data[combined_data['Season'] == season]
        
        # Group by parcel and calculate average KPI
        parcel_avg = season_data.groupby('Parcelle')[kpi].mean().reset_index()
        
        # Skip if no data for this season
        if parcel_avg.empty:
            continue
            
        # Assign quartiles based on the mapping we created
        parcel_avg['Quartile'] = parcel_avg['Parcelle'].map(parcel_quartile_map)
        
        # For parcels without a quartile assignment (e.g., new parcels), use median
        parcel_avg['Quartile'].fillna('Q2 (25-50%)', inplace=True)
        
        # Add season column
        parcel_avg['Season'] = season
        
        # Find numerical season for curve fitting
        parcel_avg['Season_Num'] = 1 if season == '2023-2024' else 2
        
        quartile_data.append(parcel_avg)
    
    # Combine data from both seasons
    if not quartile_data:
        print(f"No data available for {kpi}")
        return
        
    quartile_df = pd.concat(quartile_data, ignore_index=True)
    
    # Calculate average KPI value by quartile and season
    # observed=True to avoid the FutureWarning
    quartile_avg = quartile_df.groupby(['Quartile', 'Season', 'Season_Num'], observed=True)[kpi].mean().reset_index()
    
    # Also calculate average across all parcels
    all_parcels_avg = quartile_df.groupby(['Season', 'Season_Num'])[kpi].mean().reset_index()
    all_parcels_avg['Quartile'] = 'All Parcels'
    
    # Combine with quartile data
    plot_data = pd.concat([quartile_avg, all_parcels_avg], ignore_index=True)
    
    # Sort data
    plot_data = plot_data.sort_values(['Quartile', 'Season_Num'])
    
    # Ensure we have complete data for each quartile and season
    # Get unique quartiles and seasons
    quartiles = plot_data['Quartile'].unique()
    seasons = ['2023-2024', '2024-2025']
    season_nums = [1, 2]
    
    # Check for missing combinations and fill with zeros if needed
    complete_rows = []
    for q in quartiles:
        for i, s in enumerate(seasons):
            existing = plot_data[(plot_data['Quartile'] == q) & (plot_data['Season'] == s)]
            if existing.empty:
                complete_rows.append({
                    'Quartile': q,
                    'Season': s,
                    'Season_Num': season_nums[i],
                    kpi: 0.0  # Use zero as placeholder
                })
    
    if complete_rows:
        plot_data = pd.concat([plot_data, pd.DataFrame(complete_rows)], ignore_index=True)
        plot_data = plot_data.sort_values(['Quartile', 'Season_Num'])
    
    # Create plot
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Season labels for x-axis
    season_labels = ['2023-2024', '2024-2025', '2025-2026 (Projected)']
    
    # Store projection data for later use
    projections = []
    
    # Plot data for each quartile with trendline
    for quartile in plot_data['Quartile'].unique():
        quartile_data = plot_data[plot_data['Quartile'] == quartile]
        
        # Extract x and y data points
        x = quartile_data['Season_Num'].values
        y = quartile_data[kpi].values
        
        # Only proceed if we have enough data points and they're valid
        if len(x) < 2 or np.isnan(y).any():
            continue
        
        # Plot actual data points
        plt.plot(x, y, 'o-', label=quartile, color=quartile_colors.get(quartile, 'gray'), 
                 linewidth=2, markersize=8)
        
        # Fit trend curve and project next season
        predicted_y, x_curve, y_curve, params = fit_trend_curve(x, y, 3)  # 3 represents the third season
        
        if predicted_y is not None:
            # Plot the trendline
            plt.plot(x_curve, y_curve, '--', color=quartile_colors.get(quartile, 'gray'), alpha=0.7)
            
            # Plot the projected point
            plt.plot(3, predicted_y, 'o', color=quartile_colors.get(quartile, 'gray'), 
                     markersize=10, markerfacecolor='white')
            
            # Add value label to projected point
            plt.annotate(f'{predicted_y:.1f}', 
                      (3, predicted_y),
                      textcoords="offset points",
                      xytext=(0, 10),
                      ha='center',
                      fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
            # Store projection for later use
            projections.append({'Quartile': quartile, 'Projected_Value': predicted_y})
    
    # Set x-axis to show season names
    plt.xticks([1, 2, 3], season_labels)
    
    # Format y-axis with thousand separator if needed
    def format_with_comma(x, pos):
        return f'{x:,.1f}'.replace(',', ' ')
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_comma))
    
    # Customize the plot
    plt.title(f'{kpi} by Parcel Quartiles with Trend Projection', fontweight='bold')
    plt.xlabel('Season', fontweight='bold')
    plt.ylabel(kpi, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')  # Light background
    
    # Add a box around the projected region
    plt.axvspan(2.5, 3.5, alpha=0.1, color='gray')
    plt.text(3, plt.gca().get_ylim()[0] + (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])*0.05, 
             'PROJECTED', ha='center', fontsize=10, fontstyle='italic', alpha=0.7)
    
    # Add legend
    plt.legend(title='Parcel Quartiles', loc='best')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'quartile_projections/{kpi.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")}.png')
    plt.close()
    
    return projections

# Execute the plots for each KPI
projection_summary = {}
for kpi in kpis:
    projections = plot_kpi_by_quartiles(kpi)
    if projections:
        projection_summary[kpi] = projections

# Create a summary dataframe of all projections
def create_projection_summary():
    summary_rows = []
    
    for kpi, projections in projection_summary.items():
        for p in projections:
            summary_rows.append({
                'KPI': kpi,
                'Quartile': p['Quartile'],
                'Projected_Value_2025_2026': p['Projected_Value']
            })
    
    if not summary_rows:
        print("No projection data available for summary")
        return
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    summary_df.to_csv('quartile_projections/quartile_projections_summary.csv', index=False, sep=';')
    
    print("\nQuartile-based Projections Summary:")
    print(summary_df.to_string(index=False))

# Create projection summary
create_projection_summary()

print("\nAll quartile-based projections have been saved to the 'quartile_projections' directory.")
