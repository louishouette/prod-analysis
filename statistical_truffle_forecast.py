#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from datetime import datetime
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define file paths
SEASON_2023_2024_PATH = 'season_2023-2024.csv'
SEASON_2024_2025_PATH = 'season_2024-2025.csv'
RAMP_UP_PATH = 'ramp-up.csv'

# Set up output directories
OUTPUT_DIR = 'statistical_forecast'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def prepare_data_for_modeling(historical_data):
    """
    Prepare data for statistical modeling by creating features.
    """
    # Create a copy to avoid modifying the original data
    data = historical_data.copy()
    
    # Add calculated features
    data['Plant_Productivity'] = data.apply(
        lambda x: x['Plants Productifs'] / x['Plants'] * 100 if x['Plants'] > 0 else 0, 
        axis=1
    )
    data['Prod_per_Plant'] = data.apply(
        lambda x: x['Poids produit (g)'] / x['Plants'] if x['Plants'] > 0 else 0,
        axis=1
    )
    data['Age_Squared'] = data['Age'] ** 2
    data['Is_Recent_Season'] = data['Season'].apply(lambda x: 1 if x == '2024-2025' else 0)
    
    # Store original categorical values before one-hot encoding
    data['Original_Parcelle'] = data['Parcelle']
    data['Original_Espèce'] = data['Espèce']
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['Espèce'], prefix='Species')
    data = pd.get_dummies(data, columns=['Parcelle'], prefix='Parcelle')
    
    return data

def train_regression_models(X_train, y_train):
    """
    Train multiple regression models and return the best one.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_score = float('inf')
    best_model = None
    best_model_name = None
    
    print("Training regression models:")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-cv_scores.mean())
        
        print(f"  {name} - RMSE: {rmse:.4f}")
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_model_name = name
    
    print(f"Best model: {best_model_name} (RMSE: {best_score:.4f})")
    return best_model, best_model_name, best_score

def train_mixed_effects_model(data, formula, group_var):
    """
    Train a mixed-effects model to account for grouped data.
    """
    try:
        # Create and fit the mixed-effects model
        md = smf.mixedlm(formula, data, groups=data[group_var])
        result = md.fit()
        
        print(f"Mixed-effects model results:")
        print(f"  Formula: {formula}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        
        return result
    except Exception as e:
        print(f"Error fitting mixed-effects model: {e}")
        return None

def forecast_next_season(model, current_data, feature_cols=None, is_mixed_model=False):
    """
    Generate forecast for the next season.
    """
    # Make a copy of the current data for the forecast base
    forecast_data = current_data.copy()
    
    # Increment age by 1 for all plants
    forecast_data['Age'] = forecast_data['Age'] + 1
    forecast_data['Age_Squared'] = forecast_data['Age'] ** 2
    
    # Update season identifier
    forecast_data['Season'] = '2025-2026'
    forecast_data['Is_Recent_Season'] = 1
    
    # Make predictions
    if is_mixed_model:
        # For mixed-effects models
        forecast_data['Predicted_Prod_per_Plant'] = model.predict(forecast_data)
    else:
        # For scikit-learn models
        X_forecast = forecast_data[feature_cols].copy()
        forecast_data['Predicted_Prod_per_Plant'] = model.predict(X_forecast)
    
    # Ensure predictions are not negative
    forecast_data['Predicted_Prod_per_Plant'] = forecast_data['Predicted_Prod_per_Plant'].clip(lower=0)
    
    # Calculate expected total production
    forecast_data['Predicted_Total_Production'] = forecast_data['Predicted_Prod_per_Plant'] * forecast_data['Plants']
    
    # For trees with very young age, cap the predictions to reasonable values
    age_caps = {1: 0.2, 2: 0.5, 3: 1.5, 4: 3.0, 5: 5.0, 6: 8.0, 7: 12.0}
    
    for age, cap in age_caps.items():
        mask = forecast_data['Age'] == age
        forecast_data.loc[mask, 'Predicted_Prod_per_Plant'] = forecast_data.loc[mask, 'Predicted_Prod_per_Plant'].clip(upper=cap)
        forecast_data.loc[mask, 'Predicted_Total_Production'] = forecast_data.loc[mask, 'Predicted_Prod_per_Plant'] * forecast_data.loc[mask, 'Plants']
    
    return forecast_data

def analyze_forecast(forecast_data, historical_data):
    """
    Analyze and visualize the forecast results.
    """
    # Calculate summary statistics
    total_production = forecast_data['Predicted_Total_Production'].sum()
    avg_per_plant = forecast_data['Predicted_Total_Production'].sum() / forecast_data['Plants'].sum()
    
    print(f"\nForecast Summary for 2025-2026 Season:")
    print(f"  Total Production: {total_production:.2f}g")
    print(f"  Average per Plant: {avg_per_plant:.2f}g/plant")
    
    # Summary by parcelle
    parcelle_summary = forecast_data.groupby('Original_Parcelle').agg({
        'Plants': 'sum',
        'Predicted_Total_Production': 'sum'
    })
    parcelle_summary['Avg_per_Plant'] = parcelle_summary['Predicted_Total_Production'] / parcelle_summary['Plants']
    parcelle_summary = parcelle_summary.sort_values('Predicted_Total_Production', ascending=False)
    
    print("\nTop 5 Parcelles by Production:")
    for i, (parcelle, row) in enumerate(parcelle_summary.head(5).iterrows()):
        print(f"  {i+1}. Parcelle {parcelle}: {row['Predicted_Total_Production']:.2f}g ({row['Avg_per_Plant']:.2f}g/plant)")
    
    # Summary by species
    species_summary = forecast_data.groupby('Original_Espèce').agg({
        'Plants': 'sum',
        'Predicted_Total_Production': 'sum'
    })
    species_summary['Avg_per_Plant'] = species_summary['Predicted_Total_Production'] / species_summary['Plants']
    species_summary = species_summary.sort_values('Predicted_Total_Production', ascending=False)
    
    print("\nTop 5 Species by Production:")
    for i, (species, row) in enumerate(species_summary.head(5).iterrows()):
        print(f"  {i+1}. {species}: {row['Predicted_Total_Production']:.2f}g ({row['Avg_per_Plant']:.2f}g/plant)")
    
    # Create visualizations
    create_forecast_visualizations(forecast_data, historical_data)
    
    # Save results to CSV
    forecast_file = os.path.join(OUTPUT_DIR, 'statistical_forecast_2025-2026.csv')
    forecast_data.to_csv(forecast_file, sep=';', index=False)
    print(f"\nDetailed forecast saved to {forecast_file}")
    
    parcelle_summary_file = os.path.join(OUTPUT_DIR, 'parcelle_summary.csv')
    parcelle_summary.to_csv(parcelle_summary_file, sep=';')
    
    species_summary_file = os.path.join(OUTPUT_DIR, 'species_summary.csv')
    species_summary.to_csv(species_summary_file, sep=';')
    
    print(f"Summary files saved to {OUTPUT_DIR}/")

def create_forecast_visualizations(forecast_data, historical_data):
    """
    Create visualizations of the forecast results.
    """
    # 1. Total production by season
    plt.figure(figsize=(10, 6))
    
    historical_by_season = historical_data.groupby('Season')['Poids produit (g)'].sum()
    forecast_total = forecast_data['Predicted_Total_Production'].sum()
    
    seasons = list(historical_by_season.index) + ['2025-2026']
    productions = list(historical_by_season.values) + [forecast_total]
    
    plt.bar(seasons, productions, color=['blue', 'green', 'red'])
    plt.title('Total Truffle Production by Season')
    plt.xlabel('Season')
    plt.ylabel('Production (g)')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(productions):
        plt.text(i, v + 100, f"{v:.0f}g", ha='center')
    
    plt.savefig(os.path.join(PLOTS_DIR, 'total_production_by_season.png'))
    
    # 2. Production by age
    plt.figure(figsize=(12, 6))
    
    # Historical data by age
    age_data = {}
    for season in historical_data['Season'].unique():
        season_data = historical_data[historical_data['Season'] == season]
        age_prod = {}
        
        for age, group in season_data.groupby('Age'):
            prod_per_plant = group['Poids produit (g)'].sum() / group['Plants'].sum() if group['Plants'].sum() > 0 else 0
            age_prod[age] = prod_per_plant
        
        age_data[season] = age_prod
    
    # Forecast data by age
    forecast_by_age = {}
    for age, group in forecast_data.groupby('Age'):
        forecast_by_age[age] = group['Predicted_Prod_per_Plant'].mean()
    
    # Plot the data
    for season, age_vals in age_data.items():
        plt.plot(list(age_vals.keys()), list(age_vals.values()), 'o-', label=f'Actual {season}')
    
    plt.plot(list(forecast_by_age.keys()), list(forecast_by_age.values()), 's--', color='red', label='Forecast 2025-2026')
    
    plt.title('Production per Plant by Age')
    plt.xlabel('Age (years)')
    plt.ylabel('Production per Plant (g)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_age.png'))
    
    # 3. Forecast by top species
    plt.figure(figsize=(10, 6))
    
    species_prod = forecast_data.groupby('Original_Espèce')['Predicted_Total_Production'].sum().sort_values(ascending=False)
    top_species = species_prod.head(8)
    
    plt.pie(top_species, labels=[f"{s}: {v:.0f}g" for s, v in top_species.items()],
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Forecast Production by Species (2025-2026)')
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_species.png'))
    
    # 4. Forecast by parcelle
    plt.figure(figsize=(12, 6))
    
    parcelle_prod = forecast_data.groupby('Original_Parcelle')['Predicted_Total_Production'].sum().sort_values(ascending=False)
    
    plt.bar(parcelle_prod.index, parcelle_prod.values, color='lightgreen')
    plt.title('Forecast Production by Parcelle (2025-2026)')
    plt.xlabel('Parcelle')
    plt.ylabel('Production (g)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(parcelle_prod.values):
        plt.text(i, v + 100, f"{v:.0f}g", ha='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_parcelle.png'))
    
    print(f"Visualizations saved to {PLOTS_DIR}/")

def main():
    print("\n===== STATISTICAL TRUFFLE PRODUCTION FORECAST =====\n")
    
    # 1. Load and prepare data
    print("Loading data...")
    season_2023_2024 = load_data(SEASON_2023_2024_PATH)
    season_2024_2025 = load_data(SEASON_2024_2025_PATH)
    ramp_up = load_data(RAMP_UP_PATH)
    
    if season_2023_2024 is None or season_2024_2025 is None or ramp_up is None:
        print("Failed to load required data files.")
        return
    
    # Add season labels
    season_2023_2024['Season'] = '2023-2024'
    season_2024_2025['Season'] = '2024-2025'
    
    # Combine datasets for analysis
    historical_data = pd.concat([season_2023_2024, season_2024_2025])
    
    print(f"Data loaded successfully.")
    print(f"  Total rows: {len(historical_data)}")
    print(f"  Seasons: {', '.join(historical_data['Season'].unique())}")
    
    # 2. Prepare data for modeling
    print("\nPreparing data for modeling...")
    modeling_data = prepare_data_for_modeling(historical_data)
    
    # Target variable and features for regression models
    target = 'Prod_per_Plant'
    
    # Select important features
    numeric_features = ['Age', 'Age_Squared', 'Plant_Productivity', 'Is_Recent_Season']
    
    # Get categorical features (one-hot encoded)
    species_features = [col for col in modeling_data.columns if col.startswith('Species_')]
    parcelle_features = [col for col in modeling_data.columns if col.startswith('Parcelle_')]
    
    # Combine all features
    all_features = numeric_features + species_features + parcelle_features
    
    # Create feature matrix and target vector
    X = modeling_data[all_features]
    y = modeling_data[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train regression models
    print("\nTraining multiple regression models...")
    best_model, best_model_name, best_score = train_regression_models(X_train, y_train)
    
    # 4. Try mixed-effects model
    print("\nTraining mixed-effects model...")
    me_data = historical_data.copy()
    me_data['Prod_per_Plant'] = me_data['Poids produit (g)'] / me_data['Plants']
    
    # Define formula for mixed-effects model
    formula = "Prod_per_Plant ~ Age + Age**2 + C(Original_Espèce)"
    
    # Train mixed-effects model
    mixed_model = train_mixed_effects_model(me_data, formula, 'Original_Parcelle')
    
    # 5. Generate forecast based on best model
    print("\nGenerating forecast for 2025-2026 season...")
    
    # Use the best regression model for forecasting
    forecast_data = forecast_next_season(best_model, modeling_data, all_features, is_mixed_model=False)
    
    # 6. Analyze and visualize forecast results
    analyze_forecast(forecast_data, historical_data)
    
    print("\n===== Forecast Complete =====\n")

if __name__ == "__main__":
    main()
