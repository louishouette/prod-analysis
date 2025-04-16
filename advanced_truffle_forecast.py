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
OUTPUT_DIR = 'advanced_forecast'
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

# Function to prepare data for modeling
def prepare_modeling_data(historical_data):
    """
    Prepare data for statistical modeling by creating features and calculating targets.
    """
    # Create a copy to avoid modifying the original data
    data = historical_data.copy()
    
    # Generate additional features that might be predictive
    data['Plant_Productivity'] = data['Plants Productifs'] / data['Plants'] * 100
    data['Age_Squared'] = data['Age'] ** 2  # Non-linear age effect
    data['Log_Age'] = np.log1p(data['Age'])  # Log transformation for age
    
    # Calculate targets
    data['Prod_per_Plant'] = data['Poids produit (g)'] / data['Plants']
    data['Prod_per_Productive_Plant'] = data.apply(
        lambda x: x['Poids produit (g)'] / x['Plants Productifs'] if x['Plants Productifs'] > 0 else 0, axis=1
    )
    
    # Create season-specific features (e.g., is 2024-2025 season)
    data['Is_Recent_Season'] = data['Season'].apply(lambda x: 1 if x == '2024-2025' else 0)
    
    # Create species categorical encodings
    data = pd.get_dummies(data, columns=['Espèce'], prefix='Species')
    
    # Create parcelle categorical encodings
    data = pd.get_dummies(data, columns=['Parcelle'], prefix='Parcelle')
    
    return data

# Function to build various regression models
def build_models(X_train, y_train):
    """
    Build and compare multiple regression models.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Polynomial Regression': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    model_scores = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        model_scores[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std()
        }
        
        print(f"  Cross-validation RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    
    return trained_models, model_scores

# Function to optimize the best model with hyperparameter tuning
def optimize_best_model(X_train, y_train, best_model_name):
    """
    Optimize the best performing model with hyperparameter tuning.
    """
    print(f"Optimizing {best_model_name} with hyperparameter tuning...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    elif best_model_name == 'Ridge Regression':
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        model = Ridge(random_state=42)
    
    elif best_model_name == 'Lasso Regression':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        model = Lasso(random_state=42)
    
    elif best_model_name == 'Polynomial Regression':
        param_grid = {
            'poly__degree': [2, 3],
            'linear__fit_intercept': [True, False]
        }
        model = Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression())
        ])
    
    else:  # Linear Regression or default
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        model = LinearRegression()
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_

# Function to build mixed-effects models using statsmodels
def build_mixed_effects_models(data, formula_list, group_vars):
    """
    Build mixed-effects models to account for grouping variables.
    """
    mixed_models = {}
    mixed_model_results = {}
    
    for formula in formula_list:
        model_name = f"Mixed_Model_{len(mixed_models) + 1}"
        print(f"Fitting mixed-effects model: {formula}")
        
        try:
            # Create a mixed linear model
            md = smf.mixedlm(formula, data, groups=data[group_vars])
            mdf = md.fit()
            
            # Store model and results
            mixed_models[model_name] = mdf
            mixed_model_results[model_name] = {
                'formula': formula,
                'aic': mdf.aic,
                'bic': mdf.bic,
                'rsquared': mdf.rsquared,
                'rsquared_adj': mdf.rsquared_adj
            }
            
            print(f"  AIC: {mdf.aic:.2f}, BIC: {mdf.bic:.2f}")
            print(f"  R-squared: {mdf.rsquared:.4f}, Adj. R-squared: {mdf.rsquared_adj:.4f}")
            
        except Exception as e:
            print(f"  Error fitting mixed-effects model: {e}")
    
    return mixed_models, mixed_model_results

# Function to perform feature importance analysis
def analyze_feature_importance(model, feature_names, plot_path=None):
    """
    Analyze and visualize feature importance from the trained model.
    """
    # Get feature importances based on model type
    if hasattr(model, 'feature_importances_'):  # Tree-based models
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
    elif hasattr(model, 'coef_'):  # Linear models
        importances = np.abs(model.coef_)
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
    else:
        print("Model doesn't provide feature importance information.")
        return None
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()
    
    return feature_importance

# Function to generate forecast for next season
def forecast_next_season(model, current_data, X_features, group_by_vars=None):
    """
    Generate forecast for the next season based on the trained model.
    """
    # Make a copy of current data for forecasting
    forecast_base = current_data.copy()
    
    # Increment age by 1 for all plants
    forecast_base['Age'] = forecast_base['Age'] + 1
    forecast_base['Age Brut'] = forecast_base['Age Brut'] + 1
    
    # Update season identifier
    forecast_base['Season'] = '2025-2026'
    forecast_base['Is_Recent_Season'] = 1  # Now this is the recent season
    
    # Recalculate age-based features
    forecast_base['Age_Squared'] = forecast_base['Age'] ** 2
    forecast_base['Log_Age'] = np.log1p(forecast_base['Age'])
    
    # Create X matrix for prediction
    X_forecast = forecast_base[X_features].copy()
    
    # Make predictions
    if hasattr(model, 'predict'):
        # For scikit-learn models
        forecast_base['Predicted_Prod_per_Plant'] = model.predict(X_forecast)
    else:
        # For statsmodels models
        forecast_base['Predicted_Prod_per_Plant'] = model.predict(forecast_base)
    
    # Ensure predictions are not negative
    forecast_base['Predicted_Prod_per_Plant'] = forecast_base['Predicted_Prod_per_Plant'].clip(lower=0)
    
    # Calculate total predicted production
    forecast_base['Predicted_Total_Production'] = forecast_base['Predicted_Prod_per_Plant'] * forecast_base['Plants']
    
    # Calculate expected number of productive plants
    # We'll use a separate model or the current productivity rate with an adjustment factor
    if 'Plant_Productivity' in forecast_base.columns:
        # Apply a growth factor to the current productivity rate
        avg_productivity_growth = 1.3  # This can be refined based on historical patterns
        forecast_base['Predicted_Productivity_Rate'] = forecast_base['Plant_Productivity'] * avg_productivity_growth
        forecast_base['Predicted_Productivity_Rate'] = forecast_base['Predicted_Productivity_Rate'].clip(upper=100)
        forecast_base['Predicted_Productive_Plants'] = np.round(forecast_base['Plants'] * forecast_base['Predicted_Productivity_Rate'] / 100)
    
    # Summarize forecast if grouping variables are provided
    if group_by_vars:
        forecast_summary = forecast_base.groupby(group_by_vars).agg({
            'Plants': 'sum',
            'Predicted_Productive_Plants': 'sum',
            'Predicted_Total_Production': 'sum'
        })
        
        forecast_summary['Avg_Prod_per_Plant'] = forecast_summary['Predicted_Total_Production'] / forecast_summary['Plants']
        forecast_summary['Productivity_Rate'] = forecast_summary['Predicted_Productive_Plants'] / forecast_summary['Plants'] * 100
        
        return forecast_base, forecast_summary
    
    return forecast_base

# Function to evaluate model performance
def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate the model performance on test data.
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    
    # Add identity line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }

# Function to visualize model results
def visualize_forecast_results(historical_data, forecast_data, ramp_up=None):
    """
    Create visualizations of the forecast results.
    """
    # Create output directory for plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Total production by season
    plt.figure(figsize=(12, 6))
    
    # Aggregate historical data by season
    historical_by_season = historical_data.groupby('Season')['Poids produit (g)'].sum()
    
    # Get forecast total
    forecast_total = forecast_data['Predicted_Total_Production'].sum()
    
    # Create bar chart
    seasons = list(historical_by_season.index) + ['2025-2026']
    productions = list(historical_by_season.values) + [forecast_total]
    
    plt.bar(seasons, productions, color=['blue', 'green', 'red'])
    plt.title('Total Truffle Production by Season', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Production (g)', fontsize=12)
    
    # Add data labels on bars
    for i, v in enumerate(productions):
        plt.text(i, v + 0.1, f"{v:.1f}g", ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'total_production_by_season.png'))
    
    # 2. Production by age
    plt.figure(figsize=(14, 8))
    
    # Aggregate historical data by age
    production_by_age = {}
    
    for season in historical_data['Season'].unique():
        season_data = historical_data[historical_data['Season'] == season]
        age_prod = {}
        
        for age, group in season_data.groupby('Age'):
            total_plants = group['Plants'].sum()
            total_production = group['Poids produit (g)'].sum()
            prod_per_plant = total_production / total_plants if total_plants > 0 else 0
            age_prod[age] = prod_per_plant
        
        production_by_age[season] = age_prod
    
    # Get forecast production by age
    forecast_by_age = {}
    for age, group in forecast_data.groupby('Age'):
        forecast_by_age[age] = group['Predicted_Prod_per_Plant'].mean()
    
    # Plot all series
    for season, age_data in production_by_age.items():
        ages = list(age_data.keys())
        values = list(age_data.values())
        plt.plot(ages, values, 'o-', label=f'Actual {season}')
    
    # Plot forecast
    forecast_ages = list(forecast_by_age.keys())
    forecast_values = list(forecast_by_age.values())
    plt.plot(forecast_ages, forecast_values, 's--', color='red', linewidth=2, label='Forecast 2025-2026')
    
    # Add ramp-up target if provided
    if ramp_up is not None:
        plt.plot(ramp_up['Age'], ramp_up['Production au plant (g)'], '*-.', color='gray', alpha=0.5, label='Ramp-up Target (Reference)')
    
    plt.title('Production per Plant by Age', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Production per Plant (g)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_age.png'))
    
    # 3. Forecast by species
    plt.figure(figsize=(14, 8))
    
    # Get top species by total production
    species_forecast = forecast_data.groupby('Espèce').agg({
        'Plants': 'sum',
        'Predicted_Total_Production': 'sum',
        'Predicted_Prod_per_Plant': 'mean'
    }).sort_values('Predicted_Total_Production', ascending=False)
    
    # Plot top 8 species for readability
    top_species = species_forecast.head(8)
    
    # Create bar chart
    plt.barh(top_species.index, top_species['Predicted_Total_Production'], color='skyblue')
    plt.title('Forecast Production by Species (2025-2026)', fontsize=14)
    plt.xlabel('Total Production (g)', fontsize=12)
    plt.ylabel('Species', fontsize=12)
    
    # Add data labels
    for i, v in enumerate(top_species['Predicted_Total_Production']):
        plt.text(v + 100, i, f"{v:.1f}g", va='center')
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_species.png'))
    
    # 4. Forecast by parcelle
    plt.figure(figsize=(14, 8))
    
    # Aggregate by parcelle
    parcelle_forecast = forecast_data.groupby('Parcelle').agg({
        'Plants': 'sum',
        'Predicted_Total_Production': 'sum',
        'Predicted_Prod_per_Plant': 'mean'
    }).sort_values('Predicted_Total_Production', ascending=False)
    
    # Create bar chart
    plt.bar(parcelle_forecast.index, parcelle_forecast['Predicted_Total_Production'], color='lightgreen')
    plt.title('Forecast Production by Parcelle (2025-2026)', fontsize=14)
    plt.xlabel('Parcelle', fontsize=12)
    plt.ylabel('Total Production (g)', fontsize=12)
    
    # Add data labels
    for i, v in enumerate(parcelle_forecast['Predicted_Total_Production']):
        plt.text(i, v + 100, f"{v:.1f}g", ha='center', rotation=90)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_by_parcelle.png'))
    
    # 5. Age distribution of production
    plt.figure(figsize=(12, 6))
    
    age_production = forecast_data.groupby('Age')['Predicted_Total_Production'].sum()
    plt.pie(age_production, labels=[f'Age {age}: {prod:.1f}g' for age, prod in age_production.items()],
           autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Forecast Production by Tree Age (2025-2026)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'production_distribution_by_age.png'))
    
    return True

# Main function to run the advanced statistical forecast
def main():
    print("
===== ADVANCED STATISTICAL TRUFFLE PRODUCTION FORECAST =====
")
    
    # Load data
    print("Loading data...")
    season_2023_2024 = load_data(SEASON_2023_2024_PATH)
    season_2024_2025 = load_data(SEASON_2024_2025_PATH)
    ramp_up = load_data(RAMP_UP_PATH)
    
    if season_2023_2024 is None or season_2024_2025 is None or ramp_up is None:
        print("Failed to load required data files.")
        return
    
    # Label seasons
    season_2023_2024['Season'] = '2023-2024'
    season_2024_2025['Season'] = '2024-2025'
    
    # Combine data for modeling
    historical_data = pd.concat([season_2023_2024, season_2024_2025])
    
    print(f"Data loaded: {len(historical_data)} rows total")
    print(f"Season 2023-2024: {len(season_2023_2024)} rows")
    print(f"Season 2024-2025: {len(season_2024_2025)} rows")
    
    # Prepare data for modeling
    print("
--- Preparing data for modeling ---")
    modeling_data = prepare_modeling_data(historical_data)
    print(f"Prepared data with {modeling_data.shape[1]} features")
    
    # Define target variable and features
    target = 'Prod_per_Plant'  # Production per plant (g)
    
    # Features that might predict production per plant
    numeric_features = ['Age', 'Age_Squared', 'Log_Age', 'Plant_Productivity', 'Is_Recent_Season']
    
    # Get all dummy columns for species and parcelles
    species_cols = [col for col in modeling_data.columns if col.startswith('Species_')]
    parcelle_cols = [col for col in modeling_data.columns if col.startswith('Parcelle_')]
    
    # Combine all features
    all_features = numeric_features + species_cols + parcelle_cols
    
    # Create feature matrix and target vector
    X = modeling_data[all_features]
    y = modeling_data[target]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build and evaluate multiple regression models
    print("
--- Building Regression Models ---")
    trained_models, model_scores = build_models(X_train, y_train)
    
    # Find the best performing model
    best_model_name = min(model_scores, key=lambda k: model_scores[k]['mean_rmse'])
    print(f"Best model: {best_model_name} with RMSE: {model_scores[best_model_name]['mean_rmse']:.4f}")
    
    # Optimize the best model
    best_model = optimize_best_model(X_train, y_train, best_model_name)
    
    # Evaluate the optimized model
    print("
--- Evaluating Optimized Model ---")
    evaluation = evaluate_model_performance(best_model, X_test, y_test)
    
    # Analyze feature importance
    print("
--- Analyzing Feature Importance ---")
    feature_importance = analyze_feature_importance(
        best_model, X.columns, 
        os.path.join(PLOTS_DIR, 'feature_importance.png')
    )
    
    print("Top 10 most important features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                               feature_importance['Importance'].head(10))):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Try mixed-effects models for grouped data
    print("
--- Building Mixed-Effects Models ---")
    
    # Prepare data for mixed-effects modeling (no dummies needed)
    me_data = historical_data.copy()
    me_data['Prod_per_Plant'] = me_data['Poids produit (g)'] / me_data['Plants']
    me_data['Plant_Productivity'] = me_data['Plants Productifs'] / me_data['Plants'] * 100
    me_data['Age_Squared'] = me_data['Age'] ** 2
    me_data['Log_Age'] = np.log1p(me_data['Age'])
    
    # Define formulas for mixed-effects models
    # These model different fixed effects while treating Parcelle as a random effect
    formulas = [
        "Prod_per_Plant ~ Age + C(Espèce)",
        "Prod_per_Plant ~ Age + Age_Squared + C(Espèce)",
        "Prod_per_Plant ~ Age + Log_Age + C(Espèce)",
        "Prod_per_Plant ~ Age + Plant_Productivity + C(Espèce)",
        "Prod_per_Plant ~ Age + Age_Squared + Plant_Productivity + C(Espèce)"
    ]
    
    # Build mixed-effects models
    mixed_models, mixed_results = build_mixed_effects_models(me_data, formulas, 'Parcelle')
    
    # Find best mixed-effects model based on AIC
    if mixed_results:
        best_mixed_model_name = min(mixed_results, key=lambda k: mixed_results[k]['aic'])
        print(f"Best mixed-effects model: {best_mixed_model_name}")
        print(f"Formula: {mixed_results[best_mixed_model_name]['formula']}")
        print(f"AIC: {mixed_results[best_mixed_model_name]['aic']:.2f}")
        print(f"R-squared: {mixed_results[best_mixed_model_name]['rsquared']:.4f}")
        
        # Use best mixed-effects model if it performs better than regression models
        if 'rsquared' in mixed_results[best_mixed_model_name] and mixed_results[best_mixed_model_name]['rsquared'] > 0.7:
            print("Using mixed-effects model for forecasting (better fit for grouped data)")
            use_mixed_model = True
            best_final_model = mixed_models[best_mixed_model_name]
        else:
            print("Using regression model for forecasting (better overall performance)")
            use_mixed_model = False
            best_final_model = best_model
    else:
        print("No valid mixed-effects models could be fit. Using regression model.")
        use_mixed_model = False
        best_final_model = best_model
    
    # Generate forecast for 2025-2026 season
    print("
--- Generating Forecast for 2025-2026 Season ---")
    
    if use_mixed_model:
        # For mixed-effects models, we use the stats model approach
        forecast_data = forecast_next_season(best_final_model, me_data, [], ['Parcelle', 'Espèce'])
    else:
        # For regression models, we pass features
        forecast_data = forecast_next_season(best_final_model, modeling_data, all_features, ['Parcelle', 'Espèce'])
    
    if isinstance(forecast_data, tuple):
        forecast_base, forecast_summary = forecast_data
        
        # Display summary statistics
        print(f"Total Forecast Production: {forecast_summary['Predicted_Total_Production'].sum():.2f}g")
        
        # Save forecast to CSV
        forecast_file = os.path.join(OUTPUT_DIR, 'advanced_forecast_2025-2026.csv')
        forecast_base.to_csv(forecast_file, sep=';', index=False)
        
        summary_file = os.path.join(OUTPUT_DIR, 'advanced_forecast_summary.csv')
        forecast_summary.to_csv(summary_file, sep=';')
        
        print(f"Detailed forecast saved to {forecast_file}")
        print(f"Summary saved to {summary_file}")
        
        # Visualize results
        print("
--- Creating Visualizations ---")
        visualize_forecast_results(historical_data, forecast_base, ramp_up)
        print(f"Visualizations saved to {PLOTS_DIR}/")
    else:
        print("Error: Forecast generation failed.")
    
    print("
===== Forecast Complete =====")
    
# Run the main function
if __name__ == "__main__":
    main()
