#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import warnings
from scipy.optimize import curve_fit

# Set up aesthetics for plots
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
warnings.filterwarnings('ignore')

# Load the data
print('Loading data...')
production_data = pd.read_csv('production.csv', sep=';')
rampup_data = pd.read_csv('ramp-up.csv', sep=';')

# Initial data exploration
print('\nProduction data overview:')
print(production_data.shape)
print(production_data.columns.tolist())

print('\nRamp-up data overview:')
print(rampup_data.shape)
print(rampup_data.columns.tolist())

# Prepare the data
def prepare_data(production_data):
    # Convert data types
    production_data['Age'] = pd.to_numeric(production_data['Age'], errors='coerce')
    production_data['Production au plant (g)'] = pd.to_numeric(production_data['Production au plant (g)'], errors='coerce')
    production_data['Plants'] = pd.to_numeric(production_data['Plants'], errors='coerce')
    production_data['Taux de Productivité (%)'] = pd.to_numeric(production_data['Taux de Productivité (%)'], errors='coerce')
    
    # Extract season year (the first year of the season range)
    production_data['Season_Year'] = production_data['Saison'].str.split(' - ').str[0].astype(int)
    
    # Filter rows with valid age and production values
    valid_data = production_data[
        (production_data['Age'] > 0) &
        (~production_data['Production au plant (g)'].isna())
    ].copy()
    
    # Group by parcel, species, and age for analysis
    grouped_data = valid_data.groupby(['Parcelle', 'Espèce', 'Age']).agg({
        'Plants': 'sum',
        'Plants Productifs': 'sum',
        'Poids produit (g)': 'sum',
        'Production au plant (g)': 'mean',  # Average production per plant
        'Season_Year': 'max'  # Most recent season
    }).reset_index()
    
    # Calculate productive plant percentage
    grouped_data['Productive_Percentage'] = 100 * grouped_data['Plants Productifs'] / grouped_data['Plants']
    
    # Replace zeros with NaN for log transformations
    grouped_data.loc[grouped_data['Production au plant (g)'] == 0, 'Production au plant (g)'] = np.nan
    
    return valid_data, grouped_data

# Fit Gompertz function to ramp-up data to get fixed gamma parameter
def gompertz(age, A, beta, gamma):
    """Gompertz growth function"""
    return A * np.exp(-beta * np.exp(-gamma * age))

def fit_ramp_up_curve(rampup_data):
    # Convert data for fitting
    x_data = rampup_data['Age'].values
    y_data = rampup_data['Production au plant (g)'].values
    
    # Initial parameter guess
    p0 = [100, 5, 0.3]  # A, beta, gamma
    
    try:
        # Fit the Gompertz function
        params, pcov = curve_fit(gompertz, x_data, y_data, p0=p0)
        A_fixed, beta_fixed, gamma_fixed = params
        
        # Calculate R-squared
        y_pred = gompertz(x_data, *params)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        ss_res = np.sum((y_data - y_pred)**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f'\nFixed effect parameters from ramp-up curve:')
        print(f'A (asymptote): {A_fixed:.2f}')
        print(f'beta: {beta_fixed:.4f}')
        print(f'gamma: {gamma_fixed:.4f}')
        print(f'R-squared: {r_squared:.4f}')
        
        return gamma_fixed, A_fixed, beta_fixed, x_data, y_data, y_pred
        
    except Exception as e:
        print(f"Error fitting ramp-up curve: {e}")
        return 0.3, 100, 5, x_data, y_data, np.zeros_like(y_data)  # Default values

# Bayesian hierarchical model for parcel-specific parameters
def build_bayesian_model(data, gamma_fixed, A_mean, beta_mean):
    # Prepare data for modeling
    parcels = data['Parcelle'].unique()
    parcel_indices = {parcel: idx for idx, parcel in enumerate(parcels)}
    
    # Map parcels to indices
    data['parcel_idx'] = data['Parcelle'].map(parcel_indices)
    
    # Filter out NaN values
    model_data = data.dropna(subset=['Age', 'Production au plant (g)'])
    
    # Build the model
    with pm.Model() as model:
        # Hyperpriors for random effects
        A_mu = pm.Normal("A_mu", mu=A_mean, sigma=50)
        A_sigma = pm.HalfNormal("A_sigma", sigma=25)
        
        beta_mu = pm.Normal("beta_mu", mu=beta_mean, sigma=2)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=1)
        
        # Parcel-level random effects
        A = pm.Normal("A", mu=A_mu, sigma=A_sigma, shape=len(parcels))
        beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, shape=len(parcels))
        
        # Fixed effect (from ramp-up curve)
        gamma = pm.Deterministic("gamma", pm.math.constant(gamma_fixed))
        
        # Expected production based on Gompertz function
        age = pm.MutableData("age", model_data['Age'].values)
        parcel_idx = pm.MutableData("parcel_idx", model_data['parcel_idx'].values)
        
        mu = pm.Deterministic(
            "mu", 
            A[parcel_idx] * pm.math.exp(-beta[parcel_idx] * pm.math.exp(-gamma * age))
        )
        
        # Likelihood for observed production
        sigma = pm.HalfNormal("sigma", sigma=10)
        obs = pm.Normal(
            "obs", 
            mu=mu, 
            sigma=sigma, 
            observed=model_data['Production au plant (g)'].values
        )
        
    return model, model_data, parcels, parcel_indices

# Project future production
def project_future_production(trace, model_data, parcels, gamma_fixed, max_age=12):
    # Extract posterior samples
    A_samples = az.extract(trace, var_names="A").to_numpy()
    beta_samples = az.extract(trace, var_names="beta").to_numpy()
    
    # Calculate mean of posterior samples for each parcel
    A_mean = A_samples.mean(axis=0)
    beta_mean = beta_samples.mean(axis=0)
    
    # Project production for each parcel and age
    ages = np.arange(1, max_age + 1)
    projections = {}
    
    for i, parcel in enumerate(parcels):
        parcel_proj = np.zeros(max_age)
        parcel_A = A_mean[i]
        parcel_beta = beta_mean[i]
        
        for age in ages:
            pred = gompertz(age, parcel_A, parcel_beta, gamma_fixed)
            parcel_proj[age-1] = pred
        
        projections[parcel] = parcel_proj
    
    # Convert to DataFrame
    proj_df = pd.DataFrame(projections, index=ages)
    proj_df.index.name = 'Age'
    
    return proj_df

# Main analysis
def main():
    # Prepare the data
    valid_data, grouped_data = prepare_data(production_data)
    
    # Fit the ramp-up curve to get the fixed gamma parameter
    gamma_fixed, A_fixed, beta_fixed, x_ramp, y_ramp, y_pred = fit_ramp_up_curve(rampup_data)
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs('generated/plots/production_projections', exist_ok=True)
    os.makedirs('generated/data/projections', exist_ok=True)
    
    # Plot the ramp-up curve and fit
    plt.figure(figsize=(10, 6))
    plt.scatter(x_ramp, y_ramp, color='darkblue', label='Target Ramp-up Data')
    plt.plot(x_ramp, y_pred, color='red', linestyle='--', label='Gompertz Fit')
    plt.xlabel('Tree Age (years)')
    plt.ylabel('Production per Plant (g)')
    plt.title('Truffle Production Ramp-up Curve and Gompertz Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig('generated/plots/production_projections/rampup_curve_fit.png', dpi=300, bbox_inches='tight')
    
    # Visualize actual production data by age
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age', y='Production au plant (g)', data=valid_data)
    plt.title('Actual Production by Tree Age')
    plt.savefig('generated/plots/production_projections/actual_production_by_age.png', dpi=300, bbox_inches='tight')
    
    # Build and sample from the Bayesian model
    print("\nBuilding Bayesian hierarchical model...")
    model, model_data, parcels, parcel_indices = build_bayesian_model(grouped_data, gamma_fixed, A_fixed, beta_fixed)
    
    with model:
        # Sample from the posterior
        trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
    # Summary of posterior samples
    print("\nPosterior summary:")
    summary = az.summary(trace, var_names=["A", "beta", "gamma", "sigma"])
    print(summary)
    
    # Create model diagnostics directory if it doesn't exist
    os.makedirs('generated/plots/model_diagnostics', exist_ok=True)
    
    # Plot trace
    az.plot_trace(trace, var_names=["A_mu", "beta_mu", "A_sigma", "beta_sigma", "sigma"])
    plt.savefig('generated/plots/model_diagnostics/model_trace.png', dpi=300, bbox_inches='tight')
    
    # Project future production
    print("\nProjecting future production for all parcels...")
    projections = project_future_production(trace, model_data, parcels, gamma_fixed)
    
    # Save projections
    projections.to_csv('generated/data/projections/projected_production.csv')
    print("Projections saved to 'generated/data/projections/projected_production.csv'")
    
    # Visualize projections
    plt.figure(figsize=(14, 10))
    projections.plot(marker='o', linestyle='-')
    
    # Add the ramp-up target line
    ages_extended = np.arange(1, projections.index.max() + 1)
    target_curve = gompertz(ages_extended, A_fixed, beta_fixed, gamma_fixed)
    plt.plot(ages_extended, target_curve, 'k--', linewidth=2, label='Target Curve')
    
    plt.title('Projected Truffle Production by Parcel and Age')
    plt.xlabel('Tree Age (years)')
    plt.ylabel('Production per Plant (g)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/production_projections.png', dpi=300, bbox_inches='tight')
    
    # Calculate ratios relative to target curve
    ratio_df = projections.copy()
    for age in ratio_df.index:
        target_value = gompertz(age, A_fixed, beta_fixed, gamma_fixed)
        ratio_df.loc[age] = ratio_df.loc[age] / target_value if target_value > 0 else np.nan
    
    # Save ratios
    ratio_df.to_csv('generated/data/projections/production_ratios.csv')
    print("Production ratio projections saved to 'generated/data/projections/production_ratios.csv'")
    
    # Visualize ratios
    plt.figure(figsize=(14, 10))
    ratio_df.plot(marker='o', linestyle='-')
    plt.axhline(y=1.0, color='k', linestyle='--', label='Target Ratio (1.0)')
    plt.title('Projected Production Ratios Relative to Target Curve')
    plt.xlabel('Tree Age (years)')
    plt.ylabel('Production Ratio (Actual/Target)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('generated/plots/production_projections/production_ratios.png', dpi=300, bbox_inches='tight')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
