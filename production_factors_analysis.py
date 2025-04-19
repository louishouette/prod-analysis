#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set visual style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Load the data
print('Loading production data...')
production_data = pd.read_csv('production.csv', sep=';')

# Data preparation
def prepare_data(df):
    # Convert relevant columns to numeric
    numeric_cols = ['Age', 'Age Brut', 'Plants', 'Plants Productifs', 
                   'Taux de Productivité (%)', 'Poids produit (g)', 
                   'Nombre de truffe', 'Poids moyen (g)', 'Production au plant (g)']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract season
    df['Season_Year'] = df['Saison'].str.split(' - ').str[0].astype(int)
    
    # Calculate derived metrics
    df['Has_Production'] = (df['Plants Productifs'] > 0).astype(int)
    df['Average_Weight_Per_Truffle'] = df['Poids moyen (g)']
    df['Truffle_Quantity'] = df['Nombre de truffe']
    df['Total_Weight'] = df['Poids produit (g)']
    
    # Filter valid data
    valid_data = df[df['Age'] > 0].copy()
    
    return valid_data

# Analyzing precocity by species and age
def analyze_precocity(df):
    print('\n--- PRECOCITY ANALYSIS ---')
    
    # Group by species and age to see when production starts
    precocity = df.groupby(['Espèce', 'Age'])['Has_Production'].mean().reset_index()
    precocity = precocity.pivot(index='Age', columns='Espèce', values='Has_Production')
    
    # Calculate the youngest age with production for each species
    youngest_productive_age = {}
    for species in df['Espèce'].unique():
        species_data = df[df['Espèce'] == species]
        productive_rows = species_data[species_data['Plants Productifs'] > 0]
        if not productive_rows.empty:
            youngest_productive_age[species] = productive_rows['Age'].min()
    
    print('\nYoungest age with production by species:')
    for species, age in sorted(youngest_productive_age.items(), key=lambda x: x[1]):
        print(f'{species}: {age} years')
    
    # Plot precocity
    plt.figure(figsize=(12, 8))
    sns.heatmap(precocity, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5)
    plt.title('Production Rate by Species and Age')
    plt.savefig('precocity_by_species_age.png', dpi=300, bbox_inches='tight')
    
    # Statistical test for age effect on precocity
    model = smf.logit('Has_Production ~ Age', data=df).fit()
    print('\nEffect of Age on Production Probability (Logistic Regression):')
    print(model.summary().tables[1])
    
    # Statistical test for species effect on precocity (controlling for age)
    # Create dummy variables for species
    dummy_species = pd.get_dummies(df['Espèce'], prefix='Species', drop_first=True)
    model_data = pd.concat([df[['Has_Production', 'Age']], dummy_species], axis=1)
    
    formula = 'Has_Production ~ Age + ' + ' + '.join(dummy_species.columns)
    try:
        model = smf.logit(formula, data=model_data).fit()
        print('\nEffect of Species on Production Probability (controlling for Age):')
        print(model.summary().tables[1])
    except:
        print('\nCould not fit species logistic model due to separation issues.')
        
        # Alternative approach: Chi-square test for species
        contingency = pd.crosstab(df['Espèce'], df['Has_Production'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f'\nChi-square test for Species effect on production:')
        print(f'Chi2 = {chi2:.2f}, p-value = {p:.4f}, df = {dof}')

# Analyzing average weight by species and age
def analyze_average_weight(df):
    print('\n--- AVERAGE TRUFFLE WEIGHT ANALYSIS ---')
    
    # Filter rows with actual production
    prod_data = df[df['Poids moyen (g)'] > 0].copy()
    
    # Group by species and age
    avg_weight = prod_data.groupby(['Espèce', 'Age'])['Poids moyen (g)'].mean().reset_index()
    
    # Plot average weight by species and age
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Espèce', y='Poids moyen (g)', data=prod_data)
    plt.title('Average Truffle Weight by Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('average_weight_by_species.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age', y='Poids moyen (g)', data=prod_data)
    plt.title('Average Truffle Weight by Tree Age')
    plt.tight_layout()
    plt.savefig('average_weight_by_age.png', dpi=300, bbox_inches='tight')
    
    # Statistical tests
    print('\nDescriptive statistics of average truffle weight by species:')
    print(prod_data.groupby('Espèce')['Poids moyen (g)'].describe())
    
    # ANOVA for species effect
    species_list = prod_data['Espèce'].unique()
    if len(species_list) >= 2:
        try:
            anova_species = stats.f_oneway(*[prod_data[prod_data['Espèce'] == species]['Poids moyen (g)'] 
                                         for species in species_list if sum(prod_data['Espèce'] == species) > 0])
            print('\nANOVA for Species effect on average truffle weight:')
            print(f'F = {anova_species.statistic:.2f}, p-value = {anova_species.pvalue:.4f}')
        except:
            print('\nCould not perform ANOVA for species (insufficient data in some groups)')
    
    # Regression for age effect
    if len(prod_data) > 5:
        model = sm.OLS.from_formula('Q("Poids moyen (g)") ~ Age', data=prod_data).fit()
        print('\nLinear regression of Age on average truffle weight:')
        print(model.summary().tables[1])

# Analyzing truffle quantity by species and age
def analyze_quantity(df):
    print('\n--- TRUFFLE QUANTITY ANALYSIS ---')
    
    # Filter rows with actual production
    prod_data = df[df['Nombre de truffe'] > 0].copy()
    
    # Create normalized quantity (per 100 plants)
    prod_data['Truffles_per_100_plants'] = 100 * prod_data['Nombre de truffe'] / prod_data['Plants']
    
    # Plot truffle quantity by species and age
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Espèce', y='Truffles_per_100_plants', data=prod_data)
    plt.title('Truffle Quantity per 100 Plants by Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('quantity_by_species.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age', y='Truffles_per_100_plants', data=prod_data)
    plt.title('Truffle Quantity per 100 Plants by Tree Age')
    plt.tight_layout()
    plt.savefig('quantity_by_age.png', dpi=300, bbox_inches='tight')
    
    # Statistical tests
    print('\nDescriptive statistics of truffle quantity per 100 plants by species:')
    print(prod_data.groupby('Espèce')['Truffles_per_100_plants'].describe())
    
    # Regression for age effect
    if len(prod_data) > 5:
        model = sm.OLS.from_formula('Truffles_per_100_plants ~ Age', data=prod_data).fit()
        print('\nLinear regression of Age on truffle quantity per 100 plants:')
        print(model.summary().tables[1])

# Analyzing total weight by species and age
def analyze_total_weight(df):
    print('\n--- TOTAL WEIGHT ANALYSIS ---')
    
    # Filter rows with actual production
    prod_data = df[df['Poids produit (g)'] > 0].copy()
    
    # Create normalized weight (per 100 plants)
    prod_data['Weight_per_100_plants'] = 100 * prod_data['Poids produit (g)'] / prod_data['Plants']
    
    # Plot by species and age
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Espèce', y='Weight_per_100_plants', data=prod_data)
    plt.title('Total Truffle Weight per 100 Plants by Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('total_weight_by_species.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age', y='Weight_per_100_plants', data=prod_data)
    plt.title('Total Truffle Weight per 100 Plants by Tree Age')
    plt.tight_layout()
    plt.savefig('total_weight_by_age.png', dpi=300, bbox_inches='tight')
    
    # Statistical tests
    print('\nDescriptive statistics of total weight per 100 plants by species:')
    print(prod_data.groupby('Espèce')['Weight_per_100_plants'].describe())
    
    # Regression for age effect
    if len(prod_data) > 5:
        model = sm.OLS.from_formula('Weight_per_100_plants ~ Age', data=prod_data).fit()
        print('\nLinear regression of Age on total truffle weight per 100 plants:')
        print(model.summary().tables[1])
        
    # Multiple regression with age and species
    if len(prod_data) > 10:
        # Create dummy variables for species
        dummy_species = pd.get_dummies(prod_data['Espèce'], prefix='Species', drop_first=True)
        model_data = pd.concat([prod_data[['Weight_per_100_plants', 'Age']], dummy_species], axis=1)
        
        formula = 'Weight_per_100_plants ~ Age + ' + ' + '.join(dummy_species.columns)
        try:
            model = sm.OLS.from_formula(formula, data=model_data).fit()
            print('\nMultiple regression of Age and Species on total weight per 100 plants:')
            print(model.summary().tables[1])
        except:
            print('\nCould not fit multiple regression model due to collinearity or insufficient data.')

# Create correlation plot for overall relationships
def correlation_analysis(df):
    print('\n--- CORRELATION ANALYSIS ---')
    
    # Select only numeric columns with actual values
    numeric_cols = ['Age', 'Plants', 'Plants Productifs', 'Taux de Productivité (%)', 
                   'Poids produit (g)', 'Nombre de truffe', 'Poids moyen (g)', 
                   'Production au plant (g)']
    
    # Filter rows with at least some production
    prod_data = df[df['Plants Productifs'] > 0].copy()
    
    # Calculate correlation matrix
    corr_matrix = prod_data[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5)
    plt.title('Correlation Matrix of Production Metrics')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    
    # Print key correlations
    print('\nKey correlations with Age:')
    age_corr = corr_matrix['Age'].sort_values(ascending=False)
    print(age_corr)

# Examining the impact of the planting season (penalty)
def analyze_planting_penalty(df):
    print('\n--- PLANTING SEASON PENALTY ANALYSIS ---')
    
    # Classify rows into penalty or not
    df['Has_Penalty'] = df['Pénalité'].notna() & (df['Pénalité'] != 'None')
    
    # Group by penalty status and age
    penalty_data = df.groupby(['Has_Penalty', 'Age']).agg({
        'Taux de Productivité (%)': 'mean', 
        'Production au plant (g)': 'mean'
    }).reset_index()
    
    # Plot productivity rate by penalty status and age
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Age', y='Taux de Productivité (%)', hue='Has_Penalty', 
                 markers=True, data=penalty_data)
    plt.title('Productivity Rate by Planting Season Penalty and Age')
    plt.xlabel('Tree Age (years)')
    plt.ylabel('Productivity Rate (%)')
    plt.legend(title='Planted Outside Optimal Season')
    plt.tight_layout()
    plt.savefig('planting_penalty_productivity.png', dpi=300, bbox_inches='tight')
    
    # Statistical test
    prod_data = df[df['Plants Productifs'] > 0].copy()
    if len(prod_data) > 10:
        try:
            model = sm.OLS.from_formula('Q("Taux de Productivité (%)") ~ Age + Has_Penalty', data=prod_data).fit()
            print('\nEffect of planting season penalty on productivity rate (controlling for age):')
            print(model.summary().tables[1])
        except:
            print('\nInsufficient data to analyze planting penalty effect statistically.')

# Predicting next year's production based on current trends
def predict_next_season(df):
    print('\n--- NEXT SEASON PREDICTION ---')
    
    # Get current season year
    current_season = df['Season_Year'].max()
    next_season = current_season + 1
    
    print(f'\nPredicting production for {next_season}-{next_season+1} season')
    
    # Group data by parcels and species
    current_trees = df[df['Season_Year'] == current_season].copy()
    
    # Increment age by 1 for next season
    current_trees['Next_Age'] = current_trees['Age'] + 1
    
    # Fit simple age-based model for productivity rate
    if len(df[df['Plants Productifs'] > 0]) > 5:
        try:
            # Model for productivity rate by age
            prod_model = smf.ols('Q("Taux de Productivité (%)") ~ Age', 
                                 data=df[df['Plants Productifs'] > 0]).fit()
            
            # Predict productivity rate for next season
            current_trees['Pred_Productivity'] = prod_model.predict(
                pd.DataFrame({'Age': current_trees['Next_Age']}))
            
            # Clip predicted values to be non-negative
            current_trees['Pred_Productivity'] = current_trees['Pred_Productivity'].clip(lower=0)
            
            # Calculate predicted productive plants
            current_trees['Pred_Productive_Plants'] = np.round(
                current_trees['Plants'] * current_trees['Pred_Productivity'] / 100)
            
            # Group by species and age to summarize predictions
            predictions = current_trees.groupby(['Parcelle', 'Espèce', 'Next_Age'])[
                'Plants', 'Pred_Productivity', 'Pred_Productive_Plants'
            ].sum().reset_index()
            
            # Filter to only show predictions with some productive plants
            predictions = predictions[predictions['Pred_Productive_Plants'] > 0]
            
            # Sort by parcelle and species
            predictions = predictions.sort_values(['Parcelle', 'Espèce', 'Next_Age'])
            
            # Save predictions
            predictions.to_csv('next_season_predictions.csv', index=False)
            print('\nPredictions for next season saved to "next_season_predictions.csv"')
            
            # Show summary by species
            species_summary = predictions.groupby('Espèce')[
                'Plants', 'Pred_Productive_Plants'
            ].sum().reset_index()
            species_summary['Pred_Productivity_Rate'] = 100 * species_summary['Pred_Productive_Plants'] / species_summary['Plants']
            species_summary = species_summary.sort_values('Pred_Productivity_Rate', ascending=False)
            
            print('\nPredicted productivity by species for next season:')
            print(species_summary)
            
        except Exception as e:
            print(f'\nError in prediction model: {e}')
            print('Insufficient data for reliable predictions.')
    else:
        print('\nInsufficient data for prediction model.')

# Main analysis
def main():
    # Prepare data
    data = prepare_data(production_data)
    
    # Run analyses
    analyze_precocity(data)
    analyze_average_weight(data)
    analyze_quantity(data)
    analyze_total_weight(data)
    correlation_analysis(data)
    analyze_planting_penalty(data)
    predict_next_season(data)
    
    print('\nAnalysis complete! Check the generated PNG files for visualizations.')

if __name__ == "__main__":
    main()
