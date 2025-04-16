# Truffle Production Analysis Script
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Setup
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the data with correct separator
df = pd.read_csv("export-bilan.csv", sep=";", encoding="utf-8")

# Fix numeric columns with comma as decimal separator - converting all at once before any analysis
numeric_columns = ['Taux de Productivité (%)', 'Poids produit (g)', 'Nombre de truffe', 'Poids moyen (g)', 'Production au plant (g)']
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Basic data exploration
print("\n===== DATA OVERVIEW =====\n")
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nParcels: {df['Parcelle'].unique().tolist()}")
print(f"\nSpecies: {df['Espèce'].unique().tolist()}")
print(f"\nAge range: {df['Age'].min()} to {df['Age'].max()} years")
print("\n===== SUMMARY STATISTICS =====\n")
print(df.describe())

# No need to convert again - already converted above

# 1. Production by Parcel
plt.figure()
total_by_parcel = df.groupby('Parcelle')['Poids produit (g)'].sum().sort_values(ascending=False)
sns.barplot(x=total_by_parcel.index, y=total_by_parcel.values)
plt.title('Total Truffle Weight by Parcel')
plt.xlabel('Parcel')
plt.ylabel('Total Weight (g)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('truffle_weight_by_parcel.png')

# 2. Production by Species
plt.figure()
total_by_species = df.groupby('Espèce')['Poids produit (g)'].sum().sort_values(ascending=False)
sns.barplot(x=total_by_species.index, y=total_by_species.values)
plt.title('Total Truffle Weight by Species')
plt.xlabel('Species')
plt.ylabel('Total Weight (g)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('truffle_weight_by_species.png')

# 3. Productivity Rate by Age and Species
plt.figure(figsize=(14, 8))
# Filter to only include species with some productivity
productive_species = df[df['Taux de Productivité (%)'] > 0]['Espèce'].unique()
productivity_data = df[df['Espèce'].isin(productive_species)]
productivity_by_age_species = productivity_data.groupby(['Age', 'Espèce'])['Taux de Productivité (%)'].mean().reset_index()
sns.barplot(x='Age', y='Taux de Productivité (%)', hue='Espèce', data=productivity_by_age_species)
plt.title('Average Productivity Rate by Age and Species')
plt.xlabel('Age (years)')
plt.ylabel('Productivity Rate (%)')
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('productivity_by_age_species.png')

# 4. Average Truffle Weight by Species
plt.figure()
# Filter to only include species with truffles
truffle_species = df[df['Poids moyen (g)'] > 0]['Espèce'].unique()
weight_data = df[df['Espèce'].isin(truffle_species)]
avg_weight_by_species = weight_data.groupby('Espèce')['Poids moyen (g)'].mean().sort_values(ascending=False)
sns.barplot(x=avg_weight_by_species.index, y=avg_weight_by_species.values)
plt.title('Average Truffle Weight by Species')
plt.xlabel('Species')
plt.ylabel('Average Weight (g)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_truffle_weight_by_species.png')

# 5. Production per Plant by Age
plt.figure()
# Filter to only include productive ages
productive_ages = df[df['Production au plant (g)'] > 0]['Age'].unique()
production_data = df[df['Age'].isin(productive_ages)]
prod_per_plant_by_age = production_data.groupby('Age')['Production au plant (g)'].mean()
sns.lineplot(x=prod_per_plant_by_age.index, y=prod_per_plant_by_age.values, marker='o', linewidth=2)
plt.title('Average Production per Plant by Age')
plt.xlabel('Age (years)')
plt.ylabel('Production per Plant (g)')
plt.grid(True)
plt.tight_layout()
plt.savefig('production_per_plant_by_age.png')

# 6. Advanced Analysis: Age vs Productivity by Parcel as Heatmap
plt.figure(figsize=(15, 10))
age_parcel_pivot = df.pivot_table(
    values='Taux de Productivité (%)', 
    index='Parcelle', 
    columns='Age', 
    aggfunc='mean')
sns.heatmap(age_parcel_pivot, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5)
plt.title('Productivity Rate (%) by Parcel and Age')
plt.tight_layout()
plt.savefig('productivity_heatmap.png')

# 7. Top Producers - Parcels with Highest Productivity
top_parcels = df.groupby('Parcelle')['Taux de Productivité (%)'].mean().sort_values(ascending=False).head(5)
plt.figure()
sns.barplot(x=top_parcels.index, y=top_parcels.values)
plt.title('Top 5 Parcels by Average Productivity Rate')
plt.xlabel('Parcel')
plt.ylabel('Average Productivity Rate (%)')
plt.tight_layout()
plt.savefig('top_parcels_by_productivity.png')

# 8. Top Producers - Species with Highest Productivity
top_species = df.groupby('Espèce')['Taux de Productivité (%)'].mean().sort_values(ascending=False).head(5)
plt.figure()
sns.barplot(x=top_species.index, y=top_species.values)
plt.title('Top 5 Species by Average Productivity Rate')
plt.xlabel('Species')
plt.ylabel('Average Productivity Rate (%)')
plt.tight_layout()
plt.savefig('top_species_by_productivity.png')

# 9. Number of Plants vs Productive Plants by Parcel
plants_data = df.groupby('Parcelle').agg({
    'Plants': 'sum',
    'Plants Productifs': 'sum'
}).reset_index()

plt.figure(figsize=(14, 8))
x = np.arange(len(plants_data['Parcelle']))
width = 0.35

plt.bar(x - width/2, plants_data['Plants'], width, label='Total Plants')
plt.bar(x + width/2, plants_data['Plants Productifs'], width, label='Productive Plants')

plt.xlabel('Parcel')
plt.ylabel('Number of Plants')
plt.title('Total Plants vs Productive Plants by Parcel')
plt.xticks(x, plants_data['Parcelle'])
plt.legend()
plt.tight_layout()
plt.savefig('plants_vs_productive_plants.png')

# 10. Scatter plot of Age vs Production per Plant by Species
plt.figure(figsize=(12, 8))
# Filter to only include productive plants
productive_plants = df[df['Production au plant (g)'] > 0]
sns.scatterplot(data=productive_plants, x='Age', y='Production au plant (g)', hue='Espèce', size='Plants', sizes=(20, 200), alpha=0.7)
plt.title('Age vs Production per Plant by Species')
plt.xlabel('Age (years)')
plt.ylabel('Production per Plant (g)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('age_vs_production_scatter.png')

# Create additional insights

# 11. Production by Age Distribution
plt.figure(figsize=(10, 6))
age_prod = df.groupby('Age')['Poids produit (g)'].sum()
plt.pie(age_prod, labels=age_prod.index, autopct='%1.1f%%', startangle=90, 
       colors=sns.color_palette("YlGnBu", len(age_prod)))
plt.title('Distribution of Total Production by Age')
plt.tight_layout()
plt.savefig('production_by_age_pie.png')

# 12. Species Performance Comparison
plt.figure(figsize=(14, 8))
species_perf = df.groupby('Espèce').agg({
    'Plants': 'sum',
    'Plants Productifs': 'sum',
    'Poids produit (g)': 'sum',
    'Nombre de truffe': 'sum'
}).reset_index()

# Calculate derived metrics
species_perf['Productivity Rate (%)'] = (species_perf['Plants Productifs'] / species_perf['Plants'] * 100).round(2)
species_perf['Avg Truffles per Productive Plant'] = (species_perf['Nombre de truffe'] / species_perf['Plants Productifs']).fillna(0).round(2)

# Filter for species with at least one productive plant
species_perf_filtered = species_perf[species_perf['Plants Productifs'] > 0].sort_values('Productivity Rate (%)', ascending=False)

# Plot as a horizontal bar chart
sns.barplot(y='Espèce', x='Productivity Rate (%)', data=species_perf_filtered, palette='viridis')
plt.title('Species Productivity Rate (%)')
plt.tight_layout()
plt.savefig('species_productivity_rate.png')

# Show plots
plt.show()

print("\n===== ANALYSIS INSIGHTS =====\n")
try:
    print("1. Most productive parcel: ", total_by_parcel.index[0], "with", round(total_by_parcel.values[0], 2), "g")
    print("2. Most productive species: ", total_by_species.index[0], "with", round(total_by_species.values[0], 2), "g")
    print("3. Highest productivity rate age: ", productivity_by_age_species.sort_values('Taux de Productivité (%)', ascending=False).iloc[0]['Age'], "years")
    print("4. Species with largest truffles: ", avg_weight_by_species.index[0], "with average", round(avg_weight_by_species.values[0], 2), "g")
    print("5. Age with highest production per plant: ", prod_per_plant_by_age.idxmax(), "years with", round(prod_per_plant_by_age.max(), 2), "g per plant")
except (IndexError, KeyError) as e:
    print("Some statistics could not be calculated due to insufficient data.")
