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

# Add correlation matrix between all observable parameters
plt.figure(figsize=(14, 12))
numeric_cols = ['Age', 'Plants', 'Plants Productifs', 'Taux de Productivité (%)', 
                'Poids produit (g)', 'Nombre de truffe', 'Poids moyen (g)', 'Production au plant (g)']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, mask=mask)
plt.title('Correlation Matrix Between Observable Parameters')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Add distribution analysis for key metrics
# 11. Distribution of Productivity Rate by Species
plt.figure(figsize=(14, 10))
# Filter to only include species with some productivity
species_with_prod = df[df['Taux de Productivité (%)'] > 0]['Espèce'].unique()
prod_data = df[df['Espèce'].isin(species_with_prod)]
sns.boxplot(x='Espèce', y='Taux de Productivité (%)', data=prod_data)
plt.title('Distribution of Productivity Rate by Species')
plt.xlabel('Species')
plt.ylabel('Productivity Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('productivity_distribution_by_species.png')

# 12. Violin plots for truffle weight distribution by species
plt.figure(figsize=(14, 10))
# Filter to only include species with truffles
truffle_species_data = df[(df['Poids moyen (g)'] > 0)]
sns.violinplot(x='Espèce', y='Poids moyen (g)', data=truffle_species_data)
plt.title('Truffle Weight Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Average Truffle Weight (g)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weight_distribution_by_species.png')

# 13. Pair plot for key metrics
plt.figure(figsize=(16, 16))
# Select a subset of the data to make the pair plot readable
sample_df = df[df['Taux de Productivité (%)'] > 0].sample(min(50, len(df[df['Taux de Productivité (%)'] > 0])))
sns.pairplot(sample_df[['Age', 'Taux de Productivité (%)', 'Poids moyen (g)', 'Production au plant (g)', 'Espèce']], 
             hue='Espèce', height=2.5)
plt.suptitle('Relationships Between Key Metrics', y=1.02)
plt.tight_layout()
plt.savefig('key_metrics_pairplot.png')

# 14. Bubble chart: Age vs. Productivity vs. Average Weight
plt.figure(figsize=(14, 10))
productive_data = df[df['Taux de Productivité (%)'] > 0]
species_colors = {species: color for species, color in zip(
    productive_data['Espèce'].unique(), 
    sns.color_palette('husl', len(productive_data['Espèce'].unique()))
)}

for species in productive_data['Espèce'].unique():
    species_data = productive_data[productive_data['Espèce'] == species]
    plt.scatter(
        species_data['Age'], 
        species_data['Taux de Productivité (%)'],
        s=species_data['Poids moyen (g)'] * 5,  # Size based on average weight
        alpha=0.7,
        color=species_colors[species],
        label=species
    )

plt.title('Age vs. Productivity Rate (size represents average truffle weight)')
plt.xlabel('Age (years)')
plt.ylabel('Productivity Rate (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_productivity_weight_bubble.png')

# 15. Heatmap of Production by Age and Species
plt.figure(figsize=(15, 10))
species_age_pivot = df.pivot_table(
    values='Production au plant (g)', 
    index='Espèce', 
    columns='Age', 
    aggfunc='mean')
sns.heatmap(species_age_pivot, annot=True, cmap='viridis', fmt='.1f', linewidths=.5)
plt.title('Average Production per Plant (g) by Species and Age')
plt.tight_layout()
plt.savefig('production_by_age_species_heatmap.png')

# 16. Regression plot for Age vs Productivity
plt.figure(figsize=(12, 8))
sns.regplot(x='Age', y='Taux de Productivité (%)', 
            data=df[df['Taux de Productivité (%)'] > 0], 
            scatter_kws={'alpha':0.5}, 
            line_kws={'color':'red'})
plt.title('Regression Analysis: Age vs Productivity Rate')
plt.xlabel('Age (years)')
plt.ylabel('Productivity Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_productivity_regression.png')

# Create additional insights

# 17. Production by Age Distribution
plt.figure(figsize=(10, 6))
age_prod = df.groupby('Age')['Poids produit (g)'].sum()
plt.pie(age_prod, labels=age_prod.index, autopct='%1.1f%%', startangle=90, 
       colors=sns.color_palette("YlGnBu", len(age_prod)))
plt.title('Distribution of Total Production by Age')
plt.tight_layout()
plt.savefig('production_by_age_pie.png')

# 18. Species Performance Comparison with enhanced metrics
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
species_perf['Avg Weight per Truffle (g)'] = (species_perf['Poids produit (g)'] / species_perf['Nombre de truffe']).fillna(0).round(2)
species_perf['Revenue Potential Index'] = (species_perf['Productivity Rate (%)'] * species_perf['Avg Weight per Truffle (g)']).round(2)

# Filter for species with at least one productive plant
species_perf_filtered = species_perf[species_perf['Plants Productifs'] > 0].sort_values('Revenue Potential Index', ascending=False)

# Plot as a horizontal bar chart with enhanced formatting
sns.barplot(y='Espèce', x='Revenue Potential Index', data=species_perf_filtered, palette='viridis')
plt.title('Species Revenue Potential Index (Productivity × Avg Weight)')
plt.xlabel('Revenue Potential Index')
plt.tight_layout()
plt.savefig('species_revenue_potential.png')

# 19. Radar Chart for Species Comparison
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Prepare data for radar chart - select top 5 species with highest productivity
top5_species = species_perf.sort_values('Productivity Rate (%)', ascending=False).head(5)

# Select metrics for comparison
metrics = ['Productivity Rate (%)', 'Avg Truffles per Productive Plant', 'Avg Weight per Truffle (g)']

# Normalize the data for radar chart
normalized_data = pd.DataFrame()
for metric in metrics:
    normalized_data[metric] = (top5_species[metric] - top5_species[metric].min()) / \
                           (top5_species[metric].max() - top5_species[metric].min())

# Create radar chart
num_metrics = len(metrics)
angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Add background grid and labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
plt.xticks(angles[:-1], metrics)

# Plot each species
colors = plt.cm.viridis(np.linspace(0, 1, len(top5_species)))
for i, species in enumerate(top5_species['Espèce']):
    values = normalized_data.iloc[i].values.tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, color=colors[i], linewidth=2, label=species)
    ax.fill(angles, values, color=colors[i], alpha=0.25)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Top 5 Species Performance Comparison', size=15)
plt.tight_layout()
plt.savefig('species_radar_chart.png')

# 20. Cluster Analysis of Parcels Based on Performance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Prepare data for clustering
parcel_metrics = df.groupby('Parcelle').agg({
    'Taux de Productivité (%)': 'mean',
    'Poids moyen (g)': 'mean',
    'Production au plant (g)': 'mean'
}).reset_index()

# Remove rows with NaN values
parcel_metrics = parcel_metrics.dropna()

# Standardize the data for clustering
X = parcel_metrics[['Taux de Productivité (%)', 'Poids moyen (g)', 'Production au plant (g)']]
X_scaled = StandardScaler().fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(1, min(6, len(parcel_metrics)))
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.savefig('cluster_elbow_method.png')

# Perform clustering with the optimal k (typically where the elbow occurs)
optimal_k = 3  # Set a default, this would normally be determined from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
parcel_metrics['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Taux de Productivité (%)',
    y='Production au plant (g)',
    hue='Cluster',
    size='Poids moyen (g)',
    sizes=(50, 500),
    data=parcel_metrics,
    palette='viridis'
)

# Add parcel labels to the plot
for i, row in parcel_metrics.iterrows():
    plt.annotate(row['Parcelle'], (row['Taux de Productivité (%)'], row['Production au plant (g)']), 
                 fontsize=9, alpha=0.7)

plt.title('Cluster Analysis of Parcels Based on Performance Metrics')
plt.tight_layout()
plt.savefig('parcel_cluster_analysis.png')

# 21. Species Productivity Rate (%) - original visualization with enhanced styling
plt.figure(figsize=(14, 8))
sns.barplot(y='Espèce', x='Productivity Rate (%)', data=species_perf_filtered, palette='plasma')
plt.title('Species Productivity Rate (%)', fontsize=16)
plt.xlabel('Productivity Rate (%)', fontsize=12)
plt.ylabel('Species', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
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
