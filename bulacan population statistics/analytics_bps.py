import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for output if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load data
df = pd.read_csv('bulacan_pop.csv')

# Set aesthetic style
sns.set_theme(style="whitegrid")

# Graph 1: Bar Chart of Population
plt.figure(figsize=(10, 6))
sns.barplot(x='Population', y='City_Municipality', data=df.sort_values('Population', ascending=False), palette='viridis')
plt.title('Top Populated Areas in Bulacan (2020)')
plt.savefig('output/population_bar.png')
plt.close()

# Graph 2: Scatter Plot (Land Area vs Population)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Land_Area_km2', y='Population', size='Density_per_km2', data=df, hue='City_Municipality', legend=False)
plt.title('Land Area vs Population Size')
plt.savefig('output/area_vs_pop_scatter.png')
plt.close()

# Graph 3: Box Plot of Population Density
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Density_per_km2'], color='skyblue')
plt.title('Distribution of Population Density in Bulacan LGUs')
plt.savefig('output/density_boxplot.png')
plt.close()

print("Analytics complete. Check the /output folder for graphs!")