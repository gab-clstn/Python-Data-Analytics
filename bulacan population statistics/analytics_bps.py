import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('output', exist_ok=True)
df = pd.read_csv('dataset.csv')

# Graph 1: Bar Chart - 2020 Population per Region
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Region', y='2020', palette='magma')
plt.title('Philippine Population by Region (2020)')
plt.savefig('output/population_2020.png')
plt.close()

# Graph 2: Line Plot - Population Growth Trend (DevOps requirement for trend analysis)
years = ['2000', '2010', '2015', '2020']
trend_df = df.set_index('Region')[years].T
plt.figure(figsize=(10, 6))
trend_df.plot(marker='o')
plt.title('Population Movement (2000-2020)')
plt.xlabel('Census Year')
plt.ylabel('Population Count')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('output/growth_trend.png')
plt.close()

# Graph 3: Pie Chart - 2020 Distribution
plt.figure(figsize=(8, 8))
plt.pie(df['2020'], labels=df['Region'], autopct='%1.1f%%', startangle=140)
plt.title('Regional Population Share (2020)')
plt.savefig('output/population_share.png')
plt.close()

print("Analytics complete using Kaggle dataset. Files saved to /output")