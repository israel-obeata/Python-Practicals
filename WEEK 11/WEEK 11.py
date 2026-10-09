import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. LOAD ----------
# Iris.csv is in the main folder, so if it is not found here, look one folder up
try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    df = pd.read_csv('../Iris.csv')

# Keep the measurement columns and the species (this leaves out the Id column)
measurements = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = df[measurements + ['Species']]

# ---------- 2. INSPECT ----------
print('--- Shape ---')
print(df.shape)
print('\n--- Columns ---')
print(df.columns)
print('\n--- First 5 rows ---')
print(df.head())
print('\n--- Info ---')
df.info()
print('\n--- Summary statistics ---')
print(df.describe())

# ---------- 3. CLEAN ----------
print('\n--- Missing values per column ---')
print(df.isnull().sum())

print('\nRows before removing duplicates:', len(df))
df = df.drop_duplicates()
print('Rows after removing duplicates:', len(df))

# ---------- 4. EXPLORE ----------
print('\n--- Species in the data ---')
print(df['Species'].unique())

# Filter with two conditions
big = df[(df['PetalLengthCm'] > 4) & (df['SepalLengthCm'] > 6)]
print('\nFlowers with petal length > 4 AND sepal length > 6:', len(big))

# Sort
print('\n--- 5 longest petals ---')
print(df.sort_values('PetalLengthCm', ascending=False).head())

# GroupBy: average of each measurement for each species
for col in measurements:
    print('\n--- Average', col, 'per species ---')
    print(df.groupby('Species')[col].mean())

# Descriptive statistics of one column
print('\nPetal length -> mean:', round(df['PetalLengthCm'].mean(), 2),
      '| median:', round(df['PetalLengthCm'].median(), 2),
      '| std:', round(df['PetalLengthCm'].std(), 2),
      '| min:', df['PetalLengthCm'].min(),
      '| max:', df['PetalLengthCm'].max())

# Correlation
print('\n--- Correlation ---')
print(df[measurements].corr())

# ---------- 5. VISUALIZE ----------
# Histogram: sepal length (the line marks the mean)
plt.hist(df['SepalLengthCm'], bins=15)
plt.axvline(df['SepalLengthCm'].mean())
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Number of Flowers')
plt.title('Distribution of Sepal Length')
plt.show()

# Histogram: petal length
plt.hist(df['PetalLengthCm'], bins=15)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Number of Flowers')
plt.title('Distribution of Petal Length')
plt.show()

# Scatter plot: petal length vs petal width, one colour per species
for name in df['Species'].unique():
    part = df[df['Species'] == name]
    plt.scatter(part['PetalLengthCm'], part['PetalWidthCm'], label=name)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()
plt.show()

# Bar chart: average petal length per species
df.groupby('Species')['PetalLengthCm'].mean().plot(kind='bar')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.title('Average Petal Length per Species')
plt.show()

# ---------- 6. CONCLUDE ----------
print('\n--- Conclusion ---')
print('Typical petal length (median):', round(df['PetalLengthCm'].median(), 2), 'cm')
print('Spread of petal length (std):', round(df['PetalLengthCm'].std(), 2), 'cm')
corr = df[['PetalLengthCm', 'PetalWidthCm']].corr().iloc[0, 1]
print('Strongest relationship: petal length and petal width, correlation =', round(corr, 2))
print('Average petal length per species:')
print(df.groupby('Species')['PetalLengthCm'].mean())
print('Recommendation: petal measurements separate the species best, so use them '
      'to tell the species apart.')
