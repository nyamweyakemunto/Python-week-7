import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris

data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['species'] = data.target
species_mapping = {i: name for i, name in enumerate(data.target_names)}
iris_df['species'] = iris_df['species'].map(species_mapping)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Explore the structure of the dataset
print("\nDataset info:")
print(iris_df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_df.isnull().sum())

# Clean the dataset (if necessary)
# No missing values in this dataset, so no cleaning is required

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(iris_df.describe())

# Perform groupings and compute mean for each group
grouped_means = iris_df.groupby('species').mean()
print("\nMean values of numerical columns grouped by species:")
print(grouped_means)

# Identify patterns or interesting findings
# Example: Sepal length is longest in Iris-virginica species

# Task 3: Data Visualization
# Line chart (not applicable for this dataset; creating a generic line plot)
plt.figure(figsize=(8, 5))
iris_df.groupby('species').mean().plot(kind='line', marker='o')
plt.title('Mean Feature Values by Species')
plt.ylabel('Mean Value')
plt.xlabel('Species')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=iris_df, x='species', y='sepal length (cm)', ci=None)
plt.title('Average Sepal Length per Species')
plt.ylabel('Sepal Length (cm)')
plt.xlabel('Species')
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['sepal length (cm)'], kde=True, bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
