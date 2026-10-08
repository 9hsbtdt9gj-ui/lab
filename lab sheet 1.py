# ==============================================================================
# COER UNIVERSITY - LAB SHEET-01
# Lab Environment Setup, Python Libraries Installation & Dataset Exploration
# ==============================================================================

# ------------------------------------------------------------------------------
# PROGRAM 1: Verify Installed Python Version
# ------------------------------------------------------------------------------
import sys

print("--- Program 1 ---")
print("Python Version:", sys.version)


# ------------------------------------------------------------------------------
# PROGRAM 2: Launch Jupyter Notebook (Terminal Command)
# ------------------------------------------------------------------------------
# Command to execute in Terminal / Command Prompt:
# pip install notebook
# jupyter notebook


# ------------------------------------------------------------------------------
# PROGRAM 3: Verify NumPy Installation
# ------------------------------------------------------------------------------
import numpy as np

print("\n--- Program 3 ---")
print("NumPy Version:", np.__version__)


# ------------------------------------------------------------------------------
# PROGRAM 4: Verify Pandas Installation
# ------------------------------------------------------------------------------
import pandas as pd

print("\n--- Program 4 ---")
print("Pandas Version:", pd.__version__)


# ------------------------------------------------------------------------------
# PROGRAM 5: Matplotlib Simple Line Plot
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

x_vals = [1, 2, 3, 4, 5]
y_vals = [2, 4, 6, 8, 10]

plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, marker='o')
plt.title("Program 5: Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 6: Seaborn Statistical Plot
# ------------------------------------------------------------------------------
import seaborn as sns

data_samples = [12, 15, 12, 18, 20, 25, 30, 22, 24]

plt.figure(figsize=(6, 4))
sns.boxplot(data=data_samples)
plt.title("Program 6: Basic Statistical Box Plot")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 7: Verify Scikit-learn Installation
# ------------------------------------------------------------------------------
import sklearn

print("\n--- Program 7 ---")
print("Scikit-learn Version:", sklearn.__version__)


# ------------------------------------------------------------------------------
# PROGRAM 8: First Program in Jupyter Notebook
# ------------------------------------------------------------------------------
student_name = "MCA Student"
print("\n--- Program 8 ---")
print(f"Hello, {student_name}! Welcome to Data Science Lab.")


# ------------------------------------------------------------------------------
# PROGRAM 9: Script Execution Example
# ------------------------------------------------------------------------------
def execute_script():
    return "Program 9: Script execution completed successfully."

print("\n--- Program 9 ---")
print(execute_script())


# ------------------------------------------------------------------------------
# PROGRAM 10: Virtual Environment Setup Commands
# ------------------------------------------------------------------------------
# Commands to run in Terminal / Command Prompt:
# python -m venv myenv
# myenv\Scripts\activate          # Windows
# source myenv/bin/activate       # macOS/Linux
# pip install numpy pandas matplotlib seaborn scikit-learn


# ------------------------------------------------------------------------------
# PROGRAM 32 (Prerequisite for Datasets): Load Scikit-learn Dataset
# ------------------------------------------------------------------------------
from sklearn.datasets import load_iris

# Loading built-in Iris dataset for demonstration of data manipulations
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target


# ------------------------------------------------------------------------------
# PROGRAM 11: Load CSV Dataset using Pandas
# ------------------------------------------------------------------------------
# df = pd.read_csv('dataset.csv')  # Standard CSV loading command
print("\n--- Program 11 ---")
print("Dataset loaded successfully.")


# ------------------------------------------------------------------------------
# PROGRAM 12: Display First Five Records
# ------------------------------------------------------------------------------
print("\n--- Program 12: First 5 Records ---")
print(df.head())


# ------------------------------------------------------------------------------
# PROGRAM 13: Display Last Five Records
# ------------------------------------------------------------------------------
print("\n--- Program 13: Last 5 Records ---")
print(df.tail())


# ------------------------------------------------------------------------------
# PROGRAM 14: Total Rows and Columns
# ------------------------------------------------------------------------------
total_rows, total_cols = df.shape
print("\n--- Program 14 ---")
print(f"Total Rows: {total_rows}, Total Columns: {total_cols}")


# ------------------------------------------------------------------------------
# PROGRAM 15: Display Names of All Columns
# ------------------------------------------------------------------------------
print("\n--- Program 15 ---")
print("Column Names:", df.columns.tolist())


# ------------------------------------------------------------------------------
# PROGRAM 16: Data Types of All Columns
# ------------------------------------------------------------------------------
print("\n--- Program 16 ---")
print(df.dtypes)


# ------------------------------------------------------------------------------
# PROGRAM 17: Descriptive Statistics
# ------------------------------------------------------------------------------
print("\n--- Program 17: Descriptive Statistics ---")
print(df.describe())


# ------------------------------------------------------------------------------
# PROGRAM 18: Complete Information of Dataset
# ------------------------------------------------------------------------------
print("\n--- Program 18: Dataset Info ---")
df.info()


# ------------------------------------------------------------------------------
# PROGRAM 19: Identify Missing Values
# ------------------------------------------------------------------------------
print("\n--- Program 19: Missing Values Matrix (Sample) ---")
print(df.isnull().head())


# ------------------------------------------------------------------------------
# PROGRAM 20: Count Missing Values Per Column
# ------------------------------------------------------------------------------
print("\n--- Program 20: Total Missing Values Per Column ---")
print(df.isnull().sum())


# ------------------------------------------------------------------------------
# PROGRAM 21: Unique Values in Selected Column
# ------------------------------------------------------------------------------
print("\n--- Program 21 ---")
print("Unique species values:", df['species'].unique())


# ------------------------------------------------------------------------------
# PROGRAM 22: Frequency of Unique Values
# ------------------------------------------------------------------------------
print("\n--- Program 22 ---")
print("Frequency of species values:\n", df['species'].value_counts())


# ------------------------------------------------------------------------------
# PROGRAM 23: Rename Columns
# ------------------------------------------------------------------------------
df = df.rename(columns={'sepal length (cm)': 'sepal_length'})
print("\n--- Program 23 ---")
print("Updated Columns:", df.columns.tolist())


# ------------------------------------------------------------------------------
# PROGRAM 24: Select Rows and Columns using loc[]
# ------------------------------------------------------------------------------
loc_subset = df.loc[0:4, ['sepal_length', 'species']]
print("\n--- Program 24: Selection via loc[] ---")
print(loc_subset)


# ------------------------------------------------------------------------------
# PROGRAM 25: Select Rows and Columns using iloc[]
# ------------------------------------------------------------------------------
iloc_subset = df.iloc[0:5, 0:2]
print("\n--- Program 25: Selection via iloc[] ---")
print(iloc_subset)


# ------------------------------------------------------------------------------
# PROGRAM 26: Filter Records Based on Condition
# ------------------------------------------------------------------------------
filtered_data = df[df['sepal_length'] > 6.0]
print("\n--- Program 26: Filtered Records (sepal_length > 6.0) ---")
print(filtered_data.head())


# ------------------------------------------------------------------------------
# PROGRAM 27: Sort Dataset
# ------------------------------------------------------------------------------
sorted_data = df.sort_values(by='sepal_length', ascending=False)
print("\n--- Program 27: Sorted Dataset (Descending Sepal Length) ---")
print(sorted_data.head())


# ------------------------------------------------------------------------------
# PROGRAM 28: Add New Column
# ------------------------------------------------------------------------------
df['is_large_sepal'] = df['sepal_length'] > 5.5
print("\n--- Program 28: New Column Added ---")
print(df[['sepal_length', 'is_large_sepal']].head())


# ------------------------------------------------------------------------------
# PROGRAM 29: Delete Column
# ------------------------------------------------------------------------------
df = df.drop(columns=['is_large_sepal'])
print("\n--- Program 29: Column Deleted ---")
print("Columns after deletion:", df.columns.tolist())


# ------------------------------------------------------------------------------
# PROGRAM 30: Remove Duplicate Records
# ------------------------------------------------------------------------------
df = df.drop_duplicates()
print("\n--- Program 30 ---")
print("Shape after removing duplicates:", df.shape)


# ------------------------------------------------------------------------------
# PROGRAM 31: Save Modified Dataset to CSV
# ------------------------------------------------------------------------------
df.to_csv('modified_dataset.csv', index=False)
print("\n--- Program 31 ---")
print("Dataset saved to 'modified_dataset.csv'.")


# ------------------------------------------------------------------------------
# PROGRAM 33: Histogram for Numerical Feature
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df['sepal_length'], bins=10, color='skyblue', edgecolor='black')
plt.title("Program 33: Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 34: Scatter Plot Between Two Variables
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(df['sepal_length'], df['sepal width (cm)'], color='purple')
plt.title("Program 34: Scatter Plot (Sepal Length vs Sepal Width)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 35: Correlation Matrix and Heatmap
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Program 35: Correlation Matrix Heatmap")
plt.show()
