import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore # for outliers

dataset_path = 'Task7/diabetes.csv'

data = pd.read_csv(dataset_path)

# print(data.info())
# print(data.describe())
# print(data.isnull().sum())

# Since there are no NAN values or missing values but there are 0 in columns where 0 is not possible so we will replace the 0 with NAN
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)

# print(data.isnull().sum())
data.fillna(data.median(), inplace=True)
print(data.isnull().sum())

# Visualization

# Histograms
# data.hist(figsize=(12, 10), bins=20)
# plt.show()

# # Boxplots to check outliers
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data)
# plt.xticks(rotation=90)
# plt.show()


# # Correlation Analysis
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# sns.pairplot(data, hue="Outcome")
# plt.show()


# # Feature-Target Analysis
# sns.countplot(x=data["Outcome"])
# plt.title("Diabetes Cases Distribution")
# plt.show()

# for column in data.columns[:-1]:  # Exclude 'Outcome'
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=data["Outcome"], y=data[column])
#     plt.title(f"{column} vs Diabetes Outcome")
#     plt.show()




