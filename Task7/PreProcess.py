import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore # for outliers
from sklearn.preprocessing import StandardScaler

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



# Feature scaling which is only nessesary for models like SVM, neural networks, logistic regression, knn and not nessesary for models like Decision trees and Gradient boosting.
# Select numeric features
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

scaled_data = data
scaler = StandardScaler()
scaled_data[features] = scaler.fit_transform(data[features])

# feature selection
# we can also drop the feature if it is highly corelated with another feature by using corelation matrix
# plt.figure(figsize=(10,8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show() # according to this there are no highly corelated feature so on the bases of correlation we are not going to drop any feature

# saving the data and scaled data unscaled data
scaled_data.to_csv('Task7/CleanedScaledData.csv', index=False)
data.to_csv('Task7/CleanedData.csv', index=False)

