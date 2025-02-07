import pandas as pd
import numpy as np

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
