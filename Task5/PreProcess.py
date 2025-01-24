import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE # for oversampling the minority class
from sklearn.preprocessing import LabelEncoder

file_path = 'Task5/HR-Employee-Attrition.csv'
saving_path = 'Task5/processed.csv'
data_df = pd.read_csv(file_path)

# print(data_df.info())
# print(data_df.describe())
# print(data_df.isnull().sum())

# EDA
# print(data_df["Attrition"].value_counts())
# data_df["Attrition"].value_counts().plot(kind="bar", color=["blue", "orange"])
# plt.title("Attrition Class Distribution")
# plt.xlabel("Attrition")
# plt.ylabel("Count")
# plt.show()

# # This analysis helps uncover trends, such as whether people with lower income or certain job roles are more likely to leave.
# # Univariate Analysis: Bar Chart for Gender
# data_df["Gender"].value_counts().plot(kind="bar", color=["purple", "green"])
# plt.title("Gender Distribution")
# plt.show()

# # Bivariate Analysis: Boxplot of MonthlyIncome vs Attrition
# sns.boxplot(x="Attrition", y="MonthlyIncome", data=data_df)
# plt.title("Monthly Income vs Attrition")
# plt.show()


# Cleaning and processing the data
label_encoder = LabelEncoder()
# converting categorical values to numerical
processed_data_df=data_df
processed_data_df['JobRole_Encoded'] = label_encoder.fit_transform(processed_data_df['JobRole'])
processed_data_df = processed_data_df.drop(columns=['JobRole'])
processed_data_df['Department_Encoded'] = label_encoder.fit_transform(processed_data_df['Department'])
processed_data_df = processed_data_df.drop(columns=['Department'])
# processed_data_df = pd.get_dummies(data_df, columns=["JobRole", "Department"], drop_first=True)
# conerting binary categories in binary
processed_data_df["Gender"] = processed_data_df["Gender"].replace({"Male": 0, "Female": 1})
processed_data_df['Attrition'] = processed_data_df['Attrition'].replace({"Yes": 1, "No": 0})

# print(processed_data_df[[col for col in processed_data_df.columns if "JobRole" in col or "Department" in col]].head())

scaler = MinMaxScaler()
numeric_columns = ["Age", "MonthlyIncome", "YearsAtCompany"]
processed_data_df[numeric_columns] = scaler.fit_transform(processed_data_df[numeric_columns])


processed_data_df['BusinessTravel_Encoded'] = label_encoder.fit_transform(processed_data_df['BusinessTravel'])
processed_data_df = processed_data_df.drop(columns=['BusinessTravel'])

categorical_columns = ['EducationField', 'MaritalStatus', 'Over18', 'OverTime']
for col in categorical_columns:
    processed_data_df[col] = label_encoder.fit_transform(processed_data_df[col])

print(processed_data_df.dtypes)


# Feature filtering
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = processed_data_df.drop('Attrition', axis=1)  # Feature matrix
y = processed_data_df['Attrition']  # Target variable

# print(X.dtypes)

chi2_selector = SelectKBest(chi2, k='all')  # Use 'all' to get all scores
chi2_selector.fit(X, y)
chi2_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': chi2_selector.scores_
})
chi2_scores = chi2_scores.sort_values(by='Score', ascending=False)
print(chi2_scores)

# droping irrelevant columns
irrelevant_columns = ['EmployeeNumber', 'BusinessTravel_Encoded', 'EmployeeCount', 'StandardHours', 'Over18', 'HourlyRate', 'PercentSalaryHike', 'Education', 'Gender', 'EducationField']
processed_data_df = processed_data_df.drop(columns=irrelevant_columns)



processed_data_df.to_csv(saving_path, index=False)