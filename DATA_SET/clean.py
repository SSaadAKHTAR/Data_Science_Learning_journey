import pandas as pd

file_path = "DATA_SET/airnb.csv"
saving_path = "DATA_SET/cleaned_dataset/airnb_clean.csv"

airnb_data = pd.read_csv(file_path)
# print(airnb_data.info())
# print(" ")
# print(airnb_data.head())

# cleaning
airnb_data['Price(in dollar)'] = pd.to_numeric(airnb_data["Price(in dollar)"].str.replace(',','').str.strip())
airnb_data['Offer price(in dollar)'] = pd.to_numeric(airnb_data['Offer price(in dollar)'].str.replace(',', '').str.strip())
# print(airnb_data.info())
# print(" ")
# print(airnb_data.head())

airnb_data[['Rating', 'Number of Reviews']] = airnb_data['Review and rating'].str.extract(r'([\d\.]+)\s+\((\d+)\)')
airnb_data['Rating'] = pd.to_numeric(airnb_data['Rating'])
airnb_data['Number of Reviews'] = pd.to_numeric(airnb_data['Number of Reviews'])
# print(airnb_data.info())
# print(" ")
# print(airnb_data.head())

airnb_data['Number of bed'] = airnb_data['Number of bed'].str.extract(r'(\d+)').astype(float)
# print(airnb_data.info())
# print(" ")
# print(airnb_data.head())

airnb_data[['Start Date', 'End Date']] = airnb_data['Date'].str.extract(r'(\w+ \d+) - (\d+)')
airnb_data['Start Date'] = pd.to_datetime(airnb_data['Start Date'] + " 2024", format='%b %d %Y', errors='coerce')
airnb_data['End Date'] = pd.to_datetime(airnb_data['End Date'].astype(str) + " 2024", format='%d %Y', errors='coerce')
# print(airnb_data.info())
# print(" ")
# print(airnb_data.head())

data_cleaned = airnb_data.drop(columns=['Review and rating', 'Date'])

# missing_summary = data_cleaned.isnull().sum()
# print(data_cleaned.head(), missing_summary)

# data_cleaned.to_csv(saving_path, index=False)



import matplotlib.pyplot as plt
import seaborn as sns

# Handle Missing Data
# Impute 'Offer price(in dollar)' with the mean value
data_cleaned['Offer price(in dollar)'].fillna(data_cleaned['Offer price(in dollar)'].mean(), inplace=True)

# Drop rows where 'Rating', 'Number of Reviews', or 'Start Date' is missing
data_cleaned.dropna(subset=['Rating', 'Number of Reviews', 'Start Date'], inplace=True)

# Calculate the length of stay
data_cleaned['Length of Stay'] = (data_cleaned['End Date'] - data_cleaned['Start Date']).dt.days

# Perform EDA: Summary statistics
summary_stats = data_cleaned.describe()
print(summary_stats)
data_cleaned.to_csv(saving_path, index=False)
missing_summary = data_cleaned.isnull().sum()
print(data_cleaned.head(), missing_summary)


# Visualize price distribution
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['Price(in dollar)'], kde=True, bins=30, color='blue')
plt.title('Price Distribution', fontsize=16)
plt.xlabel('Price (in dollars)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y')
plt.show()

# Visualize relationship between price and rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Price(in dollar)', data=data_cleaned, hue='Number of Reviews', palette='viridis')
plt.title('Price vs Rating', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Price (in dollars)', fontsize=12)
# plt.colorbar(label='Number of Reviews')
plt.grid(axis='both')
plt.show()

# Visualize length of stay distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_cleaned, x='Length of Stay', color='lightgreen')
plt.title('Length of Stay Distribution', fontsize=16)
plt.xlabel('Length of Stay (days)', fontsize=12)
plt.grid(axis='y')
plt.show()

# Display the cleaned data summary statistics
print(summary_stats)
# Correcting the scatter plot visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Rating',
    y='Price(in dollar)',
    data=data_cleaned,
    size='Number of Reviews',
    sizes=(10, 200),
    color='blue',
    alpha=0.7
)
plt.title('Price vs Rating', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Price (in dollars)', fontsize=12)
plt.legend(title='Number of Reviews', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(axis='both')
plt.show()





# Bar Chart: Count of Properties by Number of Beds
plt.figure(figsize=(10, 6))
sns.countplot(data=data_cleaned, x='Number of bed', palette='pastel')
plt.title('Count of Properties by Number of Beds', fontsize=16)
plt.xlabel('Number of Beds', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Explanation: 
# This chart shows how many properties are available for each bed count. 
# Peaks indicate the most common number of beds offered in the dataset.


# Correlation Heatmap for Numeric Features
numeric_features = ['Price(in dollar)', 'Offer price(in dollar)', 'Rating', 'Number of Reviews', 'Length of Stay']
correlation_matrix = data_cleaned[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
plt.show()

# Explanation:
# The heatmap visually represents the strength and direction of correlations between numeric variables.
# Correlations closer to 1 or -1 indicate stronger linear relationships.