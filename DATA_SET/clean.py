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

missing_summary = data_cleaned.isnull().sum()
print(data_cleaned.head(), missing_summary)

data_cleaned.to_csv(saving_path, index=False)