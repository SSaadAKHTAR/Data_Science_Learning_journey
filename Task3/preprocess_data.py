import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


data_path = 'Task3/fraudTest.csv'
saving_path = 'Task3/ProcessedFraudTest.csv'
data=pd.read_csv(data_path)
print(data.info())
print(data.describe())
print(data.head())
print(data.isnull().sum())

data['trans_date_trans_time']=pd.to_datetime(data['trans_date_trans_time'])
data['hour'] = data['trans_date_trans_time'].dt.hour
data['day'] = data['trans_date_trans_time'].dt.day
data['weekday'] = data['trans_date_trans_time'].dt.weekday
data['month'] = data['trans_date_trans_time'].dt.month


# data = pd.get_dummies(data, columns=['category', 'gender', 'job'], drop_first=True)

data['age'] = pd.to_datetime('2020-06-21') - pd.to_datetime(data['dob'])
data['age'] = data['age'].dt.days // 365

data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time'], inplace=True)


category_encoded = pd.get_dummies(data['category'], prefix='category')
label_encoder = LabelEncoder()
data['merchant_encoded'] = label_encoder.fit_transform(data['merchant'])

data = pd.concat([data, category_encoded], axis=1).drop(['merchant', 'category'], axis=1)

numerical_columns = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age']
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print(data.isnull().sum())

data.to_csv(saving_path, index=False)