import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

processed_data = 'Task3/ProcessedFraudTest.csv'
training_data = pd.read_csv(processed_data)

smote = SMOTE(random_state=42)

X = training_data.drop(columns=['is_fraud', 'gender', 'job'])
y = training_data['is_fraud']

X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)


    
import joblib
# saving the trained model
# joblib.dump(model, 'fraud_detection_model.pkl')
# print("its saved")

loaded_model = joblib.load('Task3/fraud_detection_model.pkl')
print("Model loaded successfully!")

# Use the loaded model for predictions
y_pred = loaded_model.predict(X_test)
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

model_path = 'Task3/fraud_detection_model.pkl'
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from '{model_path}'!")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Please ensure the model is saved.")
    exit()

# Testing Interface
while True:
    print("\nEnter transaction details below (or type 'exit' to quit):")
    amt = input("Transaction amount: ")
    if amt.lower() == 'exit':
        break

    try:
        amt = float(amt)
        lat = float(input("Customer latitude: "))
        long = float(input("Customer longitude: "))
        city_pop = float(input("City population: "))
        unix_time = float(input("Unix time of transaction: "))
        merch_lat = float(input("Merchant latitude: "))
        merch_long = float(input("Merchant longitude: "))
        age = float(input("Customer age: "))
        hour = int(input("Hour of transaction (0-23): "))
        day = int(input("Day of transaction: "))
        weekday = int(input("Day of the week (0=Sunday, 6=Saturday): "))
        month = int(input("Month of transaction (1-12): "))
        category = input("Category (e.g., entertainment, food_dining): ")

        # Process input data
        category_encoded = {f'category_{cat}': 0 for cat in [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos', 
            'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 
            'personal_care', 'shopping_net', 'shopping_pos', 'travel'
        ]}
        if f'category_{category}' in category_encoded:
            category_encoded[f'category_{category}'] = 1
        else:
            print(f"Warning: '{category}' is not a recognized category. Defaulting to 'unknown'.")

        # Create input dataframe
        input_data = pd.DataFrame([{
            'amt': amt,
            'lat': lat,
            'long': long,
            'city_pop': city_pop,
            'unix_time': unix_time,
            'merch_lat': merch_lat,
            'merch_long': merch_long,
            'hour': hour,
            'day': day,
            'weekday': weekday,
            'month': month,
            'age': age,
            'merchant_encoded': 0,  # Default value since this feature is no longer used
            **category_encoded
        }])

        # Predict the class (fraud or not fraud)
        prediction = model.predict(input_data)

        print("\n--- Prediction ---")
        print("Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction")

    except ValueError as e:
        print(f"Invalid input: {e}. Please enter the details again.")