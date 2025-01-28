import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import lime

processed_data_path = 'Task5/processed.csv'

data = pd.read_csv(processed_data_path)

X = data.drop(columns=['Attrition'])
y = data['Attrition']
feature_names = X.columns
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame after scaling
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

logistic_model =  LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# Evaluate the logistic_regression_model
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,  # Pass only this, remove extra argument
    feature_names=feature_names.tolist(),  # Ensure feature names are a list
    class_names=['No Attrition', 'Attrition'], 
    mode='classification'
)

# Pick an instance to explain (e.g., first test sample)
i = 0  # Change this index to visualize different samples
exp = explainer.explain_instance(X_test.iloc[i], logistic_model.predict_proba)

# Save explanation to an HTML file
exp.save_to_file('lime_explanation_logistic_reg.html')

print("LIME explanation saved as 'lime_explanation.html'. Open it in your browser.")