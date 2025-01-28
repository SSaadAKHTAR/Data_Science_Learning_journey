import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import lime

# Load and process data
processed_data_path = 'Task5/processed.csv'
data = pd.read_csv(processed_data_path)

X = data.drop(columns=['Attrition'])
y = data['Attrition']

feature_names = X.columns

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame after scaling
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# # Explain model predictions with SHAP
# explainer = shap.TreeExplainer(rf_model)
# shap_values = explainer.shap_values(X_test)

# # Debugging shapes
# print("SHAP values shape (class 1):", shap_values[1].shape)
# print("X_test shape:", X_test.shape)

# # Fix summary plot
# if shap_values[1].shape[1] == X_test.shape[1]:
#     shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
# else:
#     print("Mismatch in SHAP values and X_test dimensions!")

# # SHAP force plot for a specific observation (e.g., index 0)
# shap.force_plot(
#     explainer.expected_value[1],
#     shap_values[1][0],
#     X_test.iloc[0].values,
#     feature_names=feature_names
# )



# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,  # Pass only this, remove extra argument
    feature_names=feature_names.tolist(),  # Ensure feature names are a list
    class_names=['No Attrition', 'Attrition'], 
    mode='classification'
)

# Pick an instance to explain (e.g., first test sample)
i = 1  # Change this index to visualize different samples
exp = explainer.explain_instance(X_test.iloc[i], rf_model.predict_proba)

# Save explanation to an HTML file
exp.save_to_file('lime_explanation_rf.html')

print("LIME explanation saved as 'lime_explanation.html'. Open it in your browser.")