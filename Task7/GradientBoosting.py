import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lime
from lime.lime_tabular import LimeTabularExplainer

# importing scaled_data
scaled_data_path = 'Task7/CleanedData.csv'
data = pd.read_csv(scaled_data_path)

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# using smote to over sample the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=42)
gb_model.fit(X_train_resampled, y_train_resampled)
y_pred = gb_model.predict(X_test)

print("Classification report: \n", classification_report(y_test, y_pred))
print("AUC-ROC score: \n", roc_auc_score(y_test,y_pred))

feature_names = X.columns
# lime
# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,  # Pass only this, remove extra argument
    feature_names=feature_names.tolist(),  # Ensure feature names are a list
    class_names=['No', 'yes'], 
    mode='classification'
)

# Pick an instance to explain (e.g., first test sample)
i = 0  # Change this index to visualize different samples
exp = explainer.explain_instance(X_test.iloc[i], gb_model.predict_proba)

# Save explanation to an HTML file
exp.save_to_file('lime_explanation_gb_model.html')

print("LIME explanation saved as 'lime_explanation.html'. Open it in your browser.")


# Classification report: 
#                precision    recall  f1-score   support

#            0       0.84      0.75      0.79       100
#            1       0.62      0.74      0.67        54

#     accuracy                           0.75       154
#    macro avg       0.73      0.75      0.73       154
# weighted avg       0.76      0.75      0.75       154

# AUC-ROC score: 
#  0.7453703703703703
