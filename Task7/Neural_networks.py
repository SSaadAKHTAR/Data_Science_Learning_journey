import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lime
from lime.lime_tabular import LimeTabularExplainer


# importing scaled_data
scaled_data_path = 'Task7/CleanedScaledData.csv'
data = pd.read_csv(scaled_data_path)

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# using smote to over sample the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

nn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", max_iter=2000, random_state=42)
nn_model.fit(X_train_resampled, y_train_resampled)
y_pred = nn_model.predict(X_test)
y_pred_proba = nn_model.predict_proba(X_test)[:, 1]


print("Classification report of neural networks; n", classification_report(y_test, y_pred))
print("Neural Network AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

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
exp = explainer.explain_instance(X_test.iloc[i], nn_model.predict_proba)

# Save explanation to an HTML file
exp.save_to_file('lime_explanation_nn_model.html')

print("LIME explanation saved as 'lime_explanation.html'. Open it in your browser.")