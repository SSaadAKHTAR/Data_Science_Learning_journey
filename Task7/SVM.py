import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import lime
from lime.lime_tabular import LimeTabularExplainer

# importing scaled_data
scaled_data_path = '/content/Data_Science_Learning_journey/Task7/CleanedScaledData.csv'
data = pd.read_csv(scaled_data_path)

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# using smote to over sample the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

SVm_model = SVC(kernel='rbf', probability=True, random_state=42)
SVm_model.fit(X_train_resampled, y_train_resampled)
ypred = SVm_model.predict(X_test)
y_prob_svm = SVm_model.predict_proba(X_test)[:, 1]  # Probabilities for AUC-ROC

# Evaluating My SVM model 
print("SVM Classification Report: \n", classification_report(y_test,ypred))
print("SVM AUC-ROC Score:", roc_auc_score(y_test, y_prob_svm))


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
exp = explainer.explain_instance(X_test.iloc[i], SVm_model.predict_proba)

# Save explanation to an HTML file
exp.save_to_file('lime_explanation_SVM_model.html')

print("LIME explanation saved as 'lime_explanation.html'. Open it in your browser.")


# param_grid = {
#     'C': [0.1, 1, 10], 
#     'kernel': ['linear', 'rbf', 'poly'], 
#     'gamma': [0.01, 0.1, 1]
# }

# grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid.fit(X_train_resampled, y_train_resampled)

# print("Best Parameters:", grid.best_params_)



# SVM Classification Report: 
#                precision    recall  f1-score   support

#            0       0.82      0.75      0.78       100
#            1       0.60      0.69      0.64        54

#     accuracy                           0.73       154
#    macro avg       0.71      0.72      0.71       154
# weighted avg       0.74      0.73      0.73       154

# SVM AUC-ROC Score: 0.19018518518518518
