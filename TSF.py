import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sktime.classification.interval_based import TimeSeriesForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('final_events_with_categories.csv')

# 2. Feature Engineering (Keep 3D time series structure)
def prepare_features(df):
    X = df.drop('Category', axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    n_events = 5
    event_features = []
    for i in range(1, n_events + 1):
        prefix = f'event{i}.'
        event_cols = [col for col in X.columns if col.startswith(prefix)]
        event_features.append(X[event_cols].values)
    
    # Return 3D array [samples, events, features]
    return np.stack(event_features, axis=1)

X_seq = prepare_features(df)  # Shape: [n_samples, n_events, n_features]
y = df['Category']

# 3. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Balance Classes (Oversampling for tree-based models)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(
    X_seq.reshape(len(X_seq), -1),  # Temporarily flatten for oversampling
    y_encoded
)
X_resampled = X_resampled.reshape(-1, X_seq.shape[1], X_seq.shape[2])  # Restore 3D

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
from sktime.classification.interval_based import CanonicalIntervalForest  # Multivariate version
# 6. Time Series Forest Model
tsf_model = CanonicalIntervalForest(
    n_estimators=100,
    min_interval=3,
    n_jobs=-1,
    random_state=42
)



# 7. Train
print("Training Time Series Forest...")
tsf_model.fit(X_train, y_train)

# 8. Evaluate
y_pred = tsf_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=label_encoder.classes_,
                          zero_division=0))

# 9. Feature Importance (Mean Decrease Impurity)
if hasattr(tsf_model, "feature_importances_"):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(tsf_model.feature_importances_)), tsf_model.feature_importances_)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Time Series Forest Feature Importance")
    plt.savefig('tsf_feature_importance.png')
    plt.close()

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.savefig('tsf_confusion_matrix.png')
plt.close()