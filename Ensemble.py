import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set the matplotlib backend to avoid GUI issues
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# 1. Data Loading and Preparation
df = pd.read_csv('final_events_with_categories.csv')

# 2. Feature Engineering
def prepare_features(df):
    # Convert all features to numeric
    X = df.drop('Category', axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Create event sequences
    n_events = 5
    event_features = []
    for i in range(1, n_events+1):
        prefix = f'event{i}.'
        event_cols = [col for col in X.columns if col.startswith(prefix)]
        event_features.append(X[event_cols].values)
    
    # Stack events to create (samples, events, features)
    X_seq = np.stack(event_features, axis=1)
    return X_seq

X_seq = prepare_features(df)
y = df['Category']

# 3. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Flatten the sequential data
n_samples, n_events, n_features = X_seq.shape
X_flat = X_seq.reshape(n_samples, n_events * n_features)

# 5. Balance Classes - Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y_encoded)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 7. Stacked Ensemble Model
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=100, random_state=42))
]

stacked_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 8. Train the model
print("Training Stacked Ensemble...")
stacked_model.fit(X_train, y_train)

# 9. Evaluation
y_pred = stacked_model.predict(X_test)

print("\nStacked Ensemble Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=label_encoder.classes_,
                          zero_division=0))

# 10. Save visualizations to files instead of showing
print("Generating visualizations...")

# Feature Importance (using the first base estimator - RandomForest)
if hasattr(stacked_model.estimators_[0], 'feature_importances_'):
    importance = stacked_model.estimators_[0].feature_importances_
    sorted_idx = np.argsort(importance)[-20:]  # Top 20 features
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
    plt.xlabel('Feature Importance (from RandomForest base estimator)')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("Visualizations saved to files:")
print("- feature_importance.png")
print("- confusion_matrix.png")