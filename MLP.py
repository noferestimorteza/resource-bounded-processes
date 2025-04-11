import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
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
num_classes = len(label_encoder.classes_)
y_onehot = to_categorical(y_encoded)

# 4. Flatten the sequential data for MLP
n_samples, n_events, n_features = X_seq.shape
X_flat = X_seq.reshape(n_samples, n_events * n_features)

# 5. Balance Classes - Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y_encoded)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 7. Feature Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. MLP Model Architecture
def build_mlp_model(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 9. Class Weighting
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# 10. Create and train MLP
mlp_model = build_mlp_model(X_train_scaled.shape[1], num_classes)
mlp_model.summary()

# 11. Training with Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('best_mlp_model.h5', save_best_only=True)
]

history = mlp_model.fit(
    X_train_scaled, 
    to_categorical(y_train),
    validation_split=0.2,
    epochs=300,
    batch_size=128,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# 12. Evaluation
y_pred = mlp_model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nMLP Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=label_encoder.classes_,
                          zero_division=0))

# 13. Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()