import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# 1. Data Loading and Preparation
df = pd.read_csv('final_events_with_categories.csv')

# 2. Handle Class Imbalance
print("Class distribution before balancing:")
print(df['Category'].value_counts())

# 3. Feature Engineering
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

# 4. Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_onehot = to_categorical(y_encoded)

# 5. Balance Classes - Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_reshaped = X_seq.reshape(X_seq.shape[0], -1)  # Flatten for oversampling
X_resampled, y_resampled = ros.fit_resample(X_reshaped, y_encoded)
X_resampled = X_resampled.reshape(-1, X_seq.shape[1], X_seq.shape[2])  # Reshape back

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 7. Feature Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
n_samples, n_events, n_features = X_train.shape
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, n_events, n_features)
X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], n_events, n_features)

# 8. Enhanced Model Architecture
def build_enhanced_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # CNN Branch with multiple filter sizes
    conv1a = Conv1D(64, 2, activation='relu', padding='same')(inputs)
    conv1b = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    conv1c = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    conv_merged = Concatenate()([conv1a, conv1b, conv1c])
    
    pool1 = MaxPooling1D(pool_size=2)(conv_merged)
    conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool2)
    cnn_out = Flatten()(conv3)
    
    # BiLSTM Branch with attention
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    bilstm2 = Bidirectional(LSTM(64))(bilstm1)
    
    # Combined features
    merged = Concatenate()([cnn_out, bilstm2])
    
    # Enhanced dense layers
    dense1 = Dense(512, activation='relu')(merged)
    dropout1 = Dropout(0.6)(dense1)
    dense2 = Dense(256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    dense3 = Dense(128, activation='relu')(dropout2)
    
    # Output with class weighting
    outputs = Dense(num_classes, activation='softmax')(dense3)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 9. Train with Class Weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

model = build_enhanced_model((X_train_scaled.shape[1], X_train_scaled.shape[2]), num_classes)

# 10. Training with Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_enhanced_model.h5', save_best_only=True)
]

history = model.fit(
    X_train_scaled, 
    to_categorical(y_train),
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# 11. Evaluation
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nEnhanced Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=label_encoder.classes_,
                          zero_division=0))