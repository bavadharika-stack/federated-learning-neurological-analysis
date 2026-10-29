import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from models.cnn_model_mri import build_cnn_mri
from models.rnn_model_eeg import build_rnn_eeg

# ==== 1. Parameters ====
num_classes = 3
num_samples_per_class = 50  # Increase for better learning (total = 150)
input_shape_mri = (32, 32, 32, 1)
input_shape_eeg = (100, 32)
class_names = ['Healthy', 'Alzheimer', 'Tumor']

# ==== 2. Generate Class-Specific Synthetic Data ====
def make_class_data(class_id, shape):
    np.random.seed(class_id)
    base = np.random.rand(*shape) * (class_id + 1)  # Vary by class
    noise = np.random.normal(0, 0.05, shape)
    return (base + noise).astype(np.float32)

X_mri = np.concatenate([
    make_class_data(0, (num_samples_per_class, *input_shape_mri)),
    make_class_data(1, (num_samples_per_class, *input_shape_mri)),
    make_class_data(2, (num_samples_per_class, *input_shape_mri))
])

X_eeg = np.concatenate([
    make_class_data(0, (num_samples_per_class, *input_shape_eeg)),
    make_class_data(1, (num_samples_per_class, *input_shape_eeg)),
    make_class_data(2, (num_samples_per_class, *input_shape_eeg))
])

y = np.array([0]*num_samples_per_class + [1]*num_samples_per_class + [2]*num_samples_per_class)
y_cat = to_categorical(y, num_classes=num_classes)

# ==== 3. Train-Test Split ====
X_train_mri, X_test_mri, y_train_mri, y_test_mri = train_test_split(X_mri, y_cat, test_size=0.3, random_state=42)
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg, y_cat, test_size=0.3, random_state=42)

# ==== 4. Build & Train Models ====
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# MRI CNN Model
mri_model = build_cnn_mri(input_shape=input_shape_mri, num_classes=num_classes)
print("🧠 Training CNN MRI model...")
mri_model.fit(X_train_mri, y_train_mri, epochs=15, batch_size=8, verbose=1, callbacks=[early_stop])

# EEG RNN Model
eeg_model = build_rnn_eeg(input_shape=input_shape_eeg, num_classes=num_classes)
print("🧠 Training RNN EEG model...")
eeg_model.fit(X_train_eeg, y_train_eeg, epochs=15, batch_size=8, verbose=1, callbacks=[early_stop])

# ==== 5. Predictions ====
pred_mri = np.argmax(mri_model.predict(X_test_mri), axis=1)
true_mri = np.argmax(y_test_mri, axis=1)

pred_eeg = np.argmax(eeg_model.predict(X_test_eeg), axis=1)
true_eeg = np.argmax(y_test_eeg, axis=1)

# ==== 6. Evaluation ====
print("\n📊 MRI Model Evaluation:")
print("Accuracy:", accuracy_score(true_mri, pred_mri))
print("Classification Report:\n", classification_report(true_mri, pred_mri, target_names=class_names))

print("\n📊 EEG Model Evaluation:")
print("Accuracy:", accuracy_score(true_eeg, pred_eeg))
print("Classification Report:\n", classification_report(true_eeg, pred_eeg, target_names=class_names))
