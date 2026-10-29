from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout

def build_cnn_mri(input_shape=(32, 32, 32, 1), num_classes=3):
    model = Sequential([
        Input(shape=input_shape),
        Conv3D(16, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        Dropout(0.2),
        Conv3D(32, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
