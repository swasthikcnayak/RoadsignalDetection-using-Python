from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def create_model(X_train, Y_train, X_valid, Y_valid):
    model = models.Sequential([
        Conv2D(8, (5, 5), activation='relu', padding="same", input_shape=X_train[0].shape),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    Model = model.fit(X_train, Y_train, batch_size=500, epochs=15, verbose=1, validation_data=(X_valid, Y_valid))
    return model
