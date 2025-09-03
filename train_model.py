import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
def load_and_preprocess_data(train_path):
    train_data = pd.read_csv(train_path)

    # Features and labels
    X = train_data.iloc[:, 1:].values / 255.0  # normalize
    X = X.reshape(-1, 28, 28, 1)  # reshape to 28x28x1

    y = to_categorical(train_data.iloc[:, 0], num_classes=10)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------
# 2. Model Building (CNN)
# ---------------------------
def build_cnn_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ---------------------------
# 3. Training
# ---------------------------
def train_model(model, X_train, y_train, X_val, y_val):
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True),  # stop early if no improvement
        ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=5,              
        batch_size=128,       
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ---------------------------
# 4. Visualization
# ---------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss")

    plt.show()


def plot_confusion_matrix(model, X_val, y_val):
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


# ---------------------------
# 5. Testing on Unseen Data
# ---------------------------
def test_and_visualize(model, test_path):
    test_data = pd.read_csv(test_path)
    X_test = test_data.values / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Show first 5 predictions
    for i in range(5):
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {predicted_labels[i]}")
        plt.axis('off')
        plt.show()

    return predicted_labels


# ---------------------------
# 6. Main Script
# ---------------------------
if __name__ == "__main__":
    train_path = r"C:\Users\hp\Desktop\PROJECTS\0 SEMS\SEM 8 DigitRecognitionUsingML\Train.csv"
    test_path = r"C:\Users\hp\Desktop\PROJECTS\0 SEMS\SEM 8 DigitRecognitionUsingML\test.csv"

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(train_path)

    # Build model
    model = build_cnn_model()
    model.summary()

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    # Plots
    plot_training_history(history)
    plot_confusion_matrix(model, X_val, y_val)

    # Save model
    model.save(r"C:\Users\hp\Desktop\PROJECTS\0 SEMS\SEM 8 DigitRecognitionUsingML\digit_recognition_model.h5")
    print("Model saved as digit_recognition_model.h5")

    # Test predictions
    predicted_labels = test_and_visualize(model, test_path)
