"""
Digit Recognition using CNN (MNIST-style)
-----------------------------------------
A simple CNN-based digit recognition pipeline:
- Loads and preprocesses data
- Builds and trains a CNN
- Evaluates model with confusion matrix
- Saves model and training history
- Runs predictions on test data

Usage:
    python digit_recognition.py --train data/Train.csv --test data/test.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
def load_and_preprocess_data(train_path):
    """
    Load training data and preprocess it for CNN.
    Args:
        train_path (str): Path to training CSV file.
    Returns:
        tuple: X_train, X_val, y_train, y_val
    """
    train_data = pd.read_csv(train_path)

    # Features and labels
    X = train_data.iloc[:, 1:].values / 255.0
    X = X.reshape(-1, 28, 28, 1)

    y = to_categorical(train_data.iloc[:, 0], num_classes=10)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------
# 2. Model Building (CNN)
# ---------------------------
def build_cnn_model():
    """Build and compile a CNN model for digit recognition."""
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
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """Train CNN model with early stopping and LR scheduling."""
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ---------------------------
# 4. Visualization
# ---------------------------
def plot_training_history(history, save_path=None):
    """Plot training history and optionally save as image."""
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

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(model, X_val, y_val, save_path=None):
    """Plot confusion matrix for validation data."""
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------
# 5. Testing on Unseen Data
# ---------------------------
def test_and_visualize(model, test_path):
    """
    Run model on test dataset and visualize first few predictions.
    Args:
        model: Trained Keras model.
        test_path (str): Path to test CSV file.
    Returns:
        np.ndarray: Predicted labels for test data.
    """
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
    parser = argparse.ArgumentParser(description="Digit Recognition using CNN")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--output", type=str, default="models", help="Output directory for model/results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(args.train)

    # Build model
    model = build_cnn_model()
    model.summary()

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    # Plots
    plot_training_history(history, save_path=os.path.join(args.output, "training_history.png"))
    plot_confusion_matrix(model, X_val, y_val, save_path=os.path.join(args.output, "confusion_matrix.png"))

    # Save model
    model.save(os.path.join(args.output, "digit_recognition_model.h5"))
    print("✅ Model saved!")

    # Save training history
    pd.DataFrame(history.history).to_csv(os.path.join(args.output, "training_history.csv"), index=False)

    # Test predictions
    predicted_labels = test_and_visualize(model, args.test)
    pd.DataFrame(predicted_labels, columns=["Label"]).to_csv(os.path.join(args.output, "test_predictions.csv"), index=False)
    print("✅ Predictions saved!")

#train_model.py --train data/Train.csv --test data/test.csv --epochs 10 --batch_size 64 --output results
