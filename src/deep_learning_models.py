# src/deep_learning_models.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int, encoding_dim: int = 16):
    
    i = keras.Input(shape=(input_dim,))
    e = layers.Dense(encoding_dim, activation="relu")(i)
    d = layers.Dense(input_dim, activation="linear")(e)
    m = keras.Model(i, d)
    m.compile(optimizer="adam", loss="mse")
    return m

def train_autoencoder(X, encoding_dim=16, epochs=20, batch_size=256, validation_split=0.1, verbose=1):
    X = np.asarray(X, dtype=np.float32)
    m = build_autoencoder(X.shape[1], encoding_dim)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    h = m.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=cb, verbose=verbose)
    return m, h

def reconstruction_errors(model, X):
    X = np.asarray(X, dtype=np.float32)
    preds = model.predict(X, verbose=0)
    return np.mean((X - preds) ** 2, axis=1)
