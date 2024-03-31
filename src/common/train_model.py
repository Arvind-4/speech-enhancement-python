import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from model_unet import unet
from data_tools import scaled_in, scaled_ou


def training(
    path_save_spectrogram,
    weights_path,
    name_model,
    training_from_scratch,
    epochs,
    batch_size,
):
    X_in = np.load(path_save_spectrogram + "noisy_voice_amp_db" + ".npy")
    X_ou = np.load(path_save_spectrogram + "voice_amp_db" + ".npy")
    X_ou = X_in - X_ou

    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    print(X_in.shape)
    print(X_ou.shape)
    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    X_in = X_in[:, :, :]
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    X_ou = X_ou[:, :, :]
    X_ou = X_ou.reshape(X_ou.shape[0], X_ou.shape[1], X_ou.shape[2], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_in, X_ou, test_size=0.10, random_state=42
    )

    if training_from_scratch:
        generator_nn = unet()
    else:
        generator_nn = unet(pretrained_weights=weights_path + name_model + ".h5")

    checkpoint = ModelCheckpoint(
        weights_path + "/model_best.h5",
        verbose=1,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
    )

    generator_nn.summary()
    history = generator_nn.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=(X_test, y_test),
    )

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.yscale("log")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
