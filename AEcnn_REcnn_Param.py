
"""
Created on Thu December 2026

@author: Majid GHORBANNEZHAD
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, LeakyReLU, Flatten,
    Reshape, MaxPooling2D, BatchNormalization,
    Conv2DTranspose
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim


# 1️ Initialisation

np.random.seed(42)
tf.random.set_seed(42)


# 2️ Chargement des données

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

H, W, C = 28, 28, 1
latent_dim = 40


# 3️ Modèle Autoencodeur

inputs = Input(shape=(H, W, C))

# ----- Encoder -----
x = Conv2D(32, (3,3), padding="same")(inputs)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(256)(x)
x = Dense(latent_dim)(x)
x = LeakyReLU()(x)

# ----- Decoder -----
x = Dense(7*7*64)(x)
x = Reshape((7,7,64))(x)

x = Conv2DTranspose(64, (3,3), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(32, (3,3), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy"
)

autoencoder.summary()


# 4️ Entraînement

history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)


# 5️ Courbes de loss

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction loss")
plt.legend()
plt.grid()
plt.show()


# 6️ Reconstruction

x_pred = autoencoder.predict(x_test)


# 7️ Métriques quantitatives


# MSE globale
mse_global = np.mean((x_test - x_pred) ** 2)
print("MSE globale :", mse_global)

# MSE par image
mse_per_image = np.mean((x_test - x_pred) ** 2, axis=(1,2,3))
print("MSE image 0 :", mse_per_image[0])

# PSNR
def compute_psnr(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

print("PSNR :", compute_psnr(x_test, x_pred))

# SSIM
ssim_vals = []
for i in range(len(x_test)):
    ssim_val = ssim(
        x_test[i].reshape(28,28),
        x_pred[i].reshape(28,28),
        data_range=1.0
    )
    ssim_vals.append(ssim_val)

print("SSIM moyenne :", np.mean(ssim_vals))


# 8️ Visualisation

n = 10
plt.figure(figsize=(20,4))

for i in range(n):
    # Image originale
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    ax.set_title("Original")
    ax.axis("off")

    # Reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(x_pred[i].reshape(28,28), cmap="gray")
    ax.set_title("Reconstr.")
    ax.axis("off")

plt.show()


# 9️ Carte d’erreur (exemple)

i = 0
error_map = np.abs(x_test[i] - x_pred[i])

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(x_test[i].reshape(28,28), cmap="gray")

plt.subplot(1,3,2)
plt.title("Reconstruction")
plt.imshow(x_pred[i].reshape(28,28), cmap="gray")

plt.subplot(1,3,3)
plt.title("Erreur |x - x̂|")
plt.imshow(error_map.reshape(28,28), cmap="hot")

plt.show()
