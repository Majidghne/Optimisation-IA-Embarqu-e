
"""
Created on Thu December 2026

@author: Majid GHORBANNEZHAD
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 1️ Configuration & Chargement

np.random.seed(42)
tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

H, W, C = 28, 28, 1
latent_dim = 60


# 2️ Aplatissement

x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat  = x_test.reshape(len(x_test), -1)


# 3️ Autoencodeur Dense

inputs = Input(shape=(H * W * C,))
x = Dense(128, activation='relu')(inputs)
latent = Dense(latent_dim, activation='relu')(x)
x = Dense(128, activation='relu')(latent)
outputs = Dense(H * W * C, activation='sigmoid')(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(
    optimizer=Adam(1e-3),
    loss='binary_crossentropy'
)

print("\n--- Entraînement Autoencodeur ---")
start_time = time.perf_counter()

autoencoder.fit(
    x_train_flat, x_train_flat,
    epochs=15,
    batch_size=256,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)

# Synchronisation GPU
autoencoder.evaluate(x_test_flat, x_test_flat, verbose=0)

ae_train_time = time.perf_counter() - start_time
print(f" Temps entraînement Autoencodeur : {ae_train_time:.2f} s")


# 4️ Reconstruction

start_time = time.perf_counter()

x_train_recon_flat = autoencoder.predict(x_train_flat, batch_size=256)
x_test_recon_flat  = autoencoder.predict(x_test_flat, batch_size=256)

recon_time = time.perf_counter() - start_time

x_train_recon = x_train_recon_flat.reshape(-1, 28, 28, 1)
x_test_recon  = x_test_recon_flat.reshape(-1, 28, 28, 1)

print(f" Temps reconstruction (test) : {recon_time:.2f} s")
print(f" Temps / image              : {recon_time / len(x_test):.6f} s")


# 5️ Qualité reconstruction

def compute_psnr(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

mse_val = np.mean((x_test - x_test_recon) ** 2)
psnr_val = compute_psnr(x_test, x_test_recon)
ssim_val = np.mean([
    ssim(
        x_test[i].reshape(28,28),
        x_test_recon[i].reshape(28,28),
        data_range=1.0
    )
    for i in range(len(x_test))
])

print("\n[METRIQUES AUTOENCODEUR]")
print(f"MSE  : {mse_val:.4f}")
print(f"PSNR : {psnr_val:.2f} dB")
print(f"SSIM : {ssim_val:.4f}")

# ==========================================
# 6️ Classification
# ==========================================
x_train_combined = np.concatenate([x_train, x_train_recon], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

def build_classifier():
    model = tf.keras.Sequential([
        Flatten(input_shape=(H, W, C)), 
        Dense(60, activation='relu'), # Dense layer
        Dropout(0.25), # Dropout for regularization
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def train_and_time(model, x_tr, y_tr, x_te, y_te, name):
    start = time.perf_counter()
    model.fit(x_tr, y_tr, epochs=5, batch_size=128, verbose=0)
    model.evaluate(x_te, y_te, verbose=0)
    elapsed = time.perf_counter() - start
    acc = model.evaluate(x_te, y_te, verbose=0)[1]
    print(f"Temps {name:<15} : {elapsed:.2f} s | Accuracy : {acc:.4f}")
    return elapsed, acc

print("\n--- Classification ---")
clf_orig = build_classifier()
time_orig, acc_orig = train_and_time(
    clf_orig, x_train, y_train, x_test, y_test, "Originales"
)

clf_recon = build_classifier()
time_recon, acc_recon = train_and_time(
    clf_recon, x_train_recon, y_train, x_test_recon, y_test, "Reconstruites"
)

clf_comb = build_classifier()
time_comb, acc_comb = train_and_time(
    clf_comb, x_train_combined, y_train_combined, x_test, y_test, "Combinées"
)


# 7️ Matrices de confusion

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def plot_cm(model, x, y, title):
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.show()

plot_cm(clf_orig, x_test, y_test, "CM – Originales")
plot_cm(clf_recon, x_test_recon, y_test, "CM – Reconstruites")
plot_cm(clf_comb, x_test, y_test, "CM – Combinées")


# 8️ Résumé final

print("\n=========== RÉSUMÉ FINAL ===========")
print(f"Autoencodeur (train) : {ae_train_time:.2f} s")
print(f"Reconstruction       : {recon_time:.2f} s")
print("----------------------------------")
print(f"Orig  -> Acc {acc_orig:.4f} | Temps {time_orig:.2f} s")
print(f"Recon -> Acc {acc_recon:.4f} | Temps {time_recon:.2f} s")
print(f"Comb  -> Acc {acc_comb:.4f} | Temps {time_comb:.2f} s")

plt.figure(figsize=(8,4))
plt.bar(["Orig", "Recon", "Comb"], [time_orig, time_recon, time_comb])
plt.ylabel("Temps (s)")
plt.title("Temps d'entraînement des classifieurs")
plt.grid(True)
plt.show()
