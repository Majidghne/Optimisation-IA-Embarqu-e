
"""
Created on Thu December 2026

@author: Majid GHORBANNEZHAD
"""

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
from tensorflow.keras.layers import Dropout
import time # Import the time module



# 1. Configuration et Chargement

np.random.seed(42)
tf.random.set_seed(42)

# Chargement avec labels pour la classification
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# Normalisation
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Reshape pour CNN (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

H, W, C = 28, 28, 1
latent_dim = 60


# 2. Construction de l'Autoencodeur

inputs = Input(shape=(H, W, C))

# Encoder
x = Conv2D(32, (3,3), padding="same")(inputs)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(256)(x)
x = Dense(latent_dim)(x)
x = LeakyReLU(alpha=0.2)(x)

# Decoder
x = Dense(7*7*64)(x)
x = Reshape((7,7,64))(x)

x = Conv2DTranspose(64, (3,3), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

x = Conv2DTranspose(32, (3,3), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

# Entraînement
print("--- Entraînement de l'Autoencodeur ---")
start_time_ae_train = time.time()
history = autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=256,
    validation_data=(x_test, x_test),
    verbose=1
)
end_time_ae_train = time.time()
print(f"Temps d'entraînement Autoencodeur: {end_time_ae_train - start_time_ae_train:.2f} secondes")

# Reconstruction des images pour la suite
start_time_ae_recon_train = time.time()
x_train_recon = autoencoder.predict(x_train, batch_size=128)
end_time_ae_recon_train = time.time()
print(f"Temps de reconstruction (train): {end_time_ae_recon_train - start_time_ae_recon_train:.2f} secondes")

start_time_ae_recon_test = time.time()
x_test_recon = autoencoder.predict(x_test, batch_size=128)
end_time_ae_recon_test = time.time()
print(f"Temps de reconstruction (test): {end_time_ae_recon_test - start_time_ae_recon_test:.2f} secondes")


# 3. Évaluation de la Qualité (Metrics)
=
def compute_psnr(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

mse_val = np.mean((x_test - x_test_recon) ** 2)
psnr_val = compute_psnr(x_test, x_test_recon)
ssim_val = np.mean([ssim(x_test[i].reshape(28,28), x_test_recon[i].reshape(28,28), data_range=1.0) for i in range(len(x_test))])
print(f"\n[METRIQUES AE]")
print(f"MSE: {mse_val:.4f}")
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM (moyenne sur 1000 images): {ssim_val:.4f}")


# 4. Classification Comparative


# Préparation du 3ème dataset (Combiné)
x_train_combined = np.concatenate([x_train, x_train_recon], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

def build_classifier():
    model = tf.keras.Sequential([
        Flatten(input_shape=(H, W, C)), # Flatten the input images
        Dense(60, activation='relu'), # Dense layer
        Dropout(0.25), # Dropout for regularization
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Test 1: Originales
clf_orig = build_classifier()
print("\n--- Classifieur sur Images Originales ---")
start_time_clf_orig_train = time.time()
clf_orig.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)
end_time_clf_orig_train = time.time()
print(f"Temps d'entraînement (Originales): {end_time_clf_orig_train - start_time_clf_orig_train:.2f} secondes")

start_time_clf_orig_eval = time.time()
acc_orig = clf_orig.evaluate(x_test, y_test, verbose=0)[1]
end_time_clf_orig_eval = time.time()
print(f"Temps d'évaluation (Originales): {end_time_clf_orig_eval - start_time_clf_orig_eval:.2f} secondes")

# Test 2: Reconstruites
clf_recon = build_classifier()
print("--- Classifieur sur Images Reconstruites ---")
start_time_clf_recon_train = time.time()
clf_recon.fit(x_train_recon, y_train, epochs=5, batch_size=128, verbose=0)
end_time_clf_recon_train = time.time()
print(f"Temps d'entraînement (Reconstruites): {end_time_clf_recon_train - start_time_clf_recon_train:.2f} secondes")

start_time_clf_recon_eval = time.time()
acc_recon = clf_recon.evaluate(x_test, y_test, verbose=0)[1]
end_time_clf_recon_eval = time.time()
print(f"Temps d'évaluation (Reconstruites): {end_time_clf_recon_eval - start_time_clf_recon_eval:.2f} secondes")

# Test 3: Combinées
clf_comb = build_classifier()
print("--- Classifieur sur Images Combinées ---")
start_time_clf_comb_train = time.time()
clf_comb.fit(x_train_combined, y_train_combined, epochs=5, batch_size=128, verbose=0)
end_time_clf_comb_train = time.time()
print(f"Temps d'entraînement (Combinées): {end_time_clf_comb_train - start_time_clf_comb_train:.2f} secondes")

start_time_clf_comb_eval = time.time()
acc_comb = clf_comb.evaluate(x_test, y_test, verbose=0)[1]
end_time_clf_comb_eval = time.time()
print(f"Temps d'évaluation (Combinées): {end_time_clf_comb_eval - start_time_clf_comb_eval:.2f} secondes")


# 5. Visualisation des Résultats 

# Visualisation Classification
plt.figure(figsize=(10, 5))
labels = ['Originales', 'Reconstruites', 'Combinées']
accs = [acc_orig, acc_recon, acc_comb]
plt.bar(labels, accs, color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Comparaison de l\'Accuracy du Classifieur')
plt.ylabel('Accuracy')
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

# Visualisation AE (Original vs Reconstruction)
n = 8
plt.figure(figsize=(16, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0: ax.set_title("Original")

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_recon[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0: ax.set_title("Reconstruit")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Noms des classes Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_confusion_matrix(model, x_test, y_test, title):
    # Obtenir les prédictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculer la matrice
    cm = confusion_matrix(y_test, y_pred)

    # Affichage
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues', ax=ax)
    plt.title(f"Matrice de Confusion : {title}")
    plt.show()

# --- Génération des matrices pour chaque modèle ---
print("\n--- Génération des Matrices de Confusion ---")

# 1. Pour les images originales
plot_confusion_matrix(clf_orig, x_test, y_test, "Images Originales")

# 2. Pour les images reconstruites
# Note : on teste ici sur les images reconstruites de test
plot_confusion_matrix(clf_recon, x_test_recon, y_test, "Images Reconstruites")

# 3. Pour le modèle combiné
# On teste sur les images originales pour voir si l'entraînement hybride a aidé
plot_confusion_matrix(clf_comb, x_test, y_test, "Modèle Combiné")