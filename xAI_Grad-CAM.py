import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os

# Ordner für XAI-Ergebnisse erstellen
os.makedirs("xai_results/plots", exist_ok=True)

print("="*60)
print("1. Lade und vorbereite Fashion-MNIST Datensatz (mit tf.data)...")
print("="*60)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisierung & Kanal-Dimension hinzufügen
train_images = (train_images.astype("float32") / 255.0)[..., tf.newaxis]
test_images = (test_images.astype("float32") / 255.0)[..., tf.newaxis]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Optimierte tf.data Pipelines für schnelles Training
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_split_idx = int(len(train_images) * 0.8)
val_dataset = tf.data.Dataset.from_tensor_slices((train_images[val_split_idx:], train_labels[val_split_idx:]))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==========================================
# MODELL-ERSTELLUNG & TRAINING
# ==========================================
print("\n" + "="*60)
print("2. Trainiere das CNN (Optimierte Architektur)...")
print("="*60)

inputs = tf.keras.Input(shape=(28, 28, 1), name="input_layer")
x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', name="conv1")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="pool1")(x)
x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', name="conv2")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="pool2")(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(128, activation='relu', name="feature_dense")(x)
x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
outputs = tf.keras.layers.Dense(10, activation='softmax', name="predictions")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Fashion_CNN")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training mit tf.data Pipelines
model.fit(train_dataset, validation_data=val_dataset, epochs=5, verbose=1)


# ==========================================
# TEIL 1: GRAD-CAM (Visualisierung der Aufmerksamkeit)
# ==========================================
print("\n" + "="*60)
print("TEIL 1: Generiere Grad-CAM Heatmaps...")
print("="*60)

# Korrektur: model.input statt model.inputs
grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer("conv2").output, model.output]
)

def make_gradcam_heatmap(img_array, grad_model):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

plt.figure(figsize=(12, 6))
for i in range(5):
    img = test_images[i:i+1]
    true_label = class_names[test_labels[i]]
    
    heatmap = make_gradcam_heatmap(img, grad_model)
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title(true_label)
    plt.axis('off')
    
    plt.subplot(2, 5, i + 6)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')

plt.tight_layout()
plt.savefig("xai_results/plots/gradcam_examples.png")
plt.close()
print("-> Grad-CAM Plot gespeichert unter 'xai_results/plots/gradcam_examples.png'")


# ==========================================
# TEIL 2: FAILURE ANALYSIS (Fehleranalyse)
# ==========================================
print("\n" + "="*60)
print("TEIL 2: Führe Failure Analysis durch...")
print("="*60)

preds = model.predict(test_images, batch_size=256, verbose=0)
pred_classes = np.argmax(preds, axis=1)

errors = np.where(pred_classes != test_labels)[0]
print(f"Anzahl Fehler auf dem Testdatensatz: {len(errors)} von {len(test_images)}")

plt.figure(figsize=(12, 6))
num_samples = min(10, len(errors))
for i, idx in enumerate(errors[:num_samples]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx].squeeze(), cmap='gray')
    true_name = class_names[test_labels[idx]]
    pred_name = class_names[pred_classes[idx]]
    plt.title(f"E: {true_name}\nP: {pred_name}", color='red', fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig("xai_results/plots/failure_analysis_grid.png")
plt.close()

plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_labels, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Fehler-Schwerpunkte)")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("xai_results/plots/failure_confusion_matrix.png")
plt.close()
print("-> Failure Analysis Plots gespeichert in 'xai_results/plots/'")


# ==========================================
# TEIL 3: t-SNE FEATURE SPACE (Latenter Raum)
# ==========================================
print("\n" + "="*60)
print("TEIL 3: Berechne PCA + t-SNE Projektion des Feature Space...")
print("="*60)

feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("feature_dense").output)

# Stichprobe für performante t-SNE Berechnung
sample_size = 3000
subset_data = test_images[:sample_size]
subset_labels = test_labels[:sample_size]

features = feature_extractor.predict(subset_data, batch_size=256, verbose=0)

# Schritt 1: PCA-Reduktion für Stabilität und Performance
pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(features)

# Schritt 2: t-SNE Anwenden (Korrektur: max_iter statt n_iter)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, n_jobs=-1)
features_2d = tsne.fit_transform(features_pca)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=subset_labels, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label="Klassen")
plt.title("t-SNE Visualisierung des inneren Feature-Raums (mit PCA Vorfilter)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.savefig("xai_results/plots/tsne_feature_space.png")
plt.close()

print("-> t-SNE Plot gespeichert unter 'xai_results/plots/tsne_feature_space.png'")
print("\n" + "="*60)
print("OPTIMIERTES XAI-PROJEKT ERFOLGREICH ABGESCHLOSSEN!")
print("="*60)