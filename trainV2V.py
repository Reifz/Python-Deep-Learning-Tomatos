import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.utils import class_weight

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 15
EPOCHS_FINE = 15

# Recomendo usar "dataset" pois o "dataset2" tem pouquíssimas imagens
train_dir = "dataset/train"
val_dir = "dataset/validation"

if not os.path.exists(train_dir):
    print(f"ERRO: Diretorio {train_dir} nao encontrado. Verifique a configuracao.")
    exit()

# =========================
# DATASET
# =========================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_data.class_names
print("Classes:", class_names)

# Calcular pesos das classes para lidar com desbalanceamento
labels = np.concatenate([y for x, y in train_data], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(weights))
print("Pesos das classes calculados:", class_weights)

# Performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

# =========================
# BASE MODEL
# =========================
# MobileNetV2 espera entrada no intervalo [-1, 1]
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # fase 1

# =========================
# MODEL
# =========================
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    # Correcao do preprocessamento para MobileNetV2: [-1, 1]
    layers.Rescaling(scale=1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

# =========================
# COMPILE (FASE 1)
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_tomato3_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# =========================
# TREINO FASE 1
# =========================
print("\n🔵 Treinando camadas finais...\n")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_INITIAL,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# =========================
# FINE-TUNING
# =========================
print("\n🔴 Iniciando fine-tuning...\n")

# Desbloquear as ultimas camadas do modelo base para ajuste fino
base_model.trainable = True
# Mas vamos congelar as primeiras camadas (opcional, MobileNetV2 tem 154 camadas)
# Congelar ate a camada 100
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# =========================
# SALVAR MODELO FINAL
# =========================
model.save("tomato_model_v2_d3.h5")

# =========================
# SALVAR CLASSES
# =========================
with open("classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("\n✅ Treinamento finalizado e modelos salvos!\n")