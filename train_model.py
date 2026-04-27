import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# 1. Carregar dados
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalização
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajuste de shape (28,28) -> (28,28,1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# =========================
# 2. Modelo CNN
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# =========================
# 3. Compilar
# =========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 4. Treinar
# =========================
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=2
)

# =========================
# 5. Avaliar
# =========================
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final: {acc:.4f}")

# =========================
# 6. Salvar
# =========================
model.save("model.h5")
print("Modelo salvo como model.h5")