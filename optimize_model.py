import tensorflow as tf

# ====================================
# 1. Carregamento do  modelo treinado
# ====================================
model = tf.keras.models.load_model("model.h5")



# =========================
# 2. Conversão para TFLite
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Ativa quantização (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()



# ===============================
# 3. Salvando o modelo otimizado
# ===============================
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido e salvo como model.tflite")