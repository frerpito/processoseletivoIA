from tensorflow import keras
import tensorflow as tf
import os

#insira seu código aqui
model = keras.models.load_model("model.h5")

# converter para tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_dynamic = converter.convert()

# salvar na raiz do projeto
tfl_dyn_path = "model.tflite"

with open(tfl_dyn_path, "wb") as f:
    f.write(tflite_dynamic)

