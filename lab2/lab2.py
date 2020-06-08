# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

nn_0 = np.loadtxt('data/nn_0.csv', delimiter=',', dtype=str)[1:, :]
nn_1 = np.loadtxt('data/nn_1.csv', delimiter=',', dtype=str)[1:, :]

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(2,))
])

x = nn_0[:, 0:2]
y = nn_0[:, -1]
model.compile()
history = model.fit(x, y, epochs=10, steps_per_epoch=500, validation_steps=2,
                    use_multiprocessing=True)
predictions = model.predict(nn_0[:, 0:1])
print(predictions)
