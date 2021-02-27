# f = c*1.8 + 32

import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger().setLevel(logging.ERROR)
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i, c in enumerate(celsius_q):
  print("Celsius {} is Fahranheit {}", c, fahrenheit_a[i])

# Build the model
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential(
    layer
)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("The history of training model: {}", history.history) # a list of 500 loss
print("The model of training model: {}", history.model)
print("There are the layer variables: {}\n\n".format(layer.get_weights()))

# Plot the loss over training epoch
import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

# prediction
print("10 celsius is {} Fahrenheit.\n\n".format(model.predict(np.array([10]))))

# Experiment, Train neural model with 3 layers
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("In new model predict {}".format(model.predict([10])))
print("L0 variables: {}".format(l0.weights))
print("L1 variables: {}".format(l1.weights))
print("L2 variables: {}".format(l2.weights))
