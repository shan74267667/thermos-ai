import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1) Generate synthetic data
#   Inputs: [workload%, ambient_temp°C]
#   Output: server inlet temp°C
N = 2000
workload = np.random.rand(N,1) * 100      # 0–100% load
ambient  = np.random.rand(N,1) * 15 + 20  # 20–35°C
# simple heat model: base + load*0.2 + ambient*0.5 + noise
temps = 25 + workload*0.2 + ambient*0.5 + np.random.randn(N,1)*0.5

X = np.hstack([workload, ambient])
y = temps

# 2) Build & train a small NN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=30, batch_size=32, verbose=0)

# 3) Plot training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Thermal Model Training')
plt.show()

# 4) Quick inference test
test = np.array([[80, 30], [20, 25]])   # heavy vs light load
pred = model.predict(test)
print("Predicted temps:", pred.flatten())
