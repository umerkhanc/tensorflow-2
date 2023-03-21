import tensorflow as tf
import numpy as np

# Generate synthetic data for binary classification
X = np.random.randn(100, 5).astype(np.float32)
y = np.random.randint(2, size=(100, 1)).astype(np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(5,), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the data
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
