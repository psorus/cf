import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Input layer
input_layer = Input(shape=(28, 28))

# Flatten layer
flatten_layer = Flatten()(input_layer)

# Hidden dense layer with 128 units and ReLU activation
hidden_layer = Dense(128, activation='relu')(flatten_layer)

# Output layer with 10 units (one for each digit) and softmax activation
output_layer = Dense(10, activation='softmax')(hidden_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

submodel=Model(inputs=input_layer, outputs=hidden_layer)
pred=submodel.predict(x_test)
print(pred.shape)

np.savez_compressed('pred.npz',q=pred)



