import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Save the model
model.save('handwritten_model.keras')
model = tf.keras.models.load_model('handwritten_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Loop through numbered image files
img_no = 0  # Starting from 0.png
while os.path.isfile(f"{img_no}.png"):
    try:
        # Load and preprocess the image
        img_path = f"{img_no}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        img = cv2.resize(img, (28, 28))  # Resize to 28x28
        img = np.invert(img)             # Invert the colors
        img = img / 255.0                # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img)
        print(f"Prediction for {img_path}: {np.argmax(prediction)}")

        # Display the image with its prediction
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    finally:
        img_no += 1
