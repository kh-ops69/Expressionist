import tensorflow as tf
from tensorflow.python.keras import Sequential, Model
# from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

# create train-test split of dataset
# X_train.shape, y_train.shape

X_train, X_test = 0, 0

X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units = 128, activation='relu', input_dim = 784))
autoencoder.add(Dense(units = 64, activation='relu'))
autoencoder.add(Dense(units = 32, activation='relu')) # Encoded image

# Decode
autoencoder.add(Dense(units = 64, activation='relu'))
autoencoder.add(Dense(units = 128, activation='relu'))
autoencoder.add(Dense(units = 784, activation='sigmoid'))

print(autoencoder.summary())

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
autoencoder.fit(X_train, X_train, epochs=50)

print(autoencoder.get_layer('dense').input)
print(autoencoder.get_layer('dense_2').output)
encoder = Model(inputs = autoencoder.get_layer('dense').input, outputs = autoencoder.get_layer('dense_2').output)
print(encoder.summary())

# encoded_image = encoder.predict(X_test[0].reshape(1,-1))
input_layer_decoder = Input(shape=(32,))
decoder_layer1 = autoencoder.layers[3]
decoder_layer2 = autoencoder.layers[4]
decoder_layer3 = autoencoder.layers[5]
decoder = Model(inputs = input_layer_decoder, outputs = decoder_layer3(decoder_layer2(decoder_layer1(input_layer_decoder))))
print(decoder.summary())
# decoded_image = decoder.predict(encoded_image)

# Assuming 'encoder' and 'decoder' are the trained encoder and decoder models
img_batch = np.array(X_test[:50])  # Replace with your image data

# Encode multiple images to get latent vectors
encoded_images = [encoder.predict(img.reshape(1,-1)) for img in img_batch]  # Where image_batch is a batch of images

# Stack latent vectors and find min-max range for each index across vectors
latent_vectors = np.stack(encoded_images)
min_vals = np.min(latent_vectors, axis=0)
max_vals = np.max(latent_vectors, axis=0)

# Generate a random latent vector within min-max range
random_latent = np.random.uniform(min_vals, max_vals)

# Decode to generate a new image
generated_image = decoder.predict(random_latent[np.newaxis, :].reshape(1,32))  # Reshape to include batch dimension
plt.imshow(generated_image.reshape(28,28), cmap='gray');