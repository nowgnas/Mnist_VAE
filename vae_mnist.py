import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim

        '''Encoder'''
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Flatten())

        for i in range(2):
            self.encoder.add(layers.Dense(256))
            self.encoder.add(layers.BatchNormalization())
            self.encoder.add(layers.Activation(tf.keras.activations.relu))

        self.encoder.add(layers.Dense(latent_dim))

        # layers.Flatten(),
        # layers.Dense(latent_dim, activation='relu'),

        '''Decoder'''
        self.decoder = tf.keras.Sequential()
        for i in range(2):
            self.decoder.add(layers.Dense(256))
            self.decoder.add(layers.BatchNormalization())
            self.decoder.add(layers.Activation(tf.keras.activations.sigmoid))

        self.decoder.add(layers.Dense(784, activation='sigmoid'))
        self.decoder.add(layers.Reshape((28, 28)))

        #   layers.Dense(784, activation='sigmoid'),
        #   layers.Reshape((28, 28))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
