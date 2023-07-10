import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

class AutoEncoders(tf.keras.Model):
    def __init__(self, input_dim, output_dim, create):

        super().__init__()
        if create :
            # This architecture was obtained after exploration on the Curry cluster (see file <TODO>)
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(153, activation="relu"),
                    tf.keras.layers.Dense(121, activation="relu"),
                    tf.keras.layers.Dense(89, activation="relu"),
                    tf.keras.layers.Dense(57, activation="relu"),
                    tf.keras.layers.Dense(output_dim, activation="relu"),
                ], name='sequential'
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(57, activation="relu"),
                    tf.keras.layers.Dense(89, activation="relu"),
                    tf.keras.layers.Dense(121, activation="relu"),
                    tf.keras.layers.Dense(157, activation="relu"),
                    tf.keras.layers.Dense(input_dim, activation="sigmoid"),
                ]
            )
        else:
            self.encoder = load_model('../output/models/Autoencoders/autoencoder_25.h5')

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
