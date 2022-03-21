import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from DataHandler import DataHandler
from tensorflow.keras import optimizers
import keras_tuner as kt
from Functions import timer

# for custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, batch_size):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.batch = batch_size

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(30,)),
                tf.keras.layers.Dense(
                    units=30, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    units=20, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    units=10, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(units=(self.latent_dim + self.latent_dim)),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(
                    units=10, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    units=20, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    units=30, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
            ]
        )

    @tf.function
    def sampling(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(self.batch, self.latent_dim))
        return self.decode(epsilon, apply_sigmoid=True)

    def encode(self, x):
        mean, logvariance = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvariance

    def reparameterize(self, mean, logvariance):
        epsilon = tf.random.normal(shape=(mean.shape))
        return epsilon * tf.exp(logvariance * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

class TrainPredVAE():
    def __init__(self, model):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer)

    def log_normal_pdf(self, sample, mean, logvariance, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2 * tf.exp(-logvariance) + logvariance + log2pi),
            axis=raxis,
        )

    def compute_loss(self, x):
        mean, logvariance = self.model.encode(x)
        z = self.model.reparameterize(mean, logvariance)
        x_logit = self.model.decode(z)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_entropy, axis=1)
        logpz = self.log_normal_pdf(z, 0, 0)
        logqz_x = self.log_normal_pdf(z, mean, logvariance)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function 
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainabla_variables))




if __name__ == "__main__":
    import time

    # Data handling
    print("Preparing data...")
    tf.random.set_seed(1)
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    DH.fillWithImputer()
    DH.standardScale()

    X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(whole_split=True)


    epochs = 50
    latentdim = 2
    batchsize = 4000

    vae = VAE(latentdim, batchsize)

    with tf.device("/CPU:0"):
        for epoch in range(epochs +1):
            start_time = time.time()

            # Add training session

            end_time = time.time()


            loss = tf.keras.metric.Mean()
            # Validation check

            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, end_time - start_time))
