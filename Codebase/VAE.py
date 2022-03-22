import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                tf.keras.layers.Dense(units=30, activation="leakyrelu"),
                tf.keras.layers.Dense(units=20, activation="leakyrelu"),
                tf.keras.layers.Dense(units=10, activation="leakyrelu"),
                tf.keras.layers.Dense(units=(self.latent_dim + self.latent_dim)),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=10, activation="leakyrelu"),
                tf.keras.layers.Dense(units=20, activation="leakyrelu"),
                tf.keras.layers.Dense(units=30, activation="leakyrelu"),
            ]
        )

    @tf.function
    def sampling(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim))
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


def log_normal_pdf(sample, mean, logvariance, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    val = -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvariance) + logvariance + log2pi)

    return tf.reduce_sum(
        val,
        axis=raxis,
    )


def compute_loss(model, x):
    mean, logvariance = model.encode(x)
    z = model.reparameterize(mean, logvariance)
    x_logit = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

    logpx_z = -tf.reduce_sum(cross_entropy, axis=1)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvariance)
    loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return loss


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def create_batched_data(X_train, y_train, X_val, y_val, batch_size):
    train_size = np.shape(X_train)[0]
    test_size = np.shape(X_val)[0]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(X_train.astype("float32"))
        .shuffle(train_size)
        .batch(batch_size)
    )

    validation_dataset = (
        tf.data.Dataset.from_tensor_slices(X_val.astype("float32"))
        .shuffle(test_size)
        .batch(batch_size)
    )

    return train_dataset, validation_dataset


def train_val_VAE(train_dataset, validation_dataset, batchsize, epochs=10, latentdim=5):

    # Create batches

    epochs = epochs
    latentdim = latentdim
    batchsize = batchsize

    optimizer = tf.keras.optimizers.Adam()

    vae = VAE(latentdim, batchsize)  # .compile(optimizer=optimizer)

    with tf.device("/CPU:0"):
        for epoch in range(epochs + 1):
            start_time = time.time()

            # Add training session
            for train_x in train_dataset:
                train_step(vae, train_x, optimizer)

            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            # Validation check
            for val_x in validation_dataset:
                loss(compute_loss(vae, val_x))

            elbo = -loss.result()
            print(
                "Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}".format(
                    epoch, elbo, end_time - start_time
                )
            )

    return vae


def prediction(model, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sampling(z)

    error = tf.keras.losses.mse(predictions, test_sample).numpy()
    error = error.reshape(len(error), 1)

    return error


if __name__ == "__main__":

    import time

    tf.config.run_functions_eagerly(True)

    # Data handling
    print("Preparing data...")
    tf.random.set_seed(1)
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    DH.setNanToMean()  # DH.fillWithImputer()
    DH.standardScale()

    X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(
        whole_split=True
    )

    batchsize = 4000

    train_dataset, validation_dataset = create_batched_data(
        X_train, y_train, X_val, y_val, batchsize
    )

    trained_model = train_val_VAE(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batchsize=batchsize,
        epochs=15,
        latentdim=2,
    )

    error_back = prediction(trained_model, X_back_test)
    error_sig = prediction(trained_model, X_sig_test)

    n, bins, patches = plt.hist(
        error_back, 1000, density=True, facecolor="b", label="Background"
    )
    n, bins, patches = plt.hist(
        error_sig, 1000, density=True, alpha=0.6, facecolor="r", label="Signal"
    )
    plt.legend()
    plt.show()
