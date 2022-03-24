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
import scikitplot as skplt

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
    optimizer = tf.keras.optimizers.Adam()

    vae = VAE(latentdim, batchsize)#.compile(optimizer=optimizer)

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
                "Epoch: {}, Test set ELBO: {:.2e}, time elapse for current epoch: {:.2e}s".format(
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


def plot_histo(s, b, y_val):
    sigma =np.nanstd(b)
    diff = abs(np.mean(b) - np.mean(s))/sigma
    x_start = np.mean(b) *5
    x_end =np.mean(s) *7
    y_start = 10 
    
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.hist(b, bins=100, histtype="stepfilled", facecolor="b",label = "Background", density=True)
    plt.hist(s, bins=100, histtype="stepfilled", facecolor="r",alpha=0.6, label = "Signal", density=True)
    plt.legend(fontsize = 16)
    plt.xlabel("Error", fontsize=15)
    plt.ylabel("#-of-events", fontsize=15)
    plt.title("Autoencoder error distribution", fontsize=15, fontweight = "bold")
    plt.annotate("", xy=(x_start,y_start),
                xytext=(x_end,y_start),verticalalignment="center",
                arrowprops={'arrowstyle': '|-|', 'lw': 1, "color":"black"}, va='center')
    plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                    + f" = {diff:.2f}" + r"$\sigma_b$",
                    xy=(((x_start+x_end)/2), y_start + 5), xycoords='data',fontsize=15.0,textcoords='data',ha='center')
    plt.savefig("../figures/VAE/VAE_error2.pdf", bbox_inches="tight")
    plt.show()

    probas = np.concatenate((1-err_val, err_val),axis=1)

    skplt.metrics.plot_roc(y_val, probas)
    plt.xlabel("True positive rate", fontsize=15)
    plt.ylabel("False positive rate", fontsize=15)
    plt.title("Autoencoder: ROC curve", fontsize=15, fontweight = "bold")
    plt.savefig("../figures/VAE/VAE_ROC.pdf", bbox_inches="tight")
    plt.show()

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
        X_train, y_train, X_back_test, y_val, batchsize
    )

    trained_model = train_val_VAE(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batchsize=batchsize,
        epochs=15,
        latentdim=2,
    )

    err_val = prediction(trained_model, X_val)
   
    s = err_val[np.where(y_val == 1)]
    b = err_val[np.where(y_val == 0)]

    plot_histo(s, b, y_val)
