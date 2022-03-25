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


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs 
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var)*epsilon


class Encoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim=5, intermediate_dim=20, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.dense_proj1 = tf.keras.layers.Dense(intermediate_dim-5, activation="relu")
        self.dense_proj2 = tf.keras.layers.Dense(intermediate_dim-10, activation="relu")
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim, activation="relu")
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        x1 = self.dense_proj1(x)
        x2 = self.dense_proj2(x1)
        z_mean = self.dense_mean(x2)
        z_log_var = self.dense_log_var(x2)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z 
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, orig_dim, intermediate_dim=20, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim-10, activation="relu")
        self.dense_proj1 = tf.keras.layers.Dense(intermediate_dim-5, activation="relu")
        self.dense_proj2 = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = tf.keras.layers.Dense(orig_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        x1 = self.dense_proj1(x)
        x2 = self.dense_proj2(x1)
        return self.dense_output(x2)

class VAE(tf.keras.Model):
    def __init__(self, orig_dim, intermediate_dim=20, latent_dim=5, name="vae", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(orig_dim = orig_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        #KL divergence
        kl_loss = -0.5*tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstruction


def create_train_model(train_dataset, epochs):
    

    
    orig_dim = 30

     
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()


    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            
        print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    return vae

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

def prediction(model, test_data):
    
    predictions = model(test_data)
    
    error = tf.keras.losses.mse(predictions, test_data).numpy()
    error = error.reshape(len(error), 1)

    return error


if __name__ == "__main__":

    # Data handling
    print("Preparing data...")
    tf.random.set_seed(1)
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    DH.setNanToMean()  # DH.fillWithImputer()
    DH.standardScale()

    batchsize=4000

    X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(
        whole_split=True
    )

    #train_dataset, validation_dataset = create_batched_data(
    #    X_train, y_train, X_back_test, y_val, batchsize
    #)

    

    #vae = create_train_model(train_dataset, epochs=10)
    vae = VAE(orig_dim=30, intermediate_dim=20, latent_dim=5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
    vae.fit(X_train, X_train, epochs=20, validation_data=(X_back_test, X_back_test))

    err_val = prediction(vae, X_val)
   
    s = err_val[np.where(y_val == 1)]
    b = err_val[np.where(y_val == 0)]

    plot_histo(s, b, y_val)