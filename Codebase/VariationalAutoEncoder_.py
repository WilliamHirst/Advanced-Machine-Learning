import numpy as np
import pandas as pd
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import plot_set
from sklearn.metrics import accuracy_score
import scikitplot as skplt
from Functions import *

# for custom activation function
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update(
    {"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})

class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs 
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var)*epsilon


class Encoder(tf.keras.layers.Layer):

    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(21, activation="leakyrelu")
        self.dense_proj1 = tf.keras.layers.Dense(15, activation="tanh")
        self.dense_proj2 = tf.keras.layers.Dense(8, activation="leakyrelu")
        self.dense_mean = tf.keras.layers.Dense(4)
        self.dense_log_var = tf.keras.layers.Dense(4, activation="tanh")
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
    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(9, activation="relu")
        self.dense_proj1 = tf.keras.layers.Dense(15, activation="leakyrelu")
        self.dense_proj2 = tf.keras.layers.Dense(20, activation="tanh")
        self.dense_output = tf.keras.layers.Dense(30, activation="leakyrelu")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        x1 = self.dense_proj1(x)
        x2 = self.dense_proj2(x1)
        return self.dense_output(x2)

class VAE(tf.keras.Model):
    def __init__(self, name="vae", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        #KL divergence
        kl_loss = -0.5*tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstruction


print("Preparing data...")
tf.random.set_seed(1)
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:, 0].astype(int)
X_test = X_test[:, 1:]

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
#DH.removeOutliers(5)

nr_train = DH.nrEvents
DH.X_train = np.concatenate((DH.X_train, X_test), axis=0)
DH.setNanToMean()
DH.standardScale()

X, Y = DH(include_test=False)

X_train = X[: nr_train, :]
X_test = X[nr_train:, :]
DH.X_train = X_train


X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(
    whole_split=True)


# Get optimal model through previous gridsearch
print("Fetching optimal parameters...")


#vae = create_train_model(train_dataset, epochs=10)
hypermodel = VAE()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
hypermodel.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
hypermodel.fit(X_train, X_train, epochs=20, validation_data=(X_back_test, X_back_test))




recon_val = hypermodel(X_val)
err_val = tf.keras.losses.msle(recon_val, X_val).numpy()
err_val = err_val.reshape(len(err_val), 1)
#indx = np.where(err_val<np.mean(err_val)+5*np.std(err_val))[0] #Remove outliers in reconstruction.
indx = np.where(err_val > -1)[0]
err_val = err_val[indx]/np.max(err_val[indx])

s = err_val[np.where(y_val[indx] == 1)]
b = err_val[np.where(y_val[indx] == 0)]
#s = s[np.where(s<np.mean(s)+5*np.std(s))[0]]
#b = b[np.where(b<np.mean(b)+5*np.std(b))[0]]


sigma = np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))
x_start = np.mean(b)
x_end = np.mean(s)
y_start = 8


threshold = np.mean(b) + np.std(b)
recon_val_test = hypermodel(X_test)
proba = tf.keras.losses.msle(recon_val_test, X_test).numpy()
name = '../Data/Autoencoder_test_pred.csv'
#write_to_csv(EventID, proba, threshold, name)

binsize = 150
plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                  label="Background", density=True)
n_s, bins_s, patches_s = plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r", alpha=0.6,
                                  label="Signal",  density=True)

median_s = bins_s[np.where(n_s == np.max(n_s))][0]
median_b = bins_b[np.where(n_b == np.max(n_b))][0]
#plt.axvline(x=x_start,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
#plt.axvline(x=x_end,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
#plt.axvline(x=median_s,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
#plt.axvline(x=median_b,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#Events", fontsize=15)
plt.title("Autoencoder output distribution", fontsize=15, fontweight="bold")
plt.legend(fontsize=16, loc="upper right")

plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$"
             + f" = {diff:.3f}",
             xy=((0.5), y_start+1.), xycoords='data',
             fontsize=15.0, textcoords='data', ha='center')

plt.annotate(r"$\mid s_m-b_m\mid$"+f" = {abs(median_b-median_s):.3f}", xycoords='data',
             xy=(0.5, y_start+4.),
             fontsize=15.0, textcoords='data', ha='center')
plt.savefig("../figures/AE/AE_output.pdf", bbox_inches="tight")
plt.show()


probas = np.concatenate((1-err_val, err_val), axis=1)

skplt.metrics.plot_roc(y_val[indx], probas)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("Variational Autoencoder: ROC curve", fontsize=15, fontweight="bold")
plt.savefig("../figures/VAE/VAE_ROC.pdf", bbox_inches="tight")
plt.show()
