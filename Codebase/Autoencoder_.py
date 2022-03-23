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

# for custom activation function
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})


# Data handling
print("Preparing data...")
tf.random.set_seed(1)
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.setNanToMean()#DH.fillWithImputer()
DH.standardScale()

X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(whole_split=True)


# Get optimal model through previous gridsearch
print("Fetching optimal parameters...")
name = "hypermodel_ae"
hypermodel = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")

# Train to find best epoch
print("Training model.")
with tf.device("/CPU:0"):
    history = hypermodel.fit(
        X_train, X_train, epochs=40, batch_size=4000, validation_data=(X_back_test, X_back_test)
    )
mse_hist = history.history["val_mse"]
loss_hist = history.history["val_loss"]
best_epoch = mse_hist.index(min(mse_hist))

print(
    f"Validation loss, Validation mse : {loss_hist[best_epoch]:.2f} , {mse_hist[best_epoch]:.2f}, best epoch is {best_epoch}"
)

"""
fig, ax1 = plt.subplots(num=0, dpi=80, facecolor="w", edgecolor="k")
fig.suptitle("Autoencoder history", fontsize=16)
color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
ax1.set_xlabel(r"#$Epochs$", fontsize=16)
ax1.set_ylabel(r"$mse$", fontsize=16, color=color)
ax1.plot(history.history["val_mse"], color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
ax2.set_ylabel(
    r"$Loss$", color=color, fontsize=16
)  # we already handled the x-label with ax1
ax2.plot(history.history["val_loss"], color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("../figures/AE_hist.pdf", bbox_inches="tight")
plt.show()
"""
"""
with tf.device("/CPU:0"):
    history = hypermodel.fit(
        X_train, X_train, epochs=40, batch_size=4000, validation_data=(X_back_test, X_back_test)
    )
"""
reconstruct = hypermodel(X_train)
recon_error = tf.keras.losses.msle(reconstruct, X_train)
print("Mean error: {} and std error: {}".format(np.mean(recon_error.numpy()), np.std(recon_error.numpy())))
threshold = 1000*(np.mean(recon_error.numpy()) + np.std(recon_error.numpy())   )     


prediction_back = hypermodel(X_back_test)
errorsback = tf.keras.losses.msle(prediction_back, X_back_test).numpy()
errorsback = errorsback.reshape(len(errorsback), 1)

prediction_sig = hypermodel(X_sig_test)
errorssig = tf.keras.losses.msle(prediction_sig, X_sig_test).numpy()

errorssig = errorssig.reshape(len(errorssig), 1)







recon_val = hypermodel(X_val)
err_val = tf.keras.losses.msle(recon_val, X_val).numpy()
err_val = err_val.reshape(len(err_val), 1)

s = err_val[np.where(y_val == 1)]
b = err_val[np.where(y_val == 0)]

sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))/sigma
x_start = np.mean(b) *5
x_end =np.mean(s) *7
y_start = 10 

"""
plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n, bins, patches = plt.hist(errorsback, 1000, histtype="stepfilled", density=True, facecolor="b", label="Background")
n, bins, patches = plt.hist(errorssig, 1000, histtype="stepfilled", density=True, alpha=0.6,facecolor="r", label="signal")
plt.xlabel("Error", fontsize=15)
plt.ylabel("#-of-events", fontsize=15)
plt.title("Autoencoder error distribution", fontsize=15, fontweight = "bold")
plt.annotate("", xy=(x_start,y_start),
            xytext=(x_end,y_start),verticalalignment="center",
            arrowprops={'arrowstyle': '|-|', 'lw': 1, "color":"black"}, va='center')
plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}" + r"$\sigma_b$",
                xy=(((x_start+x_end)/2), y_start+20), xycoords='data',fontsize=15.0,textcoords='data',ha='center')

plt.legend(fontsize = 16)
plt.savefig("../figures/AE/AE_error1.pdf", bbox_inches="tight")
plt.show()
"""

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
plt.savefig("../figures/AE/AE_error2.pdf", bbox_inches="tight")
plt.show()

probas = np.concatenate((1-err_val, err_val),axis=1)

skplt.metrics.plot_roc(y_val, probas)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("Autoencoder: ROC curve", fontsize=15, fontweight = "bold")
plt.savefig("../figures/AE/AE_ROC.pdf", bbox_inches="tight")
plt.show()