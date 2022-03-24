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

get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})

print("Preparing data...")
tf.random.set_seed(1)
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,0].astype(int)
X_test = X_test[:,1:]

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
#DH.removeOutliers(5)

nr_train = DH.nrEvents
DH.X_train = np.concatenate((DH.X_train, X_test), axis=0)
DH.setNanToMean()
DH.standardScale()

X, Y = DH(include_test=False)

X_train = X[: nr_train,:]
X_test = X[nr_train:,:]
DH.X_train = X_train


X_train, y_train, X_val, y_val, X_back_test, X_sig_test = DH.AE_prep(whole_split=True)


# Get optimal model through previous gridsearch
print("Fetching optimal parameters...")
name = "hypermodel_ae"
hypermodel = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")
"""
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
with tf.device("/CPU:0"):
    hypermodel.fit(
        X_train, X_train, epochs=40, batch_size=4000, validation_data=(X_back_test, X_back_test)
    )





recon_val = hypermodel(X_val)
err_val = tf.keras.losses.msle(recon_val, X_val).numpy()
err_val = err_val.reshape(len(err_val), 1)
#indx = np.where(err_val<np.mean(err_val)+5*np.std(err_val))[0] #Remove outliers in reconstruction.
indx = np.where(err_val>-1)[0]
err_val = err_val[indx]/np.max(err_val[indx])

s = err_val[np.where(y_val[indx] == 1)]
b = err_val[np.where(y_val[indx] == 0)]
#s = s[np.where(s<np.mean(s)+5*np.std(s))[0]]
#b = b[np.where(b<np.mean(b)+5*np.std(b))[0]]


sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))
x_start = np.mean(b) 
x_end =np.mean(s) 
y_start = 8


threshold = np.mean(b) + np.std(b)
recon_val_test = hypermodel(X_test)
proba = tf.keras.losses.msle(recon_val_test, X_test).numpy()
name = '../Data/Autoencoder_test_pred.csv'
#write_to_csv(EventID, proba, threshold, name)

binsize = 150
plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                     label = "Background", density=True)
n_s, bins_s, patches_s = plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, 
                                     label = "Signal",  density=True)

median_s = bins_s[np.where(n_s==np.max(n_s))][0]
median_b = bins_b[np.where(n_b==np.max(n_b))][0]
#plt.axvline(x=x_start,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
#plt.axvline(x=x_end,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
#plt.axvline(x=median_s,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
#plt.axvline(x=median_b,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#Events", fontsize=15)
plt.title("Autoencoder output distribution", fontsize=15, fontweight = "bold")
plt.legend(fontsize = 16, loc = "upper right")

plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.3f}",
                xy=((0.5), y_start+1.), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.annotate(r"$\mid s_m-b_m\mid$"+f" = {abs(median_b-median_s):.3f}", xycoords='data',
                xy =(0.5, y_start+4.), 
                fontsize=15.0,textcoords='data',ha='center')
plt.savefig("../figures/AE/AE_output.pdf", bbox_inches="tight")
plt.show()


probas = np.concatenate((1-err_val, err_val),axis=1)

skplt.metrics.plot_roc(y_val[indx], probas)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("Autoencoder: ROC curve", fontsize=15, fontweight = "bold")
plt.savefig("../figures/AE/AE_ROC.pdf", bbox_inches="tight")
plt.show()