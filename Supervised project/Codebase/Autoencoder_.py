import numpy as np
import pandas as pd
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import plot_set
from sklearn.metrics import accuracy_score

# for custom activation function
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})


# Data handling
print("Preparing data...")
tf.random.set_seed(1)
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")

#DH.removeBadFeatures()
DH.fillWithImputer()
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
        X_train, X_train, epochs=40, batch_size=4000, validation_data=(X_val, X_val)
    )
acc_hist = history.history["val_mse"]
loss_hist = history.history["val_loss"]
best_epoch = acc_hist.index(max(acc_hist))

print(
    f"Validation loss, Validation mse : {loss_hist[best_epoch]:.2f} , {acc_hist[best_epoch]:.2f}%"
)

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
plt.savefig("../figures/AE_loss.pdf", bbox_inches="tight")
plt.show()



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


#import seaborn as sns

#sns.set_style('darkgrid')
#sns.distplot(errorsback, kde = True)
#sns.distplot(errorssig, kde = True)
#plt.legend()
#plt.show()



n, bins, patches = plt.hist(errorsback, 100, density=True, facecolor='b', alpha=1, label="Background")
n, bins, patches = plt.hist(errorssig, 100, density=True, facecolor='r', alpha=0.6, label="Signal")
fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.legend()
plt.savefig("../figures/AE_hist.pdf", bbox_inches="tight")
plt.show()

#anom_mask = pd.Series(errors) > threshold 
#new_pred = anom_mask.map(lambda x: 1 if x == True else 0)

#new_pred = new_pred.to_numpy()
#print(f"Accuracy: {accuracy_score(new_pred, y_val)*100}%")

        
