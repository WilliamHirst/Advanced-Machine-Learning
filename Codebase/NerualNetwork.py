import numpy as np
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import scikitplot as skplt
import plot_set
from Functions import *


# Data handling
print("Preparing data...")
tf.random.set_seed(1)
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,0].astype(int)
X_test = X_test[:,1:]

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
nr_train = DH.nrEvents
DH.X_train = np.concatenate((DH.X_train, X_test), axis=0)
DH.setNanToMean()
DH.standardScale()
X, Y = DH(include_test=False)

DH.X_train = X[: nr_train,:]
X_test = X[nr_train:,:]
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

# Get optimal model through previous gridsearch
print("Fetching optimal parameters...")
name = "hypermodel"
epochs = 500
hypermodel = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")
tf.keras.utils.plot_model(hypermodel, to_file="../figures/Neural_network/model_plot.png", show_shapes=True, show_layer_names=True, expand_nested=True)
exit()
# Train to find best epoch
print("Training model.")

history = hypermodel.fit(X_train, y_train, epochs=epochs, batch_size=4000, validation_data=(X_val, y_val))

acc_hist = history.history["val_accuracy"]
loss_hist = history.history["val_loss"]
best_epoch = acc_hist.index(max(acc_hist))
print(f"Validation loss, Validation accuracy : {loss_hist[best_epoch]:.2f} , {acc_hist[best_epoch]*100:.2f}%, best epoch {best_epoch}")


fig, ax1 = plt.subplots(num=0, dpi=80, facecolor='w', edgecolor='k')
fig.suptitle("Neural network history", fontsize=16)
color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
ax1.set_xlabel(r"#$Epochs$", fontsize=16)
ax1.set_ylabel(r"$Accuracy$", fontsize=16, color=color)
ax1.plot(history.history["val_accuracy"], color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
ax2.set_ylabel(r"$Loss$", color=color, fontsize=16)  # we already handled the x-label with ax1
ax2.plot(history.history["val_loss"], color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("../figures/Neural_network/NN_hist.pdf", bbox_inches="tight")
plt.show()



proba = hypermodel.predict(X_val)
probas = np.concatenate((1-proba,proba),axis=1)
proba = proba.ravel()
s = proba[np.where(y_val == 1)]
b = proba[np.where(y_val == 0)]

sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))
x_start = np.mean(b)
x_end =np.mean(s)
y_start = 3

binsize = 150

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                     label = "Background", density=True)
n_s, bins_s, patches_s = plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, 
                                     label = "Signal",  density=True)

median_s = bins_s[np.where(n_s==np.max(n_s))][0]
median_b = bins_b[np.where(n_b==np.max(n_b))][0]
plt.axvline(x=x_start,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
plt.axvline(x=x_end,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.axvline(x=median_s,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
plt.axvline(x=median_b,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#Events", fontsize=15)
plt.title("Neural network output distribution", fontsize=15, fontweight = "bold")
plt.legend(fontsize = 16, loc = "upper right")
plt.annotate("", xy=(x_start,y_start),
            xytext=(x_end,y_start),verticalalignment="center",
            arrowprops={'arrowstyle': '<->', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}",
                xy=(((x_start+x_end)/2), y_start+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.annotate("", xy=(median_b,y_start*2),
            xytext=(median_s,y_start*2),verticalalignment="center",
            arrowprops={'arrowstyle': '<->', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid s_m-b_m\mid$"+f" = {abs(median_b-median_s):.2f}",
                xy=(((median_b+median_s)/2), y_start*2+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.savefig("../figures/Neural_network/NN_output.pdf", bbox_inches="tight")
plt.show()


skplt.metrics.plot_roc(y_val, probas)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("Neural network: ROC curve", fontsize=15, fontweight = "bold")
plt.savefig("../figures/Neural_network/NN_ROC.pdf", bbox_inches="tight")
plt.show()

exit()
#Train network up to best epoch.
hypermodel.fit(X_train, y_train, epochs=best_epoch, batch_size=4000, validation_split=0.2)
"""
Test data
"""
threshold = 0.85
proba = hypermodel.predict(X_test).ravel()
name = '../Data/NN_test_pred.csv'
write_to_csv(EventID, proba, threshold, name)
