import numpy as np
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import plot_set


#Data handling
print("Preparing data...")
tf.random.set_seed(1)
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.fillWithImputer()
DH.standardScale()
X, Y = DH(include_test=False)

#Get optimal model through previous gridsearch
print("Fetching optimal parameters...")
name = "hypermodel"
hypermodel = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")

#Train to find best epoch
print("Training model.")
history = hypermodel.fit(X, Y, epochs = 5, batch_size=4000,validation_split=0.2 )
acc_hist = history.history['val_accuracy']
loss_hist = history.history['val_loss']
best_epoch = acc_hist.index(max(acc_hist))

print(f"Validation loss, Validation accuracy : {loss_hist[best_epoch]:.2f} , {acc_hist[best_epoch]*100:.2f}%")


fig, ax1 = plt.subplots()

color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
ax1.set_xlabel('#Epochs')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history["val_accuracy"],color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(history.history["val_loss"],color = color)


ax2.tick_params(axis='y', labelcolor=color )

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

