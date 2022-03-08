import numpy as np
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers


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
history = hypermodel.fit(X, Y, validation_split=0.2 )
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

#Find best score
hypermodel.fit(X, Y, epochs=best_epoch, validation_split=0.2)
eval_result = hypermodel.evaluate(X, Y)
print(f"Validation loss, Validation accuracy : {eval_result[0]:.2f} , {eval_result[1]*100:.2f}%")
