import numpy as np
import pandas as pd
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers, Model  # If we need regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# import xgboost as xgb


class UnsupervisedSolver:
    def __init__(self, X_train, y_train=None, X_val=None, y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.nrFeatures = len(X_train[0])

    def getModel(self, method, epochs=100, batchSize=100, depth=10):
        m = M.Model(method, self.nrFeatures, epochs, batchSize, depth)
        self.tool = m.tool
        self.fit = m.fit
        self.model_predict = m.predict
        self.model = m()

    def train(self):
        self.trainModel = self.fit(self.X_train, X_val)

    def predict(self, X_val, y_val):
        prediction = self.model_predict(X_val)
        
        threshold = self.findThreshold()

        errors = tf.keras.losses.msle(prediction, X_val)
        anom_mask = pd.Series(errors) > threshold 
        new_pred = anom_mask.map(lambda x: 1 if x == True else 0)

        new_pred = new_pred.to_numpy()
        print(f"Accuracy: {accuracy_score(new_pred, y_val)*100}%")

        
        
    def findThreshold(self):
        reconstruct = self.model_predict(self.X_train)
        recon_error = tf.keras.losses.msle(reconstruct, self.X_train)
        threshold = np.mean(recon_error.numpy()) + np.std(recon_error.numpy())    
        
        return threshold

    



    def pairwise_distance(self, X, Y):
        euq_sq = np.square(X - Y) 
        return np.sqrt(np.sum(euq_sq, axis=1)).ravel()
  

if __name__ == "__main__":
    import time
    tf.random.set_seed(1)

    t0 = time.time()
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    #DH.removeBadFeatures(40)
    DH.fillWithImputer()
    DH.standardScale()
    
    #DH.removeOutliers(6)
    #DH.kMeansClustering()
    DH.split()
    X_train, X_val, y_train, y_val = DH(include_test=True)
    
    #print(y_val, np.shape(y_val))

    with tf.device("/CPU:0"):
        US = UnsupervisedSolver(X_train, y_train, X_val, y_val)
        US.getModel("autoencoder")
        US.train()
        US.predict(X_val, y_val)
        