import numpy as np
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers  # If we need regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import xgboost as xgb


class UnsupervisedSolver:
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
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
        self.trainModel = self.fit(self.X_train, y_train)

    def predict(self, X_all):
        prediction = self.model_predict(X_all)
        print(prediction, np.shape(prediction))


if __name__ == "__main__":
    import time

    t0 = time.time()
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    # DH.removeBadFeatures(40)
    DH.fillWithImputer()
    DH.standardScale()
    # X_background, X_all, y_all = DH.AE_prep()
    # DH.removeOutliers(6)
    # DH.kMeansClustering()
    DH.split()

    X_train, X_val, y_train, y_val = DH(include_test=True)
    print(y_val, np.shape(y_val))

    US = UnsupervisedSolver(X_train, y_train, X_val, y_val)
    US.getModel("svm")
    US.train()

    predict = US.predict(X_val)

    t1 = time.time()
    total_n = t1 - t0
    print("{:.2f}s".format(total_n))

    acc = np.sum(np.equal(predict, y_val.ravel())) / len(y_val) * 100
    print(f"Accuracy: {acc:.1f}%")
