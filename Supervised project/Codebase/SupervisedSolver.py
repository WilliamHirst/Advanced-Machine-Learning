import numpy as np
import Model as M
from DataHandler import DataHandler
import tensorflow as tf
from tensorflow.keras import optimizers  # If we need regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb


class SupervisedSolver:
    def __init__(self, features, targets):
        self.featuresTrain = features
        self.targetsTrain = targets
        self.nrEvents = len(self.targetsTrain)
        self.nrFeatures = len(features[0])

    def getModel(self, method, epochs=100, batchSize=100, depth=10):
        m = M.Model(method, self.nrFeatures, epochs, batchSize, depth)
        self.tool = m.tool
        self.fit = m.fit
        self.model_predict = m.predict
        self.model = m()

    def train(self):
        self.trainModel = self.fit(self.featuresTrain, self.targetsTrain)
        return 0

    def predict(self, featuresTest, targetTest):

        predict = self.model_predict(featuresTest)

        self.nr_of_events = len(predict)
        self.signal_nr = np.sum(predict)
        self.background_nr = self.nr_of_events - self.signal_nr

        print(
            "Background: {} -- Signal: {} -- Total events {}".format(
                self.background_nr, self.signal_nr, self.nr_of_events
            )
        )
        self.acc = (
            np.sum(np.equal(predict, targetTest.ravel())) / self.nr_of_events * 100
        )
        print(f"Accuracy: {self.acc:.1f}%")

        self.missed_events = predict[np.where(predict != targetTest.ravel())[0]]
        self.number_wrong_prediction = np.sum(self.missed_events)
        self.number_missed = len(self.missed_events) - self.number_wrong_prediction

        print(
            "Number of wrong classified: {} \nNumber of missed events: {}".format(
                self.number_wrong_prediction, self.number_missed
            )
        )

    """
    TF-MODEL
    """

    def saveModel(self, name):
        self.model.save(f"../tf_models/model_{name}.h5")

    def loadModel(self, name):
        self.model = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")

    def saveCheckpoint(self, checkpoint_name):
        self.model.save_weights(f"checkpoints/{checkpoint_name}")

    def loadFromCheckpoint(self, checkpoint_name):
        self.model.load_weights(f"tf_checkpoints/{checkpoint_name}")

    """
    FUNCTIONS
    """

    def significantEvents(self, s, b):
        s, b = self.predict()
        mu_b = 0
        n = s + b
        gauss_significant_discovery = (n - mu_b) / np.sqrt(mu_b)
        return gauss_significant_discovery

    def AMS(self, s, b):
        b_reg = 10
        ams = np.sqrt(2 * ((s + b + b_reg) * np.ln(1 + (s) / (b + b_reg))) - s)
        return ams

    def plotModel(self):
        # xgb.plot_tree(SS.model, num_trees=1)
        xgb.plot_importance(self.model)
        # model.get_booster().feature_names = ["DER mass MMC", "DER mass transverse met lep", "DER mass vis", "list"]
        plt.show()

    def plotAccuracy(self):
        """
        Plots the history of the accuracy of the predictions.
        """
        plt.plot(self.trainModel.history["accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def plotLoss(self):
        """
        Plots the history of the accuracy of the predictions.
        """
        plt.plot(self.trainModel.history["loss"])
        plt.title("Model Accuracy")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def importDataSet(dataSet):
        featuresTrain = np.load("../Data/featuresTrain.npy")
        targetsTrain = np.load("../Data/targetsTrain.npy")



    

    def callModelSave(self):
        state = True
        while state == True:
            answ = input("Do you want to save model? (y/n) ")
            if answ == "y":
                name = input("name: ")
                self.saveModel(name)
                state = False
                print("Model saved")
            elif answ == "n":
                state = False
                print("Model not saved")


if __name__ == "__main__":
    import time

    # Load data from npy storage. Must have run ReadFile.py first
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    # featuresTest = np.load("../Data/featuresTest.npy")

   

    # Place tensors on the CPU
    # with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
    t0 = time.time()
    DH = DataHandler(featuresTrain[:, 1:-1], targetsTrain)

    # DH.removeBadFeatures(30)
    DH.setNanToMean(DH.X_train)
    DH.standardScale(DH.X_train)
    # DH.removeOutliers(4)
    DH.split()

    X_train, X_test, y_train, y_test = DH(include_test = True)

    """
    Model types: neuralNetwork -- convNeuralNetwork -- GRU_NN -- decisionTree -- xGBoost 
    """

    SS = SupervisedSolver(X_train, y_train)


    # with tf.device("/GPU:0"):  # Write '/GPU:0' for large networks

    SS.getModel("neuralNetwork", epochs=5, batchSize=10000, depth=3)
    SS.train()
    SS.plotAccuracy()
    SS.plotLoss()

    SS.predict(X_test, y_test)

    t1 = time.time()
    total_n = t1 - t0
    print("{:.2f}s".format(total_n))
    # SS.plotModel()

    # pip install pywhatkit
    if SS.acc >= 85:
        import pywhatkit

        songOrArtist = "celebration"
        pywhatkit.playonyt(songOrArtist)

    if SS.tool == "tf":
        SS.callModelSave()

