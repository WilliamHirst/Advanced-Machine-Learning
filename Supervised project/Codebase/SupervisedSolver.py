import numpy as np
import Model as M 
import tensorflow as tf
from tensorflow.keras import optimizers #If we need regularizers
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 


class SupervisedSolver:
    def __init__(self, features, targets):
        self.featuresTrain = features
        self.targetsTrain = targets
        self.nrEvents = len(self.targetsTrain)
        self.nrFeatures = len(features[0])

    def get_model(self, method, epochs = 100, batchSize = 100, depth = 10):
        m = M.Model(method, self.nrFeatures, epochs, batchSize, depth)
        self.tool = m.tool
        self.fit = m.fit
        self.model = m()
            
    def train(self):
        self.trainModel =  self.fit(self.featuresTrain, self.targetsTrain)
        return 0
    
    def predict(self, featuresTest,targetTest): 
        if featuresTest.shape[1] != self.featuresTrain.shape[1]:
            featuresTest = np.delete(featuresTest,self.badFeatures,1)

        if self.tool == "tf":
            rough_predict = np.around(self.model(featuresTest).numpy().ravel())
            predict = np.around(rough_predict)

        else: 
            predict = np.around(self.model.predict(featuresTest).ravel())

        print(f"Background: {len(predict)-np.sum(predict)} -- Signal: {np.sum(predict)} -- Total events {len(predict)}" )
        print(f"Accuracy: {np.sum(predict==targetTest)/len(predict)*100:.1f}%")
    
    def accuracy(self, featuresTest, targetTest):
        if self.tool == "tf":
            predict = self.model(featuresTest).numpy().ravel()
        else: 
            predict = self.model.predict(featuresTest).ravel()
        ac = np.sum(np.around(targetTest) == np.around(predict))/self.nrEvents
        return ac


    """
    TF-MODEL
    """

    def save_model(self, name):
        self.model.save(f"tf_models/model_{name}.h5")

    def load_model(self, name):
        self.model = tf.keras.models.load_model(f"tf_models/model_{name}.h5")

    def save_checkpoint(self, checkpoint_name):
        self.model.save_weights(f"checkpoints/{checkpoint_name}")

    def load_from_checkpoint(self, checkpoint_name):
        self.model.load_weights(f"tf_checkpoints/{checkpoint_name}")
    
    """
    FUNCTIONS
    """
    
    def removeBadFeatures(self, procentage = 30):
        """
        Removes all bad features. 
        """
        self.findBadFeatures(procentage)
        self.featuresTrain = np.delete(self.featuresTrain, self.badFeatures, 1) 
        self.nrFeatures = len(self.featuresTrain[0])


    def findBadFeatures(self,procentage):
        """
        Finds all features with a certain precentage of nan values.
        """
        badFeatures = []
        for i in range(self.nrFeatures):
            nrOfNan = np.sum(np.where(np.isnan(self.featuresTrain[:,i]), 1, 0))
            featuresProcentage = nrOfNan/self.nrEvents * 100
            if featuresProcentage >= procentage:   
                badFeatures.append(i) 
        self.badFeatures = np.asarray(badFeatures)


    def significant_events(self, s, b):
        s, b = self.predict()
        mu_b = 0
        n = s + b 
        gauss_significant_discovery = (n-mu_b)/np.sqrt(mu_b)
        return gauss_significant_discovery
    
    def AMS(self, s, b):
        b_reg = 10
        ams = np.sqrt( 2*((s+b+b_reg)*np.ln(1 + (s)/(b + b_reg))) - s )
        return ams
    
    def plotModel(self):
        #xgb.plot_tree(SS.model, num_trees=1)
        xgb.plot_importance(self.model)
        #model.get_booster().feature_names = ["DER mass MMC", "DER mass transverse met lep", "DER mass vis", "list"]
        plt.show()

    def plotAccuracy(self):
        """
        Plots the history of the accuracy of the predictions.
        """
        plt.plot(self.trainModel.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def reshapeDataset(self, X, y): 
        X = X.reshape(X_test.shape[0], X_test.shape[1], 1)
        y = y.reshape(y.shape[0], 1, 1)
        
        return X, y
    def standard_scale(self, *args):
        avg_data = np.nanmean(args[0], axis = 1)
        std_data = np.nanstd(args[0], axis = 1)
        for i in range(len(args[0][0])):
            args[0][:,i] = (args[0][:,i] - avg_data[i])/(std_data)    
        return args[0]

    def setNanToMean(self, *args):
        """
        Fills all nan values with the avarage value of the certain feature.
        """
        for i in range(self.nrFeatures):
                args[0][:,i] = np.where(np.isnan(args[0][:,i]), 
                                                np.nanmean(args[0][:,i]),  
                                                args[0][:,i] )
        return args[0]

        

if __name__ == "__main__":
    import time
    import xgboost as xgb

    # Load data from npy storage. Must have run ReadFile.py first
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    #featuresTest = np.load("../Data/featuresTest.npy")
    X_train, X_test, y_train, y_test = train_test_split(featuresTrain[:,1:-1], targetsTrain, test_size = 0.15)

    """
    Model types: neuralNetwork -- decisionTree -- xGBoost -- convNeuralNetwork
    """
    
    
    # Place tensors on the CPU
    with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
        t0 = time.time()

        SS = SupervisedSolver(X_train, y_train)   

        SS.removeBadFeatures(30)
        SS.setNanToMean(SS.featuresTrain)

        SS.get_model("convNeuralNetwork",epochs = 2, batchSize= 4000, depth = 10)
        SS.standard_scale(SS.featuresTrain)

        SS.train()
        SS.plotAccuracy()

        
        X_test, y_test = SS.reshapeDataset(X_test, y_test)

        SS.setNanToMean(X_test)
        SS.standard_scale(X_test)
        SS.predict(X_test, y_test)

        t1 = time.time()
        total_n = t1-t0
        print("{:.2f}s".format(total_n))
        #SS.plotModel()
    