import numpy as np
import Model as M 
#import matplotlib.pyplt as plt #for plotting
import tensorflow as tf
from tensorflow.keras import optimizers#, regularizers #If we need regularizers

    
 
class SupervisedSolver:
    def __init__(self, features, targets):
        self.featuresTrain = features
        self.targetsTrain = targets
        self.nrEvents = len(self.targetsTrain)
        self.nrFeatures = len(features[0])

    def get_model(self, method, epochs = 100, batchSize = 100):
        m = M.Model(method, self.nrFeatures, epochs, batchSize)
        self.tool = m.tool
        self.fit = m.fit
        self.model = m()
            
    def train(self):
        self.fit(self.featuresTrain, self.targetsTrain)
        return 0
    
    def predict(self, featuresTest): 
        if self.tool == "tf":
            predict = np.around(self.model(featuresTest).numpy().ravel())
        else: 
            predict = np.around(self.model.predict(featuresTest).ravel())
    
        print(f"Background: {len(predict)-np.sum(predict)} -- Signal: {np.sum(predict)} -- Total events {len(predict)}" )
    
    def accuracy(self, featuresTest, targetTest):
        if self.tool == "tf":
            predict = self.model(featuresTest).numpy().ravel()
        else: 
            predict = self.model.predict(featuresTest).ravel()
        ac = np.sum(np.around(targetTest) == np.around(predict))/self.nrEvents
        return ac

    def save_model(self, name):
        self.model.save(f"tf_models/model_{name}.h5")

    def load_model(self, name):
        self.model = tf.keras.models.load_model(f"tf_models/model_{name}.h5")

    def save_checkpoint(self, checkpoint_name):
        self.model.save_weights(f"checkpoints/{checkpoint_name}")

    def load_from_checkpoint(self, checkpoint_name):
        self.model.load_weights(f"tf_checkpoints/{checkpoint_name}")
        
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
        
    

if __name__ == "__main__":
    
    # Load data from npy storage. Must have run ReadFile.py first
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    featuresTest = np.load("../Data/featuresTest.npy")
    
    """
    Model types: neuralNetwork -- decisionTree
    """
    # Place tensors on the CPU
    with tf.device("/GPU:0"):  # Write '/GPU:0' for large networks
        SS = SupervisedSolver(featuresTrain[:,:-1], targetsTrain)
        SS.get_model("neuralNetwork", 20, 50000)
        SS.train()
        SS.predict(featuresTest)