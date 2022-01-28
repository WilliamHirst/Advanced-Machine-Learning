import numpy as np
import Model as M 
#import matplotlib.pyplot as plt #for plotting
import tensorflow as tf
from tensorflow.keras import optimizers#, regularizers #If we need regularizers
#from sklearn.model_selection import train_test_split

    
 
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
        self.fit(
                    self.featuresTrain,
                    self.targetsTrain,
                )
        return 0
    
    def predict(self): 
        return 0
    
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
        
    

if __name__ == "__main__":
    
    # Load data from npy storage. Must have run ReadFile.py first
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    
    """
    Model types: neuralNetwork -- decisionTree
    """
    # Place tensors on the CPU
    with tf.device("/GPU:0"):  # Write '/GPU:0' for large networks
        SS = SupervisedSolver(featuresTrain, targetsTrain)
        SS.get_model("decisionTree")
        SS.train()
        print(f"{SS.accuracy(featuresTrain, targetsTrain)*100:.1f}%")