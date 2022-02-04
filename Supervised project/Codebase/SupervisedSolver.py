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
        self.fit(self.featuresTrain, self.targetsTrain)
        return 0
    
    def predict(self, featuresTest,targetTest): 
        if self.tool == "tf":
            predict = np.around(self.model(featuresTest).numpy().ravel())
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
    
    def plotModel(self):
        #xgb.plot_tree(SS.model, num_trees=1)
        xgb.plot_importance(self.model)
        plt.show()
        
    

if __name__ == "__main__":
    import time
    import xgboost as xgb

    # Load data from npy storage. Must have run ReadFile.py first
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    #featuresTest = np.load("../Data/featuresTest.npy")
    X_train, X_test, y_train, y_test = train_test_split(featuresTrain[:,1:-1], targetsTrain, test_size = 0.2, random_state = 0)

    """
    Model types: neuralNetwork -- decisionTree -- xGBoost
    """
    # Place tensors on the CPU
    #with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
    t0 = time.time()

    SS = SupervisedSolver(X_train, y_train)
    SS.get_model("xGBoost", depth = 10)
    SS.train()
    SS.predict(X_test,y_test)
    
    
    t1 = time.time()
    total_n = t1-t0
    print("{:.2f}s".format(total_n))
    SS.plotModel()
    