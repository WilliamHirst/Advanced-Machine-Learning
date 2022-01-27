import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input
import pandas as pd 
    
 
class SupervisedSolver:
    def __init__(self):
        self.dataframe = np.load("../Data/data.npy")
        self.length_frame = len(self.dataframe)
        
    def setup_model(self):
        return 0
    
    def train(self):
        return 0
    
    def predict(self): 
        return 0

    def save_model(self, name):
        self.model.save(f"tf_models/model_{name}.h5")

    def load_model(self, name):
        self.model = tf.keras.models.load_model(f"tf_models/model_{name}.h5")

    def save_checkpoint(self, checkpoint_name):
        self.model.save_weights(f"checkpoints/{checkpoint_name}")

    def load_from_checkpoint(self, checkpoint_name):
        self.model.load_weights(f"tf_checkpoints/{checkpoint_name}")
        
    

if __name__ == "__main__":
    # Place tensors on the CPU
    with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
        ML = SupervisedSolver()
        print(ML.length_frame)
    