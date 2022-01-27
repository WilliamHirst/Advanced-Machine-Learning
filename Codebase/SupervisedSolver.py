import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input
import pandas as pd 
    
 
class SupervisedSolver:
    def __init__(self, datafile):
        self.datafile = datafile
        self.dataframe = pd.read_csv(self.datafile)
        self.length_frame = len(self.dataframe)
        
    

if __name__ == "__main__":
    ss = SupervisedSolver("../Data/training.csv")
    print(ss.length_frame)