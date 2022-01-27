import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input

class SupervisedSolver:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.readFile(self.datafile)


    def readFile(self, datafile):
        file = open(datafile, 'r')
        self.number_of_categories = len(file.readline().split(","))
        print(self.number_of_categories)
        
        return 0

if __name__ == "__main__":
    ss = SupervisedSolver("../Data/training.csv")
