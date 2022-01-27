import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input

class SupervisedSolver:
    def __init__(self, datafile):
        self.datafile = datafile

        file = open(self.datafile, 'r')
        print(len(file.readline().split(",")))


if __name__ == "__main__":
    ss = SupervisedSolver("../Data/training.csv")
