import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input

class SupervisedSolver:
    def __init__(self, datafile):
        data = reafFile(datafile)


    def readFile():
        file = open(datafile, 'r')
        print(len(file.readline().split(",")))
        return data

if __name__ == "__main__":
    ss = SupervisedSolver("../Data/training.csv")
