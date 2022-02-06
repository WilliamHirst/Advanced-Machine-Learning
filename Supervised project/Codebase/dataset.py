import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, dataset_train, dataset_test): 
        self.data_train = dataset_train
        self.data_test = dataset_test
        
        # Make pandas dataframe
        
        self.train_df = pd.DataFrame(self.data_train)
        self.test_df = pd.DataFrame(self.data_test)
        
        
    def __call__(self):
        return 0
    
    def corrolationPlot(self): 
        """
        Corrolation plot for all the features in the dataset
        """
        pass
    
    
    
if __name__ == "__main__":
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    
    data_obj = Dataset(featuresTrain, targetsTrain)
    
    