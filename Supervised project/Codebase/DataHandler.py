import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, features = None, targets = None):
        self.X_train = features
        self.y_train = targets
        self.nrEvents = len(self.y_train)
        self.nrFeatures = len(features[0])

    def __call__(self, include_test = False):
        if include_test:
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            return self.X_train, self.y_train


    def removeBadFeatures(self, procentage=30):
        """
        Removes all bad features.
        """
        self.findBadFeatures(procentage)
        self.X_train = np.delete(self.X_train, self.badFeatures, 1)
        self.nrFeatures = len(self.X_train[0])

    def findBadFeatures(self, procentage):
        """
        Finds all features with a certain precentage of nan values.
        """
        badFeatures = []
        for i in range(self.nrFeatures):
            nrOfNan = np.sum(np.where(np.isnan(self.X_train[:, i]), 1, 0))
            featuresProcentage = nrOfNan / self.nrEvents * 100
            if featuresProcentage >= procentage:
                badFeatures.append(i)
        self.badFeatures = np.asarray(badFeatures)

    def standardScale(self):
        arr = self.X_train
        avg_data = np.nanmean(arr, axis=1)
        std_data = np.nanstd(arr, axis=1)
        for i in range(len(arr[0])):
            arr[:, i] = (arr[:, i] - avg_data[i]) / (std_data)
            
        self.X_train = arr

    def setNanToMean(self):
        """
        Fills all nan values with the avarage value of the certain feature.
        """
        arr = self.X_train
        for i in range(self.nrFeatures):
            arr[:, i] = np.where(
                np.isnan(arr[:, i]), np.nanmean(arr[:, i]), arr[:, i]
            )
        self.X_train = arr

    def removeOutliers(self, sigma):
        arr = self.X_train
        std = np.nanstd(arr, axis=0)
        mean = np.nanmean(arr, axis=0)

        check = np.abs(arr - mean)
        isLess = np.less_equal(check, sigma * std)
        self.X_train = arr[np.all(isLess, axis=1)]
        self.y_train = self.y_train[np.all(isLess, axis=1)]
        print(
            f"#Events have been changed from {self.nrEvents} to {len(self.X_train)}"
        )
        
    def split(self, t_size = 0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X_train, self.y_train, test_size=t_size
        )

    
    def fixDataset(self):
        #from sklearn.impute import SimpleImputer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        impute_mean = IterativeImputer(missing_values=np.NaN, 
                                       initial_strategy="mean", 
                                       max_iter=1, 
                                       random_state=0)
        impute_mean.fit(self.X_train)
        
        IterativeImputer(random_state=0)
        self.X_train = impute_mean.transform(self.X_train)
        
        
        