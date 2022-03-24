import numpy as np

class DataHandler:
    def __init__(self, features = None, targets = None):
        if isinstance(features, str):
            self.importDataSet(features, targets)
        else:     
            self.X_train = features
            self.y_train = targets

        self.nrEvents = len(self.X_train[:,0])
        self.nrFeatures = len(self.X_train[0])
        
        self.checksplit = 0

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

    def removeOutliers(self, sigma):
        arr = self.nanToMean(self.X_train)
        std = np.nanstd(arr, axis=0)
        mean = np.nanmean(arr, axis=0)
        check = np.abs(arr - mean)
        isLess = np.less_equal(check, sigma * std)
        
        self.X_train = arr[np.all(isLess, axis=1)]
        self.y_train = self.y_train[np.all(isLess, axis=1)]
        print(f"#Events have been changed from {self.nrEvents} to {len(self.X_train)}")
        self.nrEvents = len(self.X_train )
        
    def split(self, t_size = 0.2):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X_train, self.y_train, test_size=t_size, random_state = 0)
        
        self.checksplit = 1
    
    def setNanToMean(self):
        """
        Fills all nan values in train with the avarage value of the certain feature.
        """
        self.nanToMean(self.X_train)

    def nanToMean(self, arr):
        for i in range(self.nrFeatures):
            arr[:, i] = np.where(
                np.isnan(arr[:, i]), np.nanmean(arr[:, i]), arr[:, i]
            )
        return arr
        
    
    def fillWithImputer(self):
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
        
    def importDataSet(self, train, test):
        self.X_train = np.load(f"../Data/{train}")
        if test != None:
            self.y_train = np.load(f"../Data/{test}")

    def saveDataSet(self, featureName, targetName ):
        np.save(f"../Data/{featureName}", self.X_train)
        np.save(f"../Data/{targetName}", self.y_train)
        print(f"New dataset saved as {featureName} and {targetName} in folder: Data")

    def kMeansClustering(self):
        from sklearn.cluster import KMeans
        Kmean = KMeans(n_clusters=2)
        Kmean.fit(self.X_train)    
        label_likelyhood = Kmean.labels_
        print(Kmean.score(self.X_train))
        self.acc = (
            np.sum(np.equal(label_likelyhood, self.y_train)) / len(self.y_train) * 100
        )
        print(f"Accuracy: {self.acc:.1f}%")
        
        np.append(self.X_train, label_likelyhood.reshape(len(label_likelyhood), 1), axis=1)

    def AE_prep(self, whole_split=False):
        if self.checksplit == 0:
            self.split()
        

    
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()
        
        index_background = np.where(y_train == 0)[0]
        X_background = X_train[index_background, :]
        y_background = y_train[index_background]

        if whole_split == False:
            return X_background, y_background, X_test, y_test
        else:
            index_signal_test = np.where(y_test == 1)[0]
            index_background_test = np.where(y_test == 0)[0]
            X_signal_test = X_test[index_signal_test,:]
            X_background_test = X_test[index_background_test, :]

            return X_background, y_background, X_test, y_test, X_background_test, X_signal_test

