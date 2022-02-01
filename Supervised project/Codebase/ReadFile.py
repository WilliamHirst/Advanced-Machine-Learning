import csv
import numpy as np

def standard_scale(dataset):

    avg_data = np.nanmean(dataset, axis = 1)
    std_data = np.nanstd(dataset, axis = 1)
    for i in range(len(dataset[0])):
        dataset[:,i] = (dataset[:,i] - avg_data[i])/(std_data)    
    return dataset

file = open("../Data/training.csv")



csvreader = csv.reader(file)
features = []
targets = []

next(csvreader)
for row in csvreader:
    event = []
    for element in row:
        try:
            event.append(float(element))
        except ValueError:
            if element == "b":
                targets.append(0)
            else:
                targets.append(1)
    features.append(event)


features = np.asarray(features)
features = np.where(features == -999.0, np.NaN, features)
features = standard_scale(features)
targets = np.asarray(targets)



np.save("../Data/featuresTrain.npy", features)
np.save("../Data/targetsTrain.npy", targets)


file = open("../Data/test.csv")
csvreader = csv.reader(file)
features = []
targets = []
next(csvreader)

for row in csvreader:
    event = []
    for element in row:
        event.append(float(element))
        
    features.append(event)

features = np.asarray(features)
features = np.where(features == -999.0, np.NaN, features)
features = standard_scale(features)

np.save("../Data/featuresTest.npy", features)


