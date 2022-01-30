import csv
import numpy as np
import pandas as pd

def standard_scale(dataset):
    avg_data = np.mean(dataset)
    std_data = np.std(dataset)
    
    scaled_dataset = (dataset - avg_data)/(std_data)
    
    return scaled_dataset

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

features = standard_scale(np.asarray(features))
targets = standard_scale(np.asarray(targets))

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

features = min_max_scale(np.asarray(features))

np.save("../Data/featuresTest.npy", features)


