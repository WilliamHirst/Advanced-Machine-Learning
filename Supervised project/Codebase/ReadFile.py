import csv
import numpy as np


def min_max_scale(dataset):
    min_data = np.min(dataset)
    max_data = np.max(dataset)
    
    scaled_dataset = (dataset - min_data)/(max_data-min_data)
    
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

features = min_max_scale(np.asarray(features))
targets = min_max_scale(np.asarray(targets))


np.save("../Data/featuresTrain.npy", features)
np.save("../Data/targetsTrain.npy", targets)


