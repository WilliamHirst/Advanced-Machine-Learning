import csv
import numpy as np

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
targets = np.asarray(targets)

np.save("../Data/featuresTrain.npy", features)
np.save("../Data/targetsTrain.npy", targets)