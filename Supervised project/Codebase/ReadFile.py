import csv
import numpy as np



file = open("../Data/training.csv")



csvreader = csv.reader(file)
features = []
targets = []

column_names = file.readline().split(",")
column_names[-1] = column_names[-1].replace("\n", "") 


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
targets = np.asarray(targets)
weights = features[:, -1]
column_names = np.asarray(column_names)



np.save("../Data/rawFeatures_TR.npy", features[:, 1:-1])
np.save("../Data/rawTargets_TR.npy", targets)
np.save("../Data/column_names.npy", column_names[1:-1])
np.save("../Data/weights.npy", weights)

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

np.save("../Data/featuresTest.npy", features[:, 1:])


