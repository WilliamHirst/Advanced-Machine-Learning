import pandas as pd 
import csv
import numpy as np

file = open("../Data/training.csv")
csvreader = csv.reader(file)
data = []
next(csvreader)
for row in csvreader:
    event = []
    for element in row:
        try:
            event.append(float(element))
        except ValueError:
            event.append(element)
    data.append(event)

data = np.asarray(data)

np.save("../Data/data.npy", data)