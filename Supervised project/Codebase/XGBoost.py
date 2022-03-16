from xgboost import XGBClassifier
from DataHandler import DataHandler
from joblib import dump, load
import numpy as np
from Functions import *
import os

threshold = 0.85
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodel.joblib")
model = load(filename)

score = model.score(X_val,y_val)

X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,1]


proba = model.predict_proba(X_test[:,1:])[:,1]
write_to_csv(id, proba, threshold)
print(model.predict(X_test[:,1:]))

exit()
print(f"\nValidation accuracy : {score*100:.2f}")
