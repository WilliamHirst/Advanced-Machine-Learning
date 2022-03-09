from xgboost import XGBClassifier
from DataHandler import DataHandler
from joblib import dump, load
import os


DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodel.joblib")
model = load(filename)

score = model.score(X_val,y_val)

print(f"\nValidation accuracy : {score*100:.2f}")
