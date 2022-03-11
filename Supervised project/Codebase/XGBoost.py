from xgboost import XGBClassifier
from DataHandler import DataHandler
from Functions import *
from joblib import dump, load
import os


X_test = np.load("../Data/featuresTest.npy")
weights = np.load("../Data/weights.npy")

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")

X , y = DH(include_test=False)

sum_wpos = sum(weights[i] for i in range(len(X)) if y[i] == 1.0)
sum_wneg = sum(weights[i] for i in range(len(X)) if y[i] == 0.0)
DH.split()


X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodel_ams.joblib")
model = load(filename)


model.scale_pos_weight =  sum_wneg/sum_wpos 

score = model.score(X_val,y_val)# scoring = 'ams@0.15')



predict = model.predict(X_test)
prob = model.predict_proba(X_test)

ams = AMS(predict, prob)



print(f"\nValidation accuracy : {score*100:.2f}")
print(f"\nAMS : {ams:.2f}")