from xgboost import XGBClassifier
from DataHandler import DataHandler
from joblib import dump, load
import numpy as np
from Functions import *
import matplotlib.pyplot as plt
import os

threshold = 0.85
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodel.joblib")
model = load(filename)

score = model.score(X_val,y_val)





proba = model.predict_proba(X_val)[:,1].ravel()

s = proba[np.where(y_val == 1)]
b = proba[np.where(y_val == 0)]

sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))/sigma

binsize = 100
plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",label = "Background", density=True)
plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, label = "Signal", density=True)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#-of-events", fontsize=15)
plt.title("XGBoost output distrubution: " 
            + r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
            + f" = {diff:.2f}" + r"$\sigma_b$")
plt.legend(fontsize = 16)
plt.annotate(r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}" + r"$\sigma_b$", xy=(0.3,6),
            xytext=(0.3,6),verticalalignment="center",
            arrowprops={'arrowstyle': '|-|', 'lw': 2}, va='center')
plt.show()

exit()
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,0].astype(int)
proba = model.predict_proba(X_test[:,1:])[:,1]
name = '../Data/xgboost_test_pred.csv'
write_to_csv(EventID, proba, threshold, name)


print(f"\nValidation accuracy : {score*100:.2f}")
