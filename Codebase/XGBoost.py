from xgboost import XGBClassifier
from DataHandler import DataHandler
from joblib import dump, load
import numpy as np
from Functions import *
import matplotlib.pyplot as plt
import os
import scikitplot as skplt
import plot_set


threshold = 0.85
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodel.joblib")
model = load(filename)

score = model.score(X_val,y_val)




probas = model.predict_proba(X_val)
proba = probas[:,1].ravel()

s = proba[np.where(y_val == 1)]
b = proba[np.where(y_val == 0)]

sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))/sigma
x_start = np.mean(b)
x_end =np.mean(s)
y_start = 3
binsize = 100


plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",label = "Background", density=True)
plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, label = "Signal", density=True)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#-of-events", fontsize=15)
plt.title("XGBoost output distribution", fontsize=15, fontweight = "bold")
plt.legend(fontsize = 16)
plt.annotate("", xy=(x_start,y_start),
            xytext=(x_end,y_start),verticalalignment="center",
            arrowprops={'arrowstyle': '|-|', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}" + r"$\sigma_b$",
                xy=(((x_start+x_end)/2), y_start+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.savefig("../figures/XGB/XGB_output.pdf", bbox_inches="tight")
plt.show()


skplt.metrics.plot_roc(y_val, probas)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("XGBoost: ROC curve", fontsize=15, fontweight = "bold")
plt.savefig("../figures/XGB/XGB_ROC.pdf", bbox_inches="tight")
plt.show()
exit()
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,0].astype(int)
proba = model.predict_proba(X_test[:,1:])[:,1]
name = '../Data/xgboost_test_pred.csv'
write_to_csv(EventID, proba, threshold, name)


print(f"\nValidation accuracy : {score*100:.2f}")
