from xgboost import XGBClassifier
from DataHandler import DataHandler
from joblib import dump, load
import numpy as np
from Functions import *
import matplotlib.pyplot as plt
import os
import scikitplot as skplt
import plot_set



# Data handling
print("Preparing data...")
X_test = np.load("../Data/featuresTest.npy")
EventID = X_test[:,0].astype(int)
X_test = X_test[:,1:]

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
nr_train = DH.nrEvents
DH.X_train = np.concatenate((DH.X_train, X_test), axis=0)
DH.setNanToMean()
X, Y = DH(include_test=False)

DH.X_train = X[: nr_train,:]
X_test = X[nr_train:,:]
DH.split()
X_train, X_val, y_train, y_val = DH(include_test=True)

dirname = os.getcwd()
filename = os.path.join(dirname, "sklearn_models/model_hypermodelDT.joblib")
model = load(filename)

model.fit(X_train, y_train)
score = model.score(X_val,y_val)


probas = model.predict(X_val)
proba = probas.ravel()



s = proba[np.where(y_val == 1)]
b = proba[np.where(y_val == 0)]


sigma =np.nanstd(b)
diff = abs(np.mean(b) - np.mean(s))
x_start = np.mean(b)
x_end =np.mean(s)
y_start = 8
binsize = 150



plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                     label = "Background", density=True)
n_s, bins_s, patches_s = plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, 
                                     label = "Signal",  density=True)

median_s = bins_s[np.where(n_s==np.max(n_s))][0]
median_b = bins_b[np.where(n_b==np.max(n_b))][0]
plt.axvline(x=x_start,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
plt.axvline(x=x_end,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.axvline(x=median_s,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
plt.axvline(x=median_b,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
plt.xlabel("Output", fontsize=15)
plt.ylabel("#Events", fontsize=15)
plt.title("Decision tree output distribution", fontsize=15, fontweight = "bold")
plt.legend(fontsize = 16, loc = "upper right")
plt.annotate("", xy=(x_start,y_start),
            xytext=(x_end,y_start),verticalalignment="center",
            arrowprops={'arrowstyle': '<->', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}",
                xy=(((x_start+x_end)/2), y_start+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.annotate("", xy=(median_b,y_start*2),
            xytext=(median_s,y_start*2),verticalalignment="center",
            arrowprops={'arrowstyle': '<->', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid s_m-b_m\mid$"+f" = {abs(median_b-median_s):.2f}",
                xy=(((median_b+median_s)/2), y_start*2+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.savefig("../figures/DT/DT_output.pdf", bbox_inches="tight")
plt.show()


probas = [[1-prob,prob] for prob in proba]#np.concatenate((1-proba,proba),axis=1)
skplt.metrics.plot_roc(y_val, probas)
plt.xlabel("False positive rate", fontsize=15)
plt.ylabel("True positive rate", fontsize=15)
plt.title("Decision tree: ROC curve", fontsize=15, fontweight = "bold")
#plt.savefig("../figures/DT/DT_ROC.pdf", bbox_inches="tight")
plt.show()


threshold = 0.85

proba = model.predict(X_test)
name = '../Data/DT_test_pred.csv'
write_to_csv(EventID, proba, threshold, name)


