import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from DataHandler import DataHandler
import plot_set
from sklearn.metrics import accuracy_score
import scikitplot as skplt

from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV


# Data handling
print("Preparing data...")

DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
DH.setNanToMean()#DH.fillWithImputer()
DH.standardScale()


X_train, y_train, X_val, y_val = DH.AE_prep()

classifier = linear_model.SGDOneClassSVM(nu = 0.01)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=30)
data_transformed = feature_map_nystroem.fit_transform(X_train)
classifier.fit(data_transformed) 


y_val = np.asarray([-1 if i == 0 else 1 for i in y_val])





exit()
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
plt.title("SVM output distribution", fontsize=15, fontweight = "bold")
plt.legend(fontsize = 16)
plt.annotate("", xy=(x_start,y_start),
            xytext=(x_end,y_start),verticalalignment="center",
            arrowprops={'arrowstyle': '|-|', 'lw': 1., "color":"black"}, va='center')
            
plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.2f}" + r"$\sigma_b$",
                xy=(((x_start+x_end)/2), y_start+0.5), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

plt.savefig("../figures/SVM/SVM_output.pdf", bbox_inches="tight")
plt.show()

exit()
skplt.metrics.plot_roc(y_val, proba)
plt.xlabel("True positive rate", fontsize=15)
plt.ylabel("False positive rate", fontsize=15)
plt.title("SVM: ROC curve", fontsize=15, fontweight = "bold")
plt.savefig("../figures/SVM/SVM_ROC.pdf", bbox_inches="tight")
plt.show()