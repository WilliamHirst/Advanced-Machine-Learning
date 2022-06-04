from sklearn.cluster import KMeans
import numpy as np
from DataHandler import DataHandler
from Functions import *
import plot_set
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt 
import scikitplot as skplt


if __name__ == "__main__":
    seed = tf.random.set_seed(1)

    X_test = np.load("../Data/featuresTest.npy")
    EventID = X_test[:,0].astype(int)
    X_test = X_test[:,1:]

    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
    
    nr_train = DH.nrEvents
    DH.X_train = np.concatenate((DH.X_train, X_test), axis=0)
    DH.setNanToMean()
    #DH.standardScale()

    X, Y = DH(include_test=False)

    X_train = X[: nr_train,:]
    X_test = X[nr_train:,:]
    DH.X_train = X_train

    DH.split()
    X_train, X_val, y_train, y_val = DH(include_test=True)

    clf = IsolationForest(random_state=seed).fit(X_train)
    prediction = clf.predict(X_val)
    prob = clf.decision_function(X_val)
    #prob = (prob - np.min(prob))/(np.max(prob)-np.min(prob))
    #prob = (1 - prob).reshape(len(prob),1)
    prob = prob.reshape(len(prob), 1)


    s = prob[np.where(y_val == 1)]
    b = prob[np.where(y_val == 0)]

    sigma =np.nanstd(b)
    diff = abs(np.mean(b) - np.mean(s))
    x_start = np.mean(b) 
    x_end =np.mean(s)
    y_start = 3

    binsize = 150
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                        label = "Background", density=True)
    n_s, bins_s, patches_s = plt.hist(s, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6, 
                                        label = "Signal",  density=True)

    median_s = bins_s[np.where(n_s==np.max(n_s))][0]
    median_b = bins_b[np.where(n_b==np.max(n_b))][0]
    #plt.axvline(x=x_start,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
    #plt.axvline(x=x_end,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
    #plt.axvline(x=median_s,linestyle="--", color="black",alpha = 0.6, linewidth = 1)
    #plt.axvline(x=median_b,linestyle="--", color="black", alpha = 0.6, linewidth = 1)
    plt.annotate(text=r"$\mid \langle s \rangle - \langle b \rangle \mid$" 
                + f" = {diff:.3f}",
                xy=((0.5), y_start+1.), xycoords='data',
                fontsize=15.0,textcoords='data',ha='center')

    plt.annotate(r"$\mid s_m-b_m\mid$"+f" = {abs(median_b-median_s):.3f}", xycoords='data',
                xy =(0.5, y_start+4.), 
                fontsize=15.0,textcoords='data',ha='center')
    plt.xlabel("Output", fontsize=15)
    plt.ylabel("#Events", fontsize=15)
    plt.title("Isolation Forest output distribution", fontsize=15, fontweight = "bold")
    plt.legend(fontsize = 16, loc = "upper right") 
    plt.savefig("../figures/Cluster/Cluster_output_u_scale.pdf", bbox_inches="tight")
    plt.show()

    print(np.shape(prob))
    probas = np.concatenate((1-prob, prob),axis=1)

    skplt.metrics.plot_roc(y_val, probas)
    plt.xlabel("False positive rate", fontsize=15)
    plt.ylabel("True positive rate", fontsize=15)
    plt.title("Isolation Forest: ROC curve", fontsize=15, fontweight = "bold")
    plt.savefig("../figures/Cluster/Cluster_ROC_u_scale.pdf", bbox_inches="tight")
    plt.show()
    
    
    from sklearn.metrics import roc_curve, auc

    print(np.shape(y_val), np.shape(probas))
    fpr, tpr, thresholds = roc_curve(y_val, prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.show()
    exit()
    threshold = 0.85
    proba = clf.predict(X_test).ravel()
    name = '../Data/iForest_test_pred_u_scale.csv'
    write_to_csv(EventID, proba, threshold, name)