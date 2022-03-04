from pyexpat import features
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import plot_set


class Dataset:
    def __init__(self, dataset_train, dataset_test, column_names):
        self.features = dataset_train
        self.targets = dataset_test
        self.column_names = column_names
        background = np.where(self.targets == 0)[0]
        signal = np.where(self.targets == 1)[0]
        self.B = self.features[background]
        self.S = self.features[signal]
        
        self.dataframe_feats = pd.DataFrame(
            self.features[:, 1:-1])#, columns=self.column_names[1:-2]
        #)

    def __call__(self):
        return 0

    def correlationPlot(self):
        """
        Corrolation plot for all the features in the dataset
        """

        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k', figsize=(10, 8)) 
        plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False,
                        labelleft = False,
                        left = False)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        sns.heatmap(self.dataframe_feats.corr().round(2), annot=False, cmap="RdBu")

        plt.savefig("../figures/corre_allfeats.png")
        plt.show()

    def massSignalBackground(self, feature, ax):

        B = self.B[:, feature]
        S = self.S[:, feature]

        binsize = 1000
        
        ax.hist(
            B, bins=binsize, histtype="stepfilled", facecolor="b"
        )
        ax.hist(S, bins=binsize, histtype="stepfilled", facecolor="r",alpha=0.6)
        ax.set_xlim([np.nanmin(B)*1.2, np.nanmax(B)*0.6])
        ax.set_title( self.column_names[feature],  fontsize=12,fontweight="bold")   

    def plotManyFeature(self, labels):
        fig = plt.figure(figsize=(6, 6))
        
        for i in range(len(labels)):
            ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
            
            self.massSignalBackground(labels[i], ax)
        ax.legend(["Background", "Signal"], fontsize = 12)
        plt.tight_layout(pad=1.0, w_pad=0.5)
        plt.savefig("../figures/featureHisto.png")
        plt.show()     
    
    def plotManyBoxes(self, labels):
        fig = plt.figure(figsize=(6, 6))
        last = False
        for i in range(len(labels)):
            ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
            if labels[i] ==labels[-1]:
                last = True
            self.box_plot(ax,labels[i],last)
        plt.tight_layout(pad=1.0, w_pad=0.5)
        plt.savefig("../figures/featureBoxes.png")
        plt.show()   

    def box_plot(self, ax, feature,last):
        B = self.B[:, feature]
        S = self.S[:, feature]
        col_mean1 = np.nanmean(B)
        col_mean2 = np.nanmean(S)
        inds1 = np.where(np.isnan(B))[0]
        inds2 = np.where(np.isnan(S))[0]
        if len(inds1)+len(inds2) >= 2:
            S[inds2] = col_mean2
            B[inds1] = col_mean1
        colors = ['blue', 'red']
        medianprops = dict(linestyle='--', linewidth=2, color='black')
        boxes = ax.boxplot([B, S], showfliers=False, 
                                   patch_artist=True, 
                                   medianprops=medianprops,
                                   widths=(0.4, 0.4))
        ax.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
        ax.set_title( self.column_names[feature],  fontsize=12,fontweight="bold")   
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
        if last:
            ax.legend(boxes['boxes'],["Background", "Signal"], loc='upper right', fontsize = 12)

    def featurePlot(self):
        
        meanTot = np.nanmean(self.features, axis = 0)
   
        meanB = np.nanmean(self.B, axis = 0)
        meanS = np.nanmean(self.S, axis = 0)
        print(meanB.shape)
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')  
        plt.plot(np.arange(30), meanB/meanTot, label = "Background")
        plt.plot(np.arange(30), meanS/meanTot, "--", label = "Signal")
        plt.legend(fontsize = 16)
        plt.xlabel(r"$Feature\ index\ [i]$", fontsize=16)
        plt.ylabel(r"$Avarage\ of \ event\ [Tot \ avarage]$", fontsize=16)
        plt.show()

    def barPlot(self):
        prosSignal = len(self.S)/len(self.features)*100
        prosBackground = len(self.B)/len(self.features)*100
        plt.bar("Signal",prosSignal)
        plt.bar("Background", prosBackground)
        plt.show()
    
   
            


if __name__ == "__main__":
    featuresTrain = np.load("../Data/rawFeatures_TR.npy")
    targetsTrain = np.load("../Data/rawTargets_TR.npy")
    column_names = np.load("../Data/column_names.npy")

    data_obj = Dataset(featuresTrain, targetsTrain, column_names)
    #data_obj.featurePlot()
    data_obj.correlationPlot()
    #data_obj.barPlot()
    exit()
    data_obj.plotManyFeature(np.array([0,1,2,4,6,10,14,17,19]))
    data_obj.plotManyBoxes(np.array([0,1,2,20,21,10,14,17,19]))

  
