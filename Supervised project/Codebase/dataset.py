import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


class Dataset:
    def __init__(self, dataset_train, dataset_test, column_names):
        self.features = dataset_train
        self.targets = dataset_test
        self.column_names = column_names
        
        self.dataframe_feats = pd.DataFrame(
            self.features[:, 1:-1], columns=self.column_names[1:-2]
        )

    def __call__(self):
        return 0

    def correlationPlot(self):
        """
        Corrolation plot for all the features in the dataset
        """

        plt.figure(figsize=(40, 40))
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        sns.heatmap(self.dataframe_feats.corr().round(2), annot=True, cmap="viridis")

        plt.savefig("../figures/corre_allfeats.png")
        plt.show()

    def massSignalBackground(self):
        background = np.where(self.targets == 0)[0]
        signal = np.where(self.targets == 1)[0]
        mass_data_background = self.features[:, 1][background]
        mass_data_signal = self.features[:, 1][signal]

        datapoint = len(self.features[:, 1])
        binsize = 1000
        print(datapoint / binsize)

        plt.hist(
            mass_data_background, bins=binsize, histtype="stepfilled", facecolor="b"
        )
        plt.hist(mass_data_signal, bins=binsize, histtype="step", facecolor="r")
        plt.legend(["Background", "Signal"])
        plt.xlabel("Mass in GeV")
        plt.ylabel("Number of events per given mass in GeV")
        plt.title("Histogram of background and signal masses")
        plt.savefig("../figures/histogram_mass_bs.pdf")
        plt.show()


if __name__ == "__main__":
    featuresTrain = np.load("../Data/featuresTrain.npy")
    targetsTrain = np.load("../Data/targetsTrain.npy")
    column_names = np.load("../Data/column_names.npy")

    data_obj = Dataset(featuresTrain, targetsTrain, column_names)

    data_obj.correlationPlot()
    # data_obj.massSignalBackground()
