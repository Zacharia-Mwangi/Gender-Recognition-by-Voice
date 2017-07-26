import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('/home/zack/Desktop/GC_Recognition/voice.csv', header=0)
data_cols = list(data.columns[0:20])

def label_frequency():
    # gender labels
    data['label'] = data['label'].map({'male':1, 'female': 0})

    # plot frequency of Labels
    sns.countplot(data['label'], label="Count")
    sns.plt.show()

    """
        Both Labels have data of equal frequencies
    """

def find_correlation():
    # correlation graphs
    corr = data[data_cols].corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size':10}, xticklabels=data_cols, yticklabels=data_cols, cmap='coolwarm')
    sns.plt.show()
    """
        Correlated group 1: meanfreq,median, Q25, Centroid
        Correlated group 2: sd, IQ2,sfm
        Correlated group 3: maxdom, dfrange, meandom
    """

prediction_variables = ['meanfreq', 'sd', 'Q75', 'skew', 'kurt', 'sp.ent', 'mode', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'modindx' ]






