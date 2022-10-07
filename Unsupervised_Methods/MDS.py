import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

def mds(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    y_list = np.unique(y_data)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    mds = MDS(2,random_state=0)
    X_2d = mds.fit_transform(X_scaled)

    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)
    for i in np.unique(y_data):
        subset = X_2d[y_data == i]
    
        x = [row[0] for row in subset]
        y = [row[1] for row in subset]
        plt.scatter(x, y, label=i)

    plt.legend()
    plt.show()