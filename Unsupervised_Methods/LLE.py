import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler

def lle(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X_data)

    lle = LocallyLinearEmbedding(n_neighbors=args.n_neighbors,
                            n_components = args.n_components,
                            eigen_solver = 'auto',
                            method='standard',)
    mds_x = lle.fit_transform(X_data)

    plt.figure(figsize=(10,10))
    for i in range(10):
        plt.scatter(mds_x[y_data==i][:,0],
                    mds_x[y_data==i][:,1],
                    label = i)
        plt.legend()
    plt.title("LLE")    
    plt.show()            
