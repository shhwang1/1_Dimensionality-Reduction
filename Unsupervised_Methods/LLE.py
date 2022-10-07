from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

def lle(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    lle = LocallyLinearEmbedding(n_components = args.n_components, n_neighbors = args.n_neighbors, random_state = args.seed)
    lle.fit(X_data)

    X_reduced = lle.transform(X_data)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_data, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)

    plt.show()