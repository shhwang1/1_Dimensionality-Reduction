import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    plt.show()

def pca(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    X_scaled = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
    y_data = data.iloc[:, -1]

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    plot_variance(pca)

    var_list = pca.explained_variance_ratio_
    alpha = 0
    pca_num = 0
    while True:
        for i in range(len(var_list)):
            alpha += var_list[i]
            if alpha >= 0.8:
                pca_num = i+1
                break
        break
        
    print('Principal Components Number : ', pca_num)
    print('PCA again with', pca_num, 'components....')

    pca2 = PCA(n_components = pca_num)
    X_pca_2 = pca2.fit_transform(X_scaled)
    
    labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca2.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        X_pca,
        labels=labels,
        dimensions=range(pca_num),
        color = y_data
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()