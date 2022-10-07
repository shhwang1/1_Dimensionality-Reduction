import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def t_sne(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    tsne = TSNE(n_components = args.n_components, random_state = args.seed)
    z = tsne.fit_transform(X_data)

    df = pd.DataFrame()
    df['y'] = y_data
    df['comp-1'] = z[:, 0]
    df['comp-2'] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(np.unique(y_data))),
                data=df).set(title="T-SNE projection")

    plt.show()