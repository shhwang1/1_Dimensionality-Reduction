## Unsupedvised Methods

#### - As a result of the supervised methods, 'abalone dataset' does not fit well, so only 3 datasets are used in unsupervised methods.

### 1. Principal Component Analysis (PCA)

PCA is a technique of summarizing and abbreviating new variables made of a linear combination of several highly correlated variables. PCA finds a new orthogonal basis while preserving the variance of the data as much as possible. After that, the sampling of the high-dimensional space is converted into a low-dimensional space with no linear association. Simply, the dimension is reduced by finding the axis of the data with the highest variance, which is the main component of PCA.

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194999981-01e2c48c-7f02-4fb8-b63e-8ca43c7e9e2e.png" width="800" height="450"></p>


### Python Code
``` C
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
``` 
The code sets the threshold of 'cumulative explained variation' to 0.8, and determines the number of components when it exceeds 0.8, as the optimal point.   

### Analysis

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195001512-3a514f0d-f35c-4f68-9b47-e7ff837d54a3.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195002774-6c889e06-3708-4674-8778-19ee112025ee.png"></p>

The winequality dataset is the optimal points when six principal components and the diabetes dataset is the optimal points when five principal components, and the PersonalLoan dataset is the optimal point when eight principal components.


![image](https://user-images.githubusercontent.com/115224653/195003434-1f0ab6f6-1e1d-4075-af90-e8ca11c470cd.png)
![image](https://user-images.githubusercontent.com/115224653/195003575-f837a4e8-004a-4147-b0a0-791008183c65.png)
![image](https://user-images.githubusercontent.com/115224653/195003099-0cd11afc-6969-4599-a887-08087e056bc0.png)

___
### 2. Multi-Dimensional Scaling (MDS)
Multi-Dimensional Scaling(MDS) addresses scale issues in input states that are essentially outputless. Given a distance matrix D defined between all points in the existing dimension space, the inner matrix B is used to create a coordinate system y space. The most important point of MDS is that it preserves the distance information between all points.   

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195004491-1a0631d5-7f61-4932-9f66-5e3cd14ba802.png"></p>

### Python Code

``` C
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
```    

### Analysis   

We used the MDS of the manifold module in the sklearn package. It was formed as a two-dimensional coordinate system in the existing dimension of each dataset, and the results are shown in the figure below.   

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195006466-723958d1-b286-44af-aa6c-a9723c72540d.png"></p>

Using Multi-Dimensional Scaling, PersonalLoan dataset determined that the two classes were relatively well clustered. However, the Wine Quality dataset and the Diabetes dataset did not cluster so well.   
___
### 3. ISOMAP

Isomap is an extension of multidimensional scaling (MDS) or principal component analysis (PCA) and a methodology that combines the two methods. Isomap algorithm seeks to effectively reduce dimensions using distance information that reflects the real features between the two data. The Isomap algorithm consists of three steps:

1. Build adjacent neighbor graphs
2. Calculate the shortest path graph between two points
3. Building d-dimensional embeddings using the MDS methodology

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195007612-63c37397-88ad-4ae8-ba13-f3555d57a6e9.png"></p>   

### Python Code

``` C
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler

def isomap(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    isomap = manifold.Isomap(n_components = 2)
    new_dim = isomap.fit_transform(X_scaled)

    df = pd.DataFrame(new_dim, columns=['X', 'Y'])
    df['label'] = y_data

    fig = plt.figure()
    fig.suptitle('Isomap', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)

    for i in np.unique(y_data):
        plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i)

    plt.legend(bbox_to_anchor=(1.25, 1))
    plt.show()
``` 
   
### Analysis   

![image](https://user-images.githubusercontent.com/115224653/195009264-3f042d45-01e1-4cc4-bcf5-44116b08de37.png)

### 4. Locally Linear Embedding (LLE)

Locally Linear Embedding (LLE) is an algorithm that maps input datasets to a single global coordinate system of low dimensions. The process of LLE can be largely divided into three stages as follows.

1. Step 1) Select neighbors 
2. Step 2) Reconstruct with linear weights
3. Step 3) Map to embedded coordinates

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195010814-cc4e4601-2da6-4184-aac7-60c362f6bab5.png"></p>  

### Python Code
``` C
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
``` 

### Analysis 

![image](https://user-images.githubusercontent.com/115224653/195011804-dbea6f81-614a-4f81-87b0-5251389a4fc2.png)


### 5. t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-Distributed Stochastic Neighbor Embedding(t-SNE) is one of manifold learning and aims at visualizing complex data. Visualize high-dimensional data by shrinking it to two or three dimensions. t-SNE is characterized by utilizing the t-distribution rather than the normal distribution. With t-SNE, similar data structures in high-dimensional space correspond closely in low-dimensional space, and non-similar data structures correspond at a distance. 

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195012417-4af7ed57-4210-4c34-9f49-87a77339195a.png"></p>  

``` C
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
```    

### Analysis   

![image](https://user-images.githubusercontent.com/115224653/195013305-d42cbe97-717b-41dd-93de-02e003c404e0.png)
