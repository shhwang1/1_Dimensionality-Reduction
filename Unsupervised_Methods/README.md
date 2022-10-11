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


### 3. ISOMAP

https://user-images.githubusercontent.com/115224653/195007612-63c37397-88ad-4ae8-ba13-f3555d57a6e9.png

Stepwise selection is a method of deleting variables that are not helpful or adding variables that improve the reference statistics the most among variables missing from the model. Stepwise selection, like Backward Selection, starts with all variables. We call the method of using a regression model using variables selected in Stepwise selection a 'stepwise regression analysis'.      

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194992685-38e77aa4-5a6d-44ff-bf3c-c6dcdcaa369c.png"></p>   

### Python Code

``` C
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

variables = X_train.columns.tolist()
y = y_train

selected_variables = []
sl_enter = 0.05
sl_remove = 0.05

sv_per_step = [] 
adjusted_r_squared = []
steps = []
step = 0
while len(variables) > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) 
    for col in remainder: 
        X = X_train[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit(disp=0)
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: 
        selected_variables.append(pval.idxmin())
        while len(selected_variables) > 0:
            selected_X = X_train[selected_variables]
            selected_X = sm.add_constant(selected_X)
            selected_pval = sm.OLS(y,selected_X).fit(disp=0).pvalues[1:] 
            max_pval = selected_pval.max()
            if max_pval >= sl_remove: 
                remove_variable = selected_pval.idxmax()
                selected_variables.remove(remove_variable)
            else:
                break

        step += 1
        steps.append(step)
        adj_r_squared = sm.OLS(y,sm.add_constant(X_train[selected_variables])).fit(disp=0).rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
    else:
        break
``` 
   
### Analysis   

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194993825-468b4277-1393-4898-b427-40406b188bc5.png"></p>  
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/195001408-3a820591-f950-421d-b7c7-3d92c16de865.png"></p> 

### 4. Genetic Algorithm

Genetic Algorithm is a meta-heuristic technique with a structure in which superior genes survive. This method first sets possible initial solutions for the problem. And we evaluate it and leave solutions that meet certain criteria. In addition, a new solution is created and repeated using the "crossover" process of creating a new solution by crossing two genes and the "mutation" of modifying existing genes. Although it cannot necessarily guarantee the optimal solution, it has the advantage of finding a close solution in a short time.   

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194995273-ec0b4a58-3a61-42e6-92a1-cbeba35f408e.png"></p> 

``` C
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import warnings

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]
          
score, best_model_index = acc_score(X_data, y_data)
    print(score)
    print('Starting Genetic-Algorithm with', classifiers[best_model_index])
``` 
First, the model with the highest accessibility is selected by using the options of the above 'models'. Accuracy for each model is calculated as shown in the results below. The following results are examples of 'Wine Quality' dataset.   

![image](https://user-images.githubusercontent.com/115224653/194996083-610bda65-3f36-4b7b-9420-37ccdddc8745.png)

``` C
def generations(logmodel, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, logmodel, X_train, X_test, Y_train, Y_test)
        print('Best score in generation',i+1,':',scores[:1])  #2
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])

    return best_chromo, best_score
``` 
It is the most basic generation code. The logmodel represents the model with the highest Accuracy, and size represents the number of chromosomes. mutation_rate represents the ratio of mutation, and n_gen represents the number of generations.   

``` C
def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate,n_feat):   
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen
```    

### Analysis 

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194998098-abf8b436-46b5-44dd-9b97-17b5c20b0bd4.png"></p>   

Looking at the results, it seems that abalone dataset and Diabetes dataset using RadialSVM are not fitted to the genetic algorithm. Therefore, for the two datasets, RadialSVM was excluded from 'models' and re-experimented.   

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194999133-8e521a26-099d-46cb-b1ea-b95adfbf628d.png"></p>   

abalone dataset has very low performance of accuracy and GA-performance compared to other dataset. Personally, it is assumed that the performance will be relatively low because there are 29 classes(y-data) in the dataset.
