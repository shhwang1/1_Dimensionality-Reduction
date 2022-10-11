## Supervised Methods

### 1. Forward Selection

Before we start, we need to know 'Wrapper'. 'Wrapper' is a supervised learning-based dimensionality reduction method that uses repeated algorithms. The Wrapper method includes Forward selection, Backward selection(elimination), Stepwise selection, and Genetic algotirithms. The first of these, 'Forward selection', is the way to find the most significant variables. Start with no vairables and move forward to increase the variables. Each step selects the best performing variable and runs it until there is no significant variables.

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194536545-570b0da8-2029-42b8-b9ba-49f2a172447c.png" width="600" height="600"></p>
<p align="center">https://quantifyinghealth.com/ Oct 07, 2022</p>

### Python Code
``` C
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

variables = X_data.columns.tolist()
    
forward_variables = []

sl_enter = args.sl_enter # selection threshold
sl_remove = args.sl_remove # elimination threshold
sv_per_step = [] # Variables selected in each steps
adj_r_squared_list = [] # Adjusted R-Square in each steps

steps = []
step = 0
``` 
sl_enter represents a threshold value of the p-value of the corresponding variable for selecting the variable.   
Conversely, sl_remove represents a threshold of a p-value for removing the corresponding variable.   

``` C
while len(variables) > 0:
    remainder = list(set(variables) - set(forward_variables))
    pval = pd.Series(index=remainder) # P-value

    for col in remainder:
        X = X_data[forward_variables+[col]]
        X = sm.add_constant(X)
        model = sm.OLS(y_data, X).fit(disp = 0)
        pval[col] = model.pvalues[col]

    min_pval = pval.min()

    if min_pval < sl_enter: # include it if p-value is lower than threshold
        forward_variables.append(pval.idxmin())
        while len(forward_variables) > 0:
            selected_X = X_data[forward_variables]
            selected_X = sm.add_constant(selected_X)
            selected_pval = sm.OLS(y_data, selected_X).fit(disp=0).pvalues[1:]
            max_pval = selected_pval.max()

            if max_pval >= sl_remove:
                remove_variable = selected_pval.idxmax()
                forward_variables.remove(remove_variable)

            else:
                break

        step += 1
        steps.append(step)
        adj_r_squared = sm.OLS(y_data, sm.add_constant(X_data[forward_variables])).fit(disp=0).rsquared_adj
        adj_r_squared_list.append(adj_r_squared)
        sv_per_step.append(forward_variables.copy())

    else:
        break
``` 
Calculate p_value through the 'statsmodel' package and determine whether to select a variable.

### Analysis

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194545328-3178119f-1829-4ef6-8878-3305ede7c2c2.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194545665-c1f44ed6-3e45-4af8-97ed-b5ba84daff97.png"></p>

In all four datasets, the adjusted-r-square value increased as the step went through.   
WineQuality and PersonalLoan datasets are judged to have no meaning in selecting variables as the increasing trend becomes insignificant when passing a specific step.
___
### 2. Backward selection(elimination)
Backward elimination is a way of eliminating meaningless variables. In contrast, it starts with a model with all the variables and move toward a backward that reduces the variables one by one. If you remove one variable, it repeats this process until a significant performance degradation occurs. Below is an image showing the process of reverse removal.

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194546699-e94c37d7-024e-446c-bacd-55556f56a91b.png"></p>
<p align="center">https://quantifyinghealth.com/ Oct 07, 2022</p>

### Python Code
Backward elimination does not differ significantly in code compared to forward selection. It starts with all variables, and compares the variable with the smallest p-value with a threshold and removes it if it is lower.

``` C
initial_list = []
threshold_out = 0.0
feature_list = X_data.columns.tolist()

for num in range(len(feature_list)-1):
  model = sm.OLS(y_data, sm.add_constant(pd.DataFrame(X_data[included]))).fit(disp=0)
  # use all coefs except intercept
  pvalues = model.pvalues.iloc[1:] # P-value of each variable
  worst_pval = pvalues.max()	# choose variable with best p-value
  if worst_pval > threshold_out:
      changed=True
      worst_feature = pvalues.idxmax()
      included.remove(worst_feature)

  step += 1
  steps.append(step)        
  adj_r_squared = sm.OLS(y_data, sm.add_constant(pd.DataFrame(X_data[included]))).fit(disp=0).rsquared_adj
  adj_r_squared_list.append(adj_r_squared)
  sv_per_step.append(included.copy())
``` 

### Analysis

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194547954-87aeeda9-5482-4b59-9e33-a50c3ad1ae4b.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194548316-d3f7e8f7-1505-471e-b430-c64ea7864a60.png"></p>

The initial Step with almost all variables shows a high adjusted-R-square value, but as the step passes, the number of variables decreases and the corresponding figure gradually decreases.   
In particular, in the last step where only one or two variables remain, it can be seen that the corresponding figure decreases rapidly.   

### 3. Stepwise Selection   

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
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194994126-c1a7ab3f-8c2e-42df-aea1-daeb9cb1369f.png"></p> 

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
