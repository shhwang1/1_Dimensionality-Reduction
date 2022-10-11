# Dimensionality-Reduction Tutorial
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194541972-557a4f92-aff2-4ea3-befa-240b89f2f45e.png" width="600" height="400"></p>



## Purpose of Tutorial
In recent years, data scientists have been increasingly dealing with data that has very many variables. Accordingly, minimizing variables and extracting only key variables has become a very important task. Accordingly, we create a github page for beginners who learn about the dimension reduction method from the beginning. We also aim to improve our skills by explaining directly to someone. From the variable selection method to the variable extraction method, we present a guide on the dimensional reduction method of the supervised method and the unsuperviced method.   
   
The table of contents is as follows.
___
### Supervised Methods [Link](https://github.com/shhwang1/1_Dimensionality-Reduction/tree/main/Supervised_Method)

#### 1. Forward Selection   
   
#### 2. Backward Selection   
   
#### 3. Stepwise Selection   
   
#### 4. Genetic Algorithm   
___
### Unsupervised Methods
#### 1. Principal Component Analysis (PCA)   
   
#### 2. Multi-Dimensional Scaling (MDS)   
   
#### 3. ISOMAP   
   
#### 4. Locally Linear Embedding (LLE)   
   
#### 5. t-Distributed Stochastic Neighbor Embedding (t-SNE)
___

## Dataset
We use 4 datasets (abalone, Diabetes, PersonalLoan, WineQuality)

abalone dataset : <https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset>     
Diabetes dataset : <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>   
PersonalLoan dataset : <https://www.kaggle.com/datasets/teertha/personal-loan-modeling>   
WineQuality datset : <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>    

Each dataset is a dataset for classification with a specific class value as y-data.   
   
In all methods, data is used in the following form.
``` C
import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='1_Dimensionality Reduction')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='abalone.csv',
                        choices = ['abalone.csv', 'BankNote.csv', 'PersonalLoan.csv', 'WineQuality.csv', 'Diabetes.csv'])
                        
data = pd.read_csv(args.data_path + args.data_type)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
```


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
