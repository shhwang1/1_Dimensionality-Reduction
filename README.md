# Dimensionality-Reduction Tutorial
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/194541972-557a4f92-aff2-4ea3-befa-240b89f2f45e.png" width="600" height="400"></p>



## Purpose of Tutorial
The purpose of this tutorial is for beginners studying Dimensionality Reduction.   
   
We also aim to improve our skills by explaining directly to someone.
   
From the variable selection method to the variable extraction method, we present a guide on the dimensional reduction method of the supervised method and the unsuperviced method.   
   
The table of contents is as follows.
___
### Supervised Methods

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

![image](https://user-images.githubusercontent.com/115224653/194545045-6d1ed8f2-782c-49d7-9405-775db3a042fe.png)


