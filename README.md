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
### Unsupervised Methods [Link](https://github.com/shhwang1/1_Dimensionality-Reduction/tree/main/Unsupervised_Methods)
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
