import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='1_Dimensionality Reduction')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='abalone.csv',
                        choices = ['abalone.csv', 'BankNote.csv', 'PersonalLoan.csv', 'WineQuality.csv', 'Diabetes.csv'])
    parser.add_argument('--seed', type=int, default=1592)              

    # Choose methods
    parser.add_argument('--method', type=str, default='PCA',
                        choices = ['BackwardSelection, ForwardSelection, StepwiseSelction, GeneticAlgorithm, PCA, MDS, ISOMAP, LLE, t-SNE'])

    # Forward_Selection 
    parser.add_argument('--num-features', type=int, default=7)
    parser.add_argument('--split-size', type=float, default=0.2)
    parser.add_argument('--sl-enter', type=int, default=0.5)
    parser.add_argument('--sl-remove', type=int, default=0.5)
    
    # Locally Linear Embedding (LLE)
    parser.add_argument('--n-components', type=int, default = 2)
    parser.add_argument('--n-neighbors', type=int, default = 3)
    
    return parser