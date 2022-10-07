from args import Parser1

from Supervised_Methods.ForwardSelection import forward_selection
from Supervised_Methods.BackwardSelection import backward_selection
from Supervised_Methods.StepwiseSelection import stepwise_selection
from Supervised_Methods.GeneticAlgorithm import genetic_algorithm

from Unsupervised_Methods.PCA import pca
from Unsupervised_Methods.MDS import mds
from Unsupervised_Methods.t_SNE import t_sne
from Unsupervised_Methods.ISOMAP import isomap
from Unsupervised_Methods.LLE import lle

def build_model():
    parser = Parser1()
    args = parser.parse_args()

    if args.method == 'BackwardSelection':
        model = backward_selection(args)
    elif args.method == 'StepwiseSelction':
        model = stepwise_selection(args)
    elif args.method == 'ForwardSelection':
        model = forward_selection(args)
    elif args.method == 'GeneticAlgorithm':
        model = genetic_algorithm(args)
    elif args.method == 'PCA':
        model = pca(args)
    elif args.method == 'MDS':
        model = mds(args)
    elif args.method == 'ISOMAP':
        model = isomap(args)
    elif args.method == 'LLE':
        model = lle(args)
    else:
        model = t_sne(args)

    return model

def main():
    build_model()

if __name__ == '__main__':
    main()
