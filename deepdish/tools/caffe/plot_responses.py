from __future__ import division, print_function, absolute_import 
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=8)
import pylab as plt
from vzlog import VzLog
import deepdish as dd
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('responses', nargs='+', type=str)
    parser.add_argument('-t', '--title', default='', type=str)
    parser.add_argument('-o', '--output', default='log-responses', type=str)
    parser.add_argument('-l', '--layers', nargs='?', type=str)
    parser.add_argument('-a', '--alpha', default=0.99, type=float)

    args = parser.parse_args()
    alpha = args.alpha

    vz = VzLog(args.output)
    vz.title(args.title)
    vz.log('alpha =', alpha)

    plt.figure()

    layers = args.layers

    for fn in args.responses:
        data = dd.io.load(fn)
        name = data['name']
        if layers is None:
            layers = data['layers']
        y = []
        ystd = []
        for l in layers:
            rs = []
            for X in data['responses'][l]:
                C = np.corrcoef(X.T)

                rs.append(np.where(np.sort(np.linalg.eigvals(C))[::-1].cumsum() / C.shape[0] > alpha)[0][0] / C.shape[0])
            y.append(np.mean(rs))
            ystd.append(np.std(rs))
        plt.errorbar(np.arange(len(y)), y, yerr=ystd, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('<- Redundancy / Identity-like ->')
    plt.legend(loc=4)
    plt.ylim((0, 1))
    plt.savefig(vz.impath('svg'))
    plt.close()

if __name__ == '__main__':
    main()
