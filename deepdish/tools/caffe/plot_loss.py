from __future__ import division, print_function, absolute_import 

import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=8)
import matplotlib.pylab as plt
import numpy as np
import deepdish as dd
from vzlog import VzLog
import itertools as itr
import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('losses', nargs='+', type=str)
    parser.add_argument('-o', '--output', default='log-loss', type=str)
    #parser.add_argument('--dataset', default='Test', type=str)
    parser.add_argument('--captions', nargs='+', type=str)

    args = parser.parse_args()

    rs = np.random.RandomState(0)

    vz = VzLog(args.output)

    plt.figure()
    for i, loss_fn in enumerate(args.losses):
        print('file', loss_fn)
        data = dd.io.load(loss_fn)
        iter = data['iterations'][0]
        losses = data['losses'].mean(0)
        losses_std = data['losses'].std(0)

        if args.captions:
            caption = args.captions[i]
        else:
            caption = loss_fn

        caption = '{} ({:.3f})'.format(caption, losses[-1])

        plt.errorbar(iter, losses, yerr=losses_std, label=caption)

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    vz.savefig()
    plt.close()

    plt.figure()
    for i, loss_fn in enumerate(args.losses):
        print('file', loss_fn)
        data = dd.io.load(loss_fn)
        iter = data['iterations'][0]
        rates = 100*(1-data['rates'].mean(0))
        rates_std = 100*data['rates'].std(0)

        if args.captions:
            caption = args.captions[i]
        else:
            caption = loss_fn

        caption = '{} ({:.2f}%)'.format(caption, rates[-1])

        plt.errorbar(iter, rates, yerr=rates_std, label=caption)

    plt.legend()
    plt.ylabel('Error rate (%)')
    plt.xlabel('Iteration')
    vz.savefig()
    plt.close()

if __name__ == '__main__':
    main()
