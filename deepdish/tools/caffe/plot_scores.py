from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=8)
import matplotlib.pylab as plt
import numpy as np
import deepdish as dd
from vzlog import VzLog
import itertools as itr

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=str)
    parser.add_argument('-o', '--output', default='log-scores', type=str)

    args = parser.parse_args()

    rs = np.random.RandomState(0)

    data = dd.io.load(args.scores)
    scores = data['scores']
    y = data['labels']
    names = data['names']

    L = 50

    all_rates = []
    all_Hs = []
    all_ambigs = []
    all_rates_std = []
    all_Hs_std = []
    all_ambigs_std = []
    all_corrs = []
    all_corrs_std = []
    the_corrs = []

    vz = VzLog(args.output)

    K = scores.shape[1]
    Y = np.zeros((y.shape[0], y.max() + 1))
    Y[np.arange(y.shape[0]), y] = 1
    Ks = np.arange(K) + 1
    for k in Ks:
        combs = list(itr.combinations(range(K), k))
        rs.shuffle(combs)

        the_rates = []
        the_Hs = []
        the_ambigs = []
        print('k', k)

        for indices in combs[:L]:
            networks = scores[:,list(indices)]

            # TODO: Make ensemble method a choice
            ensemble = np.exp(np.log(networks + 1e-10).mean(1))
            #ensemble = networks.mean(1)

            rates = (ensemble.argmax(-1) != y).mean(-1)

            the_rates.append(rates)
            Hs = 0
            the_Hs.append(Hs)
            the_ambigs.append(vars)


        all_rates.append(np.mean(the_rates, axis=0))
        all_rates_std.append(np.std(the_rates, axis=0))

        all_Hs.append(np.mean(the_Hs, axis=0))
        all_Hs_std.append(np.std(the_Hs, axis=0))

        yy = np.asarray(the_rates)

    all_rates = np.asarray(all_rates)
    all_rates_std = np.asarray(all_rates_std)

    print('all_rates', all_rates.shape)

    plt.figure()
    diffs = all_rates[-1] - all_rates[0]
    for i in range(all_rates.shape[1]):
        caption = '{} {:.2f}% ({:.2f})'.format(names[i], 100*all_rates[-1,i], 100*diffs[i])
        print('caption', caption)
        plt.errorbar(Ks, all_rates[:,i], yerr=all_rates_std[:,i], label=caption)#, yerr=all_rates_std)
    plt.xlim((0, len(Ks)+1))
    plt.legend()
    plt.xlabel('Networks')
    plt.ylabel('Error rate')
    plt.savefig(vz.impath('svg'))
    plt.close()

    plt.figure()
    plt.scatter(all_rates[0], diffs)
    for ll in plt.xlim, plt.ylim:
        lims = ll(); ll([lims[0], lims[1]]) 

    x = np.asarray([0, 1])
    y = -x + 0.14
    plt.plot(x, y)
    plt.xlabel('single-network error rate')
    plt.ylabel('10-network improvement')
    plt.savefig(vz.impath('svg'))
    plt.close()

if __name__ == '__main__':
    main()
