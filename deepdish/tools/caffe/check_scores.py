from __future__ import division, print_function, absolute_import

import numpy as np
import deepdish as dd
import itertools as itr

def concat(x, y):
    if x is None:
        return np.asarray([y])
    else:
        return np.concatenate([x, [y]])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=str)
    parser.add_argument('-k', default=1, type=int)

    args = parser.parse_args()

    rs = np.random.RandomState(0)

    L = 50

    data = dd.io.load(args.scores)
    scores = data['scores']
    y = data['labels']
    names = data['names']

    K = scores.shape[1]
    Y = np.zeros((y.shape[0], y.max() + 1))
    Y[np.arange(y.shape[0]), y] = 1
    for k in [args.k]:
        combs = list(itr.combinations(range(K), k))
        rs.shuffle(combs)

        the_rates = []

        for indices in combs[:L]:
            networks = scores[:,list(indices)]

            # TODO: Make ensemble method a choice
            ensemble = np.exp(np.log(networks + 1e-10).mean(1))

            rates = (ensemble.argmax(-1) != y).mean(-1)

            the_rates.append(rates)

        error = np.mean(the_rates)
        success = 1 - error
        #print('k={}, rate={:.2f}%', rate)
        print('Success: {:.2f}% / Error: {:.2f}%'.format(success * 100, error * 100))
