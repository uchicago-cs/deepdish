from __future__ import division, print_function, absolute_import

import numpy as np
import deepdish as dd
from collections import OrderedDict

def concat(x, y):
    if x is None:
        return np.asarray([y])
    else:
        return np.concatenate([x, [y]])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', nargs='+', type=str)
    parser.add_argument('-o', '--output', default='scores.h5', type=str)

    args = parser.parse_args()


    all_scores = None

    scores = OrderedDict()

    labels = None

    #scores = []
    for s in args.scores:
        data = dd.io.load(s)
        name = str(data['name'])
        if name not in scores:
            scores[name] = dict(scores=None, seeds=None)

        #scores[name]['names'] = np.concatenate([scores[name]['names'], [data['name']]])
        scores[name]['seeds'] = concat(scores[name]['seeds'], data['seed'])
        scores[name]['scores'] = concat(scores[name]['scores'], data['scores'])

        if labels is None:
            labels = data['labels']
        else:
            np.testing.assert_array_equal(labels, data['labels'])

    names, dicts = zip(*scores.items())
    all_scores = np.asarray([d['scores'] for d in dicts])
    names = np.asarray(names)

    dd.io.save(args.output, dict(scores=all_scores, names=names, labels=labels))
