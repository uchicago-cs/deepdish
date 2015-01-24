from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')
import amitgroup as ag
import matplotlib.pylab as plt
import numpy as np
import deepdish as dd
import itertools as itr
import caffe
import re
import os

def _interesting_layer(layer):
    return re.match(r'conv\d+', layer) or re.match(r'ip\d+', layer)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('bare', type=str)
    parser.add_argument('caffemodel', type=str)
    parser.add_argument('-o', '--output', default='tmp.h5', type=str)
    parser.add_argument('-d', '--device', default=0, type=int)
    parser.add_argument('-c', '--count', default=5000, type=int)
    parser.add_argument('--layers', nargs='+', type=str)
    parser.add_argument('-n', '--name', type=str)
    # Recommended to allow automatic detection
    parser.add_argument('-s', '--seed', type=str)

    args = parser.parse_args()
    if args.name is None:
        name = args.caffemodel
    else:
        name = args.name

    if args.seed is None:
        pattern = re.compile(r'_s(\d+)')
        m = pattern.search(os.path.basename(args.caffemodel))
        if m:
            seed = int(m.group(1))
        else:
            raise ValueError('Could not automatically determine seed')
    else:
        seed = args.seed
    print('Seed:', seed)

    X = dd.io.load(args.data)
    if isinstance(X, dict):
        X = X['data']

    data = {}
    N = args.count

    net = caffe.Classifier(args.bare, args.caffemodel)
    net.set_mode_gpu()
    net.set_device(args.device)

    if args.layers is None:
        # Extract all interesting layers
        layers = [layer for layer in net._layer_names if _interesting_layer(layer)]
    else:
        layers = args.layers

    BATCH = 500
    MAX_PER_BATCH = 2500

    rs = np.random.RandomState(0)

    for ll in layers:
        print('Processing', args.caffemodel, ll)
        while True:
            Z = np.concatenate([net.forward(data=X[[i]], end=ll).values()[0].copy() for i in range(BATCH)])
            Z = Z.transpose(0, 2, 3, 1).reshape(-1, Z.shape[1])

            if Z.shape[0] > MAX_PER_BATCH:
                II = np.arange(Z.shape[0])
                rs.shuffle(II)
                Z = Z[II[:MAX_PER_BATCH]]

            if ll not in data:
                data[ll] = Z
            else:
                data[ll] = np.concatenate([data[ll], Z])

            print('{}/{}'.format(min(data[ll].shape[0], N), N))

            if data[ll].shape[0] >= N:
                data[ll] = data[ll][:N]
                break

    for key in data:
        data[key] = np.asarray(data[key])

    dd.io.save(args.output, dict(responses=data, name=name, seed=seed))
