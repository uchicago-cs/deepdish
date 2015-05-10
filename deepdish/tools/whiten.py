from __future__ import division, print_function, absolute_import

import deepdish as dd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('convert_data', nargs='*', type=str)
    parser.add_argument('--key', '-k', default='data', type=str)
    parser.add_argument('--epsilon', '-e', default=0.1, type=float)
    parser.add_argument('--method', '-m', default='zca', choices=('zca', 'element-wise'))
    #parser.add_argument('--output', '-o', default='whitened.h5', type=str)
    args = parser.parse_args()

    data = dd.io.load(args.data)
    X = data[args.key]
    if X.dtype == np.uint8:
        X = X.astype(np.float32) / 255

    convert_data = [args.data] + args.convert_data

    if args.method == 'zca':
        W = dd.util.zca_whitening_matrix(X, args.epsilon)

        for i, fn in enumerate(convert_data):
            print('Whitening', fn)
            # Possibly reloading X, could be optimized
            if i > 0:
                data = dd.io.load(fn)
                X = data[args.key]

            if X.dtype == np.uint8:
                X = X.astype(np.float32) / 255
            #data[args.key] = dd.util.whiten(X, args.epsilon)
            #dd.io.save(args.output, data)
            Z = dd.util.apply_whitening_matrix(X, W)
            data['data'] = Z
            dd.io.save('w_'+fn, data, compress=False)
    elif args.method == 'element-wise':
        Xmean = X.mean(0)
        Xstd = np.sqrt(X.var(0) + args.epsilon)
        import pdb; pdb.set_trace()
        #X = (X - X.mean(0)) / np.sqrt(X.var(0) + args.epsilon)
        for i, fn in enumerate(convert_data):
            print('Standardizing', fn)
            if i > 0:
                data = dd.io.load(fn)
                X = data[args.key]

            if X.dtype == np.uint8:
                X = X.astype(np.float32) / 255

            Z = (X - Xmean) / Xstd
            data['data'] = Z
            dd.io.save('s_'+fn, data, compress=False)


if __name__ == '__main__':
    main()
