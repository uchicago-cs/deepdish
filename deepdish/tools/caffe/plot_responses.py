from __future__ import division, print_function, absolute_import 
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=8)
import pylab as plt
from vzlog import VzLog
import deepdish as dd
import caffe
import numpy as np


def _blob_shape(blob):
    return (blob.num, blob.channels, blob.height, blob.width)


def find_layer(mm, layer):
    for i, l in enumerate(mm.layers):
        if l.name == layer:
            return i
    raise ValueError('Could not find layer')


def get_blobs(mm, layer):
    i = find_layer(mm, layer)
    ret = []
    for j, b in enumerate(mm.layers[i].blobs):
        sh = _blob_shape(b)
        ret.append(np.asarray(b.data).reshape(sh))
    return tuple(ret)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('responses', nargs='+', type=str)
    parser.add_argument('-t', '--title', default='', type=str)
    parser.add_argument('-o', '--output', default='log-responses', type=str)
    parser.add_argument('-m', '--model', nargs='+', type=str)
    parser.add_argument('-l', '--layers', nargs='?', type=str)
    parser.add_argument('-a', '--alpha', default=0.99, type=float)

    args = parser.parse_args()
    alpha = args.alpha

    vz = VzLog(args.output)
    vz.title(args.title)
    vz.log('alpha =', alpha)

    mms = []
    for fn in args.model:
        mm = caffe.proto.caffe_pb2.NetParameter()
        with open(fn, 'rb') as f:
            mm.ParseFromString(f.read())
        mms.append(mm)

    layers = args.layers
    plt.figure()
    for fn in args.responses:
        data = dd.io.load(fn)
        name = data['name']
        if layers is None:
            layers = data['layers']
        y = []
        ystd = []
        for l in layers:
            rs = []
            #for X in data['responses'][l]:
            X = data['responses'][l]
            C = np.corrcoef(X.T)
            print(l, X.shape)
            C[np.isnan(C)] = 0.0
            try:
                rs.append(np.where(np.sort(np.linalg.eigvals(C))[::-1].cumsum() / C.shape[0] > alpha)[0][0] / C.shape[0])
            except:
                break

            y.append(np.mean(rs))
            ystd.append(np.std(rs))
        plt.errorbar(np.arange(len(y)), y, yerr=ystd, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('<- Redundancy / Identity-like ->')
    plt.legend(loc=4)
    plt.ylim((0, 1))
    plt.savefig(vz.impath('svg'))
    plt.close()

    plt.figure()
    for fn in args.responses:
        data = dd.io.load(fn)
        name = data['name']
        if layers is None:
            layers = data['layers']
        y = []
        ystd = []
        for l in layers:
            rs = []
            #for X in data['responses'][l]:
            X = data['responses'][l]

            y.append(X.mean())
        plt.plot(np.arange(len(y)), y, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Mean')
    plt.legend(loc=4)
    plt.savefig(vz.impath('svg'))
    plt.close()

    plt.figure()
    for fn in args.responses:
        data = dd.io.load(fn)
        name = data['name']
        if layers is None:
            layers = data['layers']
        y = []
        ystd = []
        for l in layers:
            rs = []
            #for X in data['responses'][l]:
            X = data['responses'][l]

            y.append((X**2).mean())
        plt.plot(np.arange(len(y)), y, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Second moment')
    plt.ylim((0, None))
    plt.legend(loc=4)
    plt.savefig(vz.impath('svg'))
    plt.close()

    plt.figure()
    for fn in args.responses:
        data = dd.io.load(fn)
        name = data['name']
        if layers is None:
            layers = data['layers']
        y = []
        ystd = []
        for l in layers:
            rs = []
            #for X in data['responses'][l]:
            X = data['responses'][l]

            y.append(X.std())
        plt.plot(np.arange(len(y)), y, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Standard deviation')
    plt.ylim((0, None))
    plt.legend(loc=4)
    plt.savefig(vz.impath('svg'))
    plt.close()

    plt.figure()
    for mm in mms:
        y = []
        for l in layers:
            blobs = get_blobs(mm, l)
            y.append(blobs[0].mean())

        plt.plot(np.arange(len(y)), y, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Weight mean')
    plt.legend(loc=1)
    plt.savefig(vz.impath('svg'))
    plt.close()


    plt.figure()
    for mm in mms:
        y = []
        for l in layers:
            blobs = get_blobs(mm, l)
            y.append(blobs[0].std())

        plt.plot(np.arange(len(y)), y, label='{}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Weight s.d.')
    plt.legend(loc=1)
    plt.savefig(vz.impath('svg'))
    plt.close()


    plt.figure()
    for mm in mms:
        y = []
        y2 = []
        for l in layers:
            blobs = get_blobs(mm, l)
            b0 = blobs[0]
            print(l, blobs[0].shape)
            if l.startswith('conv'):
                N = b0.shape[0] * b0.shape[2] * b0.shape[3]
                M = b0.shape[1] * b0.shape[2] * b0.shape[3]
            else:
                N = b0.shape[2]
                M = b0.shape[3]
            y.append(N)
            y2.append(M)

        plt.plot(np.arange(len(y)), y, label='n={}'.format(name))
        plt.plot(np.arange(len(y2)), y2, label='m={}'.format(name))
        plt.xticks(np.arange(len(y)), layers)
    plt.ylabel('Size')
    plt.legend(loc=1)
    plt.savefig(vz.impath('svg'))
    plt.close()



if __name__ == '__main__':
    main()
