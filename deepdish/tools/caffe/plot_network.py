from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=8)
import pylab as plt
from pylab import cm
import argparse
import deepdish as dd
import numpy as np
import caffe
from vzlog import VzLog

def _blob_shape(blob):
    return (blob.num, blob.channels, blob.height, blob.width)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel', type=str)
    parser.add_argument('-o', '--output', default='log-network', type=str)

    args = parser.parse_args()

    vz = VzLog(args.output)

    mm = caffe.proto.caffe_pb2.NetParameter()
    with open(args.caffemodel, 'rb') as f:
        mm.ParseFromString(f.read())

    vz.title('Network')
    c = 0
    for layer in mm.layers:
        if layer.blobs:
            blob = layer.blobs[0]
            shape = _blob_shape(blob)

            X = np.asarray(blob.data).reshape(shape)

            vz.section(layer.name)
            vz.log('min', X.min(), 'max', X.max(), 'mean', X.mean(), 'std', X.std())
            vz.log('shape', X.shape)

            if c == 0:
                grid = dd.plot.ColorImageGrid(X.transpose(0, 2, 3, 1),
                                              vmin=None, vmax=None)
            else:
                grid = dd.plot.ImageGrid(X, cmap=cm.rainbow,
                                         vmin=None, vmax=None, vsym=True)

            grid.save(vz.impath(), scale=4)


            blob = layer.blobs[1]
            shape = _blob_shape(blob)

            X = np.asarray(blob.data).reshape(shape)

            # Plot biases
            vz.text('Bias')
            vz.log('shape', X.shape)

            plt.figure(figsize=(6, 1.5))
            plt.plot(X.ravel())
            plt.xlim((0, X.size-1))
            plt.savefig(vz.impath('svg'))
            plt.close()

            c += 1

if __name__ == '__main__':
    main()
