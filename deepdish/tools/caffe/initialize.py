from __future__ import division, print_function, absolute_import

import argparse
import deepdish as dd
import numpy as np
import caffe
import itertools as itr

def _set_from_ndarray(blob, X):
    for i, x in enumerate(X.ravel()):
        blob.data[i] = x

def _blob_shape(blob):
    return (blob.num, blob.channels, blob.height, blob.width)

def _create_corr(S, alpha):
    CR = np.zeros((S*S, S*S))
    for i, (ax, ay, bx, by) in enumerate(itr.product(range(S), range(S), range(S), range(S))):
        d = np.sqrt((ax - bx)**2 + (ay - by)**2)
        ii = ax * S + ay
        jj = bx * S + by
        CR[ii, jj] = np.exp(-alpha * d)
    return CR

def smooth(cmdline, mm, rs):
    import json
    params = json.loads(cmdline.replace("'", '"'))

    std = params['std']
    print('std = {}'.format(std))

    for layer in mm.layers:
        if layer.blobs:
            print('Initializing layer {}'.format(layer.name))
            assert len(layer.blobs) == 2

            blob_filters = layer.blobs[0]
            shape = _blob_shape(blob_filters)
            if layer.name.startswith('conv'):
                mean = np.zeros(shape[2]*shape[2])
                cov = _create_corr(shape[2], 0.5)

                X = rs.multivariate_normal(mean=mean, cov=std*cov, size=shape[0]*shape[1])
            else:
                X = rs.normal(loc=0.0, scale=std, size=shape)
            _set_from_ndarray(blob_filters, X)

            blob_biases = layer.blobs[1]
            shape = _blob_shape(blob_biases)
            X = np.zeros(shape)
            _set_from_ndarray(blob_biases, X)

def gaussian0(cmdline, mm, rs):
    import json
    params = json.loads(cmdline.replace("'", '"'))

    std = params['std']
    print('std = {}'.format(std))

    for layer in mm.layers:
        if layer.blobs:
            print('Initializing layer {}'.format(layer.name))
            assert len(layer.blobs) == 2

            blob_filters = layer.blobs[0]
            shape = _blob_shape(blob_filters)
            if layer.name in ['conv2', 'conv4', 'conv6']:
                X = rs.normal(loc=0.0, scale=0.25, size=shape)

            else:
                X = rs.normal(loc=0.0, scale=std, size=shape)
            _set_from_ndarray(blob_filters, X)

            blob_biases = layer.blobs[1]
            shape = _blob_shape(blob_biases)
            X = np.zeros(shape)
            _set_from_ndarray(blob_biases, X)



def gaussian(cmdline, mm, rs):
    import json
    params = json.loads(cmdline.replace("'", '"'))

    std = params['std']
    print('std = {}'.format(std))

    for layer in mm.layers:
        if layer.blobs:
            print('Initializing layer {}'.format(layer.name))
            assert len(layer.blobs) == 2

            blob_filters = layer.blobs[0]
            shape = _blob_shape(blob_filters)
            if layer.name in ['conv2', 'conv4', 'conv7']:
                X = rs.normal(loc=0.0, scale=0.25, size=shape)

                if 0:
                    for i in range(shape[0]):
                        X[i, i, shape[2]//2, shape[3]//2] = 1.0


                        # Add another one
                        for k in range(6):
                            j = rs.randint(shape[0])
                            X[i, j, shape[2]//2, shape[3]//2] = 1.0

            elif layer.name in ['conv5', 'conv8']:
                #X = rs.normal(loc=0.0, scale=0.001, size=shape)
                X = np.zeros(shape)

                for i in range(shape[0]):
                    X[i, i, shape[2]//2, shape[3]//2] = 1.0

                    # Add another one
                    #for k in range(2):
                        #j = rs.randint(shape[0])
                        #X[i, j, shape[2]//2, shape[3]//2] = 1.0
            else:
                X = rs.normal(loc=0.0, scale=std, size=shape)
            _set_from_ndarray(blob_filters, X)

            blob_biases = layer.blobs[1]
            shape = _blob_shape(blob_biases)

            X = np.zeros(shape)
            _set_from_ndarray(blob_biases, X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('style', type=str)
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-i', '--input', default='model', type=str)
    parser.add_argument('-o', '--output', default='model', type=str)

    args = parser.parse_args()

    rs = np.random.RandomState(args.seed)

    mm = caffe.proto.caffe_pb2.NetParameter()

    with open(args.input + '.caffemodel', 'rb') as f:
        mm.ParseFromString(f.read())

    style, param_str = args.style.split('/', 1)

    if style == 'smooth':
        smooth(param_str, mm, rs)
    elif style == 'gaussian':
        gaussian(param_str, mm, rs)
    elif style == 'gaussian0':
        gaussian0(param_str, mm, rs)
    else:
        raise ValueError('Unknown initialization style')

    # Save caffemodel
    caffemodel_fn = args.output + '.caffemodel'
    with open(caffemodel_fn, 'wb') as f:
        f.write(mm.SerializeToString())

    # Now write a new solverstate
    ss = caffe.proto.caffe_pb2.SolverState()

    with open(args.input + '.solverstate', 'rb') as f:
        ss.ParseFromString(f.read())

    ss.learned_net = caffemodel_fn

    with open(args.output + '.solverstate', 'wb') as f:
        f.write(ss.SerializeToString())

if __name__ == '__main__':
    main()
