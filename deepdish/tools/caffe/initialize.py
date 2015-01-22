from __future__ import division, print_function, absolute_import

from string import Template
import argparse
import deepdish as dd

def proximity(cmdline, mm, rs):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('style', default=1, type=int)
    parser.add_argument('prototxt', type=str)
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-i', '--input', default='model', type=str)
    parser.add_argument('-o', '--output', default='model', type=str)

    args = parser.parse_args()

    rs = np.random.RandomState(args.seed)

    mm = caffe.proto.caffe_pb2.NetParameter()

    with open(args.input + '.caffemodel', 'rb') as f:
        mm.ParseFromString(f.read())

    if args.style.startswith('proximity'):
        model = proximity(args.style, mm, rs)
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
