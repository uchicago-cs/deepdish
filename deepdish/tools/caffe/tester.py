from __future__ import division, print_function, absolute_import
import caffe
import deepdish as dd
import argparse
import re
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('bare', type=str)
    parser.add_argument('caffemodel', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-d', '--device', default=0, type=int)
    parser.add_argument('-n', '--name', type=str)
    # Recommended to allow automatic detection
    parser.add_argument('-s', '--seed', type=str)

    args = parser.parse_args()

    data = dd.io.load(args.data)
    net = caffe.Classifier(args.bare, args.caffemodel)
    net.set_mode_gpu()
    net.set_device(args.device)


    x = data['data']
    y = data['label'].astype(int)
    if args.name is None:
        name = args.caffemodel
    else:
        name = args.name

    if args.seed is None:
        pattern = re.compile(r'_s(\d+)_')
        m = pattern.search(os.path.basename(args.caffemodel))
        if m:
            seed = int(m.group(1))
        else:
            raise ValueError('Could not automatically determine seed')
    else:
        seed = args.seed
    print('Seed:', seed)

    scores = net.forward_all(data=x).values()[0].squeeze((2, 3))
    yhat = scores.argmax(-1)
    if args.output:
        dd.io.save(args.output, dict(scores=scores, labels=y, name=name, seed=seed))

    success = (yhat == y).mean()
    error = 1 - success

    print('Success: {:.2f}% / Error: {:.2f}%'.format(success * 100, error * 100))
