from __future__ import division, print_function, absolute_import
import caffe
import deepdish as dd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('bare', type=str)
    parser.add_argument('caffemodel', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-d', '--device', default=0, type=int)

    args = parser.parse_args()

    net = caffe.Classifier(args.bare, args.caffemodel)
    net.set_mode_gpu()
    net.set_device(args.device)

    data = dd.io.load(args.data)

    x = data['data']
    y = data['label'].astype(int)

    yhat = net.forward_all(data=x).values()[0].squeeze((2, 3)).argmax(-1)

    if args.output:
        dd.io.save(args.output, dict(scores=yhat, labels=y))

    success = (yhat == y).mean()
    error = 1 - success

    print('Success: {:.2f}% / Error: {:.2f}%'.format(success * 100, error * 100))
