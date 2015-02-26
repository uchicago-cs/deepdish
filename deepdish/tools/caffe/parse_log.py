from __future__ import division, print_function, absolute_import 
import deepdish as dd
from collections import OrderedDict
import numpy as np
import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffelog', type=str)
    parser.add_argument('-o', '--output', default='loss.h5', type=str)
    parser.add_argument('--dataset', default='Test', type=str)
    parser.add_argument('--id', default=0, type=int)

    args = parser.parse_args()

    tid = args.id

    regex_iter = re.compile(r'Iteration (\d+), Testing net \(#(\d+)\)'.format(tid))
    gr = r'{} net output #(\d+): (\w+) = ([\d.]+)'.format(args.dataset)
    regex_loss = re.compile(gr)

    infos = []

    cur_info = 0
    cur_id = 0

    with open(args.caffelog) as f:
        infos = [
            [[], [], []]
        ]
        info = infos[cur_info]
        cur_iter = None
        cur_accuracy = None
        for line in f:
            m = regex_iter.search(line)
            if m:
                iter = int(m.group(1))
                cur_id = int(m.group(2))

                if iter < cur_iter:
                    cur_info += 1
                    infos.append([[], [], []])
                    info = infos[cur_info]

                cur_iter = iter
                continue

            m = regex_loss.search(line)
            if m and cur_id == tid:
                assert cur_iter is not None
                output = int(m.group(1))
                loss = float(m.group(3))

                if output == 0:
                    info[0].append(cur_iter)
                info[1+output].append(loss)
                if output == 0:
                    print(cur_iter, loss)

                continue
        infos.append(info)

    iterations = []
    rates = []
    losses = []
    for ii in infos:
        iterations.append(ii[0])
        rates.append(ii[1])
        if ii[2]:
            losses.append(ii[2])

    if not losses:
        raise Exception("Empty stuff")

    d = {}
    d['iterations'] = np.asarray(iterations)
    d['rates'] = np.asarray(rates)
    if losses:
        d['losses'] = np.asarray(losses)

    dd.io.save(args.output, d)


if __name__ == '__main__':
    main()
