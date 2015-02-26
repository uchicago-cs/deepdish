from __future__ import division, print_function, absolute_import
import argparse

def format_time(s, format='smhd'):
    ret = ''
    if 'd' in format:
        ss = 60 * 60 * 24
        days = s // ss
        s -= days * ss
        if days:
            ret += '{}d '.format(days)

    if 'h' in format:
        ss = 60 * 60
        hours = s // ss
        s -= hours * ss
        if hours:
            ret += '{}h '.format(hours)

    if 'm' in format:
        ss = 60
        minutes = s // ss
        s -= minutes * ss
        if minutes: 
            ret += '{}m '.format(minutes)

    if 's' in format:
        seconds = s
        if seconds:
            ret += '{}s'.format(seconds)

    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('start')
    parser.add_argument('end')
    args = parser.parse_args()

    with open(args.start) as f:
        t0 = int(f.read())
    with open(args.end) as f:
        t1 = int(f.read())

    diff = t1 - t0

    print('{} s'.format(diff))
    print('Elapsed time:', format_time(diff, format='smhd'))


if __name__ == '__main__':
    main()
