def compute(x):
    return 2 * x

if __name__ == '__main__':
    values = range(20)

    for x in map(compute, values):
        print(x)
