import deepdish as dd

def compute(x):
    return 2 * x

if dd.parallel.main(__name__):
    values = range(20)

    for x in dd.parallel.imap(compute, values):
        print(x)
