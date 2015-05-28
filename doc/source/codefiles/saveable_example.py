import deepdish as dd

class Foo(dd.util.SaveableRegistry):
    def __init__(self, x):
        self.x = x

    @classmethod
    def load_from_dict(self, d):
        obj = Foo(d['x'])
        return obj

    def save_to_dict(self):
        return {'x': self.x}


@Foo.register('bar')
class Bar(Foo):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def load_from_dict(self, d):
        obj = Bar(d['x'], d['y'])
        return obj

    def save_to_dict(self):
        return {'x': self.x, 'y': self.y}
