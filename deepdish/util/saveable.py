from __future__ import division, print_function, absolute_import
from deepdish import io

_ERR_STR = "Must override load_from_dict for Saveable interface"


class Saveable(object):
    """
    Key-value coding interface for classes. Generally, this is an interface
    that make it possible to access instance members through keys (strings),
    instead of through named variables. What this interface enables, is to save
    and load an instance of the class to file. This is done by encoding it into
    a dictionary, or decoding it from a dictionary. The dictionary is then
    saved/loaded using :func:`deepdish.io.save`.
    """
    @classmethod
    def load(cls, path):
        """
        Loads an instance of the class from a file.

        Parameters
        ----------
        path : str
            Path to an HDF5 file.

        Examples
        --------
        This is an abstract data type, but let us say that ``Foo`` inherits
        from ``Saveable``. To construct an object of this class from a file, we
        do:

        >>> foo = Foo.load('foo.h5') #doctest: +SKIP
        """
        if path is None:
            return cls.load_from_dict({})
        else:
            d = io.load(path)
            return cls.load_from_dict(d)

    def save(self, path):
        """
        Saves an instance of the class using :func:`deepdish.io.save`.

        Parameters
        ----------
        path : str
            Output path to HDF5 file.
        """
        io.save(path, self.save_to_dict())

    @classmethod
    def load_from_dict(cls, d):
        """
        Overload this function in your subclass. It takes a dictionary and
        should return a constructed object.

        When overloading, you have to decorate this function with
        ``@classmethod``.

        Parameters
        ----------
        d : dict
            Dictionary representation of an instance of your class.

        Returns
        -------
        obj : object
            Returns an object that has been constructed based on the
            dictionary.
        """
        raise NotImplementedError(_ERR_STR)

    def save_to_dict(self):
        """
        Overload this function in your subclass. It should return a dictionary
        representation of the current instance.

        If you member variables that are objects, it is best to convert them to
        dictionaries before they are entered into your dictionary hierarchy.

        Returns
        -------
        d : dict
            Returns a dictionary representation of the current instance.
        """
        raise NotImplementedError(_ERR_STR)


class NamedRegistry(object):
    """
    This class provides a named hierarchy of classes, where each class is
    associated with a string name.
    """

    REGISTRY = {}

    @property
    def name(self):
        """Returns the name of the registry entry."""
        # Automatically overloaded by 'register'
        return "noname"

    @classmethod
    def register(cls, name):
        """Decorator to register a class."""
        def register_decorator(reg_cls):
            def name_func(self):
                return name
            reg_cls.name = property(name_func)
            assert issubclass(reg_cls, cls), \
                "Must be subclass matching your NamedRegistry class"
            cls.REGISTRY[name] = reg_cls
            return reg_cls
        return register_decorator

    @classmethod
    def getclass(cls, name):
        """
        Returns the class object given its name.
        """
        return cls.REGISTRY[name]

    @classmethod
    def construct(cls, name, *args, **kwargs):
        """
        Constructs an instance of an object given its name.
        """
        return cls.REGISTRY[name](*args, **kwargs)

    @classmethod
    def registry(cls):
        return cls.REGISTRY

    @classmethod
    def root(cls, reg_cls):
        """
        Decorate your base class with this, to create
        a new registry for it
        """
        reg_cls.REGISTRY = {}
        return reg_cls


class SaveableRegistry(Saveable, NamedRegistry):
    """
    This combines the features of :class:`deepdish.util.Saveable` and
    :class:`deepdish.util.NamedRegistry`.

    See also
    --------
    Saveable, NamedRegistry
    """
    @classmethod
    def load(cls, path):
        if path is None:
            return cls.load_from_dict({})
        else:
            d = io.load(path)
            # Check class type
            class_name = d.get('name')
            if class_name is not None:
                return cls.getclass(class_name).load_from_dict(d)
            else:
                return cls.load_from_dict(d)

    def save(self, path):
        d = self.save_to_dict()
        d['name'] = self.name
        io.save(path, d)
