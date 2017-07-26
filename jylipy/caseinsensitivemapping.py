class CaseInsensitiveMapping(collections.MutableMapping):
    '''
    A case-insensitive ``dict``-like object.
    Implements all methods and operations of
    ``collections.MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items``.
    All keys are expected to be strings. The structure remembers the
    case of the last key to be set, and ``iter(instance)``,
    ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive:
        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True
    For example, ``headers['content-encoding']`` will return the
    value of a ``'Content-Encoding'`` response header, regardless
    of how the header name was originally stored.
    If the constructor, ``.update``, or equality comparison
    operations are given keys that have equal ``.lower()``s, the
    behavior is undefined.

    v1.0.0 : JYL @PSI, 1/20/2015.
      Following https://github.com/kennethreitz/requests/blob/v1.2.3/requests/structures.py#L37
    '''
    def __init__(self, cls, data=None, **kwargs):
        self._store = cls()
        if data is None:
            data = cls()
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in list(self._store.values()))

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return (
            (lowerkey, keyval[1])
            for (lowerkey, keyval)
            in list(self._store.items())
        )

    def __eq__(self, other):
        if isinstance(other, collections.Mapping):
            other = type(self)(type(self._store),other)
        else:
            return NotImplemented
        # Compare insensitively
        return type(self._store)(self.lower_items()) == type(self._store)(other.lower_items())

    # Copy is required
    def copy(self):
        return type(self)(list(self._store.values()))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, type(self._store)(list(self.items())))


class CaseInsensitiveDict(CaseInsensitiveMapping):
    '''Case insensitive dict class'''
    def __init__(self, data=None, **kwargs):
        super(CaseInsensitiveDict, self).__init__(dict, data, **kwargs)


class CaseInsensitiveOrderedDict(CaseInsensitiveMapping):
    '''Case insensitive ordered dict class'''
    def __init__(self, data=None, **kwargs):
        super(CaseInsensitiveOrderedDict, self).__init__(collections.OrderedDict, data, **kwargs)
