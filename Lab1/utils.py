import re
from collections import UserDict

class ClobberedDictKey(Exception):
    "A flag that a variable has been assigned two incompatible values."
    pass

class NoClobberDict(UserDict):  # For Python 3

    """
    A dictionary-like object that prevents its values from being
    overwritten by different values. If that happens, it indicates a
    failure to match.
    """
    def __init__(self, initial_dict=None):
        if initial_dict is None:
            self._dict = {}
        else:
            self._dict = dict(initial_dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if key in self._dict and self._dict[key] != value:
            raise ClobberedDictKey((key, value))
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()


# Regular expression for finding variables of the form (?x)
AIRegex = re.compile(r'\(\?([a-zA-Z_][a-zA-Z0-9_]*)\)')

def AIStringToRegex(AIStr):
    # Use raw strings for the replacement part to avoid escape issues
    res = AIRegex.sub(r'(?P<\g<1>>\\S+)', AIStr) + '$'
    return res


def AIStringToPyTemplate(AIStr):
    return AIRegex.sub( r'%(\1)s', AIStr )


def AIStringVars(AIStr):
    # This is not the fastest way of doing things, but
    # it is probably the most explicit and robust
    return set([ AIRegex.sub(r'\1', x) for x in AIRegex.findall(AIStr) ])