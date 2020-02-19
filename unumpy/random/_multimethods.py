import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy")

from .._multimethods import ndarray, _self_argreplacer


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rand(*tup):
    return tup
