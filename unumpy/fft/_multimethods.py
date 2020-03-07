import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy")

from .._multimethods import ndarray, _identity_argreplacer, _self_argreplacer


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def fft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ifft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def fft2(a, s=None, axes=(-2, -1), norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def fftn(a, s=None, axes=None, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ifftn(a, s=None, axes=None, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rfft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def irfft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rfft2(a, s=None, axes=(-2, -1), norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def irfft2(a, s=None, axes=(-2, -1), norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rfftn(a, s=None, axes=None, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def irfftn(a, s=None, axes=None, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def hfft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ihfft(a, n=None, axis=-1, norm=None):
    return (a,)


@create_numpy(_identity_argreplacer)
@all_of_type(ndarray)
def fftfreq(n, d=1.0):
    return ()


@create_numpy(_identity_argreplacer)
@all_of_type(ndarray)
def rfftfreq(n, d=1.0):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def fftshift(x, axes=None):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ifftshift(x, axes=None):
    return (x,)
