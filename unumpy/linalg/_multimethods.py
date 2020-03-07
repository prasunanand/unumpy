import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy")

from .._multimethods import (
    ndarray,
    _identity_argreplacer,
    _self_argreplacer,
    _dtype_argreplacer,
    mark_dtype,
)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def cholesky(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def qr(a, mode="reduced"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eig(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigh(a, UPLO="L"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigvals(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigvalsh(a, UPLO="L"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def norm(x, ord=None, axis=None, keepdims=False):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def cond(x, p=None):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def det(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def matrix_rank(M, tol=None, hermitian=False):
    return (M,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def slogdet(a):
    return (a,)
