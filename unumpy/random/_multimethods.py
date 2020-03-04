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


@create_numpy(_identity_argreplacer)
def rand(*tup):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def randn(*tup):
    return tup


@create_numpy(_dtype_argreplacer)
@all_of_type(ndarray)
def randint(low, high=None, size=None, dtype="l"):
    return mark_dtype(dtype)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def random_integers(low, high=None, size=None):
    return low


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def random_sample(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def random(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def ranf(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def sample(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def choice(a, size=None, replace=True, p=None):
    return (a, size, replace, p)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def bytes(length):
    return length


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def shuffle(x):
    return x


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def permutation(x):
    return x


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def beta(a, b, size=None):
    return (a, b, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def binomial(n, p, size=None):
    return (n, p, size)


@create_numpy(_identity_argreplacer)
@all_of_type(ndarray)
def chisquare(df, size=None):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def dirichlet(alpha, size=None):
    return (alpha, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def exponential(scale, size=None):
    return (scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def f(dfnum, dfden, size=None):
    return (dfnum, dfden, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def gamma(shape, scale=1.0, size=None):
    return (shape, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def geometric(p, size=None):
    return (p, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def gumbel(loc=0.0, scale=1.0, size=None):
    return (loc, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def hypergeometric(ngood, nbad, nsample, size=None):
    return (ngood, nbad, nsample, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def laplace(loc=0.0, scale=1.0, size=None):
    return (loc, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def logistic(loc=0.0, scale=1.0, size=None):
    return (loc, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def lognormal(mean=0.0, sigma=1.0, size=None):
    return (mean, sigma, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def logseries(p, size=None):
    return (p, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def multinomial(n, pvals, size=None):
    return (n, pvals, size)


# check
@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def multivariate_normal(mean, cov, size=None, check_valid=None, tol=None):
    return (mean, cov, size, check_valid, tol)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def negative_binomial(n, p, size=None):
    return (n, p, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def noncentral_chisquare(df, nonc, size=None):
    return (df, nonc, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def noncentral_f(dfnum, dfden, nonc, size=None):
    return (dfnum, dfden, nonc, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def normal(loc=0.0, scale=1.0, size=None):
    return (loc, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def pareto(a, size=None):
    return (a, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def poisson(lam=1.0, size=None):
    return (lam, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def power(a, size=None):
    return (a, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rayleigh(scale=1.0, size=None):
    return (scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_cauchy(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_exponential(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_gamma(shape, size=None):
    return (shape, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_normal(size=None):
    return size


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_t(df, size=None):
    return (df, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def triangular(left, mode, right, size=None):
    return (left, mode, right, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def uniform(low=0.0, high=1.0, size=None):
    return (low, high, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def vonmises(mu, kappa, size=None):
    return (mu, kappa, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def wald(mean, scale, size=None):
    return (mean, scale, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def weibull(a, size=None):
    return (a, size)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def zipf(a, size=None):
    return (a, size)


# discuss: RandomState


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def seed(seed=None):
    return seed


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def get_state():
    return


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def set_state(state):
    return state
