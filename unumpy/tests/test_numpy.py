import pytest
import uarray as ua
import unumpy as np
import numpy as onp
import torch
import dask.array as da
import sparse
import unumpy.numpy_backend as NumpyBackend

import unumpy.torch_backend as TorchBackend
import unumpy.dask_backend as DaskBackend
import unumpy.sparse_backend as SparseBackend

ua.set_global_backend(NumpyBackend)

dtypes = ["int8", "int16", "int32", "float32", "float64"]
LIST_BACKENDS = [
    (NumpyBackend, (onp.ndarray, onp.generic, onp.random)),
    (DaskBackend, (da.Array, onp.generic)),
    (SparseBackend, (sparse.SparseArray, onp.ndarray, onp.generic)),
    pytest.param(
        (TorchBackend, torch.Tensor),
        marks=pytest.mark.xfail(reason="PyTorch not fully NumPy compatible."),
    ),
]


FULLY_TESTED_BACKENDS = [NumpyBackend, DaskBackend]

try:
    import unumpy.xnd_backend as XndBackend
    import xnd
    from ndtypes import ndt

    LIST_BACKENDS.append((XndBackend, xnd.xnd))
    FULLY_TESTED_BACKENDS.append(XndBackend)
except ImportError:
    XndBackend = None  # type: ignore
    LIST_BACKENDS.append(
        pytest.param(
            (None, None), marks=pytest.mark.skip(reason="xnd is not importable")
        )
    )

try:
    import unumpy.cupy_backend as CupyBackend
    import cupy as cp

    LIST_BACKENDS.append(pytest.param((CupyBackend, (cp.ndarray, cp.generic))))
except ImportError:
    LIST_BACKENDS.append(
        pytest.param(
            (None, None), marks=pytest.mark.skip(reason="cupy is not importable")
        )
    )


EXCEPTIONS = {
    (DaskBackend, np.in1d),
    (DaskBackend, np.intersect1d),
    (DaskBackend, np.setdiff1d),
    (DaskBackend, np.setxor1d),
    (DaskBackend, np.union1d),
    (DaskBackend, np.sort),
    (DaskBackend, np.argsort),
    (DaskBackend, np.lexsort),
    (DaskBackend, np.partition),
    (DaskBackend, np.argpartition),
    (DaskBackend, np.sort_complex),
    (DaskBackend, np.msort),
    (DaskBackend, np.searchsorted),
    (DaskBackend, np.random.rand),
    (DaskBackend, np.random.randn),
    (DaskBackend, np.random.ranf),
    (DaskBackend, np.random.sample),
    (DaskBackend, np.random.bytes),
    (DaskBackend, np.random.shuffle),
    (DaskBackend, np.random.dirichlet),
    (DaskBackend, np.random.multivariate_normal),
    (DaskBackend, np.random.get_state),
}


@pytest.fixture(scope="session", params=LIST_BACKENDS)
def backend(request):
    backend = request.param
    return backend


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.add, ([1], [2]), {}),  # type: ignore
        (np.sin, ([1.0],), {}),  # type: ignore
        (np.arange, (5, 20, 5), {}),
        (np.arange, (5, 20), {}),
        (np.arange, (5,), {}),
    ],
)
def test_ufuncs_coerce(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)
    if isinstance(ret, da.Array):
        ret.compute()


def replace_args_kwargs(method, backend, args, kwargs):
    instance = ()
    while not hasattr(method, "_coerce_args"):
        instance += (method,)
        method = method.__call__

        if method is method.__call__:
            raise ValueError("Nowhere up the chain is there a multimethod.")

    args, kwargs, *_ = method._coerce_args(
        backend, instance + args, kwargs, coerce=True
    )
    return args[len(instance) :], kwargs


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.ndim, ([1, 2],), {}),
        (np.shape, ([1, 2],), {}),
        (np.size, ([1, 2],), {}),
        (np.any, ([True, False],), {}),
        (np.all, ([True, False],), {}),
        (np.min, ([1, 3, 2],), {}),
        (np.max, ([1, 3, 2],), {}),
        (np.argmin, ([1, 3, 2],), {}),
        (np.argmax, ([1, 3, 2],), {}),
        (np.nanargmin, ([1, 3, 2],), {}),
        (np.nanargmax, ([1, 3, 2],), {}),
        (np.nanmin, ([1, 3, 2],), {}),
        (np.nanmax, ([1, 3, 2],), {}),
        (np.ptp, ([1, 3, 2],), {}),
        (np.unique, ([1, 2, 2],), {}),
        (np.in1d, ([1], [1, 2, 2]), {}),
        (np.isin, ([1], [1, 2, 2]), {}),
        (np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.setdiff1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.setxor1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.sort, ([3, 1, 2, 4],), {}),
        (np.lexsort, (([1, 2, 2, 3], [3, 1, 2, 1]),), {}),
        (np.stack, (([1, 2], [3, 4]),), {}),
        (np.concatenate, (([1, 2, 3], [3, 4]),), {}),
        (np.broadcast_to, ([1, 2], (2, 2)), {}),
        (np.argsort, ([3, 1, 2, 4],), {}),
        (np.msort, ([3, 1, 2, 4],), {}),
        (np.sort_complex, ([3.0 + 1.0j, 1.0 - 1.0j, 2.0 - 3.0j, 4 - 3.0j],), {}),
        (np.partition, ([3, 1, 2, 4], 2), {}),
        (np.argpartition, ([3, 1, 2, 4], 2), {}),
        (np.transpose, ([[3, 1, 2, 4]],), {}),
        (np.swapaxes, ([[1, 2, 3]], 0, 1), {}),
        (np.rollaxis, ([[1, 2, 3], [1, 2, 3]], 0, 1), {}),
        (np.moveaxis, ([[1, 2, 3], [1, 2, 3]], 0, 1), {}),
        (np.column_stack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.hstack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.vstack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.block, ([([1, 2, 3]), ([1, 2, 3])],), {}),
        (np.reshape, ([[1, 2, 3], [1, 2, 3]], (6,)), {}),
        (np.argwhere, ([[3, 1, 2, 4]],), {}),
        (np.ravel, ([[3, 1, 2, 4]],), {}),
        (np.flatnonzero, ([[3, 1, 2, 4]],), {}),
        (np.where, ([[True, False, True, False]], [[1]], [[2]]), {}),
        (np.pad, ([1, 2, 3, 4, 5], (2, 3), "constant"), dict(constant_values=(4, 6))),
        (np.searchsorted, ([1, 2, 3, 4, 5], 2), {}),
        (np.compress, ([True, False, True, False], [0, 1, 2, 3]), {}),
        # the following case tests the fix in Quansight-Labs/unumpy#36
        (np.compress, ([False, True], [[1, 2], [3, 4], [5, 6]], 1), {}),
        (np.extract, ([True, False, True, False], [0, 1, 2, 3]), {}),
        (np.count_nonzero, ([True, False, True, False],), {}),
        (np.linspace, (0, 100, 200), {}),
        (np.logspace, (0, 4, 200), {}),
        (np.diff, ([1, 3, 2],), {}),
    ],
)
def test_functions_coerce(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if method is np.shape:
        assert isinstance(ret, tuple)
    elif method in (np.ndim, np.size):
        assert isinstance(ret, int)
    else:
        assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.prod, ([1],), {}),
        (np.sum, ([1],), {}),
        (np.std, ([1, 3, 2],), {}),
        (np.var, ([1, 3, 2],), {}),
    ],
)
def test_functions_coerce_with_dtype(backend, method, args, kwargs):
    backend, types = backend
    for dtype in dtypes:
        try:
            with ua.set_backend(backend, coerce=True):
                kwargs["dtype"] = dtype
                ret = method(*args, **kwargs)
        except ua.BackendNotImplementedError:
            if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
                raise
            pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)
    if XndBackend is not None and backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.broadcast_arrays, ([1, 2], [[3, 4]]), {}),
        (np.meshgrid, ([1, 2, 3], [4, 5], [0, 1]), {}),
        (np.nonzero, ([3, 1, 2, 4],), {}),
        (np.where, ([[3, 1, 2, 4]],), {}),
        (np.gradient, ([[0, 1, 2], [3, 4, 5], [6, 7, 8]],), {}),
    ],
)
def test_multiple_output(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert all(isinstance(arr, types) for arr in ret)

    for arr in ret:
        if isinstance(arr, da.Array):
            arr.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.eye, (2,), {}),
        (np.full, ((1, 2, 3), 1.3), {}),
        (np.ones, ((1, 2, 3),), {}),
        (np.zeros, ((1, 2, 3),), {}),
    ],
)
def test_array_creation(backend, method, args, kwargs):
    backend, types = backend
    for dtype in dtypes:
        try:
            with ua.set_backend(backend, coerce=True):
                kwargs["dtype"] = dtype
                ret = method(*args, **kwargs)
        except ua.BackendNotImplementedError:
            if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
                raise
            pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()
    if XndBackend is not None and backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.random.rand, (1, 2), {}),
        (np.random.randn, (1, 2), {}),
        (np.random.randint, ([1, 2],), {}),
        (np.random.random_integers, (1, 2), {}),
        (np.random.random_sample, (), {}),
        (np.random.random, ((2, 2),), {}),
        (np.random.ranf, (), {}),
        (np.random.sample, (), {}),
        (np.random.choice, ([1, 2],), {}),
        (np.random.bytes, (10,), {}),
        (np.random.shuffle, ([10, 11, 12],), {}),
        (np.random.permutation, ([10, 11, 12],), {}),
        (np.random.beta, (1, 2), {}),
        (np.random.binomial, (10, 0.5, 1000), {}),
        (np.random.chisquare, (1, 2), {}),
        (np.random.dirichlet, ((10, 5, 3),), {}),
        (np.random.exponential, (1, 2), {}),
        (np.random.f, (1, 2), {}),
        (np.random.gamma, (1, 2), {}),
        (np.random.geometric, (1, 2), {}),
        (np.random.gumbel, (1, 2), {}),
        (np.random.hypergeometric, (100, 2, 10), {}),
        (np.random.laplace, (1, 2), {}),
        (np.random.logistic, (1, 2), {}),
        (np.random.lognormal, (1, 2), {}),
        (np.random.logseries, (0.6, 1000), {}),
        (np.random.multinomial, (20, [1 / 6.0] * 6), {}),
        (np.random.multivariate_normal, ([0, 0], [[1, 0], [0, 100]], 5000), {}),
        (np.random.negative_binomial, (1, 0.1, 100000), {}),
        (np.random.noncentral_chisquare, (3, 20, 100000), {}),
        (np.random.noncentral_f, (3, 20, 3.0, 1000000), {}),
        (np.random.normal, (), {}),
        (np.random.pareto, (3, 1000), {}),
        (np.random.poisson, (), {}),
        (np.random.power, (5, 1000), {}),
        (np.random.rayleigh, (), {}),
        (np.random.standard_cauchy, (), {}),
        (np.random.standard_exponential, (), {}),
        (np.random.standard_gamma, (2.0, 1000000), {}),
        (np.random.standard_normal, (), {}),
        (np.random.standard_t, ([10]), {}),  # iterable: discuss
        (np.random.triangular, (-3, 0, 8, 100000), {}),
        (np.random.uniform, (-1, 0, 1000), {}),
        (np.random.vonmises, (0.0, 4.0, 1000), {}),
        (np.random.wald, (3, 2, 100000), {}),
        (np.random.weibull, (5.0, 1000), {}),
        (np.random.zipf, (2.0, 1000), {}),
        (np.random.seed, (), {}),
        (np.random.get_state, (), {}),
        # (np.random.set_state, (1, 2), {}),
    ],
)
def test_random(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if (
            backend in FULLY_TESTED_BACKENDS
            and (backend, method) not in EXCEPTIONS
            and backend is not XndBackend
        ):
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()
