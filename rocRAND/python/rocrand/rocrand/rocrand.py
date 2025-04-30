# Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""rocRAND Python Wrapper"""

import os
import ctypes
import ctypes.util
from ctypes import *

import numbers
import numpy as np

from .hip import load_hip, HIP_PATHS, stream_synchronize
from .hip import empty, DeviceNDArray, device_pointer

from .utils import find_library, expand_paths

# Finalize is only supported by python >= 3.4
try:
  from weakref import finalize
except ImportError:
  from backports.weakref import finalize


ROCRAND_PATHS = [
        os.getenv("ROCRAND_PATH")
    ] + expand_paths(HIP_PATHS, ["", "rocrand"])

def load_rocrand():
    global rocrand

    try:
        rocrand = CDLL(find_library(ROCRAND_PATHS, "librocrand.so"))
    except OSError as e:
        raise ImportError("librocrand.so cannot be loaded: " + str(e))

    load_hip()

# Delay the loading of rocrand to the first use
# so no code is executed when loading this module
class _load_rocrand_on_access(object):
    def __getattribute__(self, name):
        global rocrand
        load_rocrand()
        return getattr(rocrand, name)

rocrand = _load_rocrand_on_access()

ROCRAND_RNG_PSEUDO_DEFAULT = 400
ROCRAND_RNG_PSEUDO_XORWOW = 401
ROCRAND_RNG_PSEUDO_MRG32K3A = 402
ROCRAND_RNG_PSEUDO_MTGP32 = 403
ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404
ROCRAND_RNG_PSEUDO_MRG31K3P = 405
ROCRAND_RNG_PSEUDO_LFSR113 = 406
ROCRAND_RNG_PSEUDO_MT19937 = 407
ROCRAND_RNG_PSEUDO_THREEFRY2_32_20 = 408
ROCRAND_RNG_PSEUDO_THREEFRY2_64_20 = 409
ROCRAND_RNG_PSEUDO_THREEFRY4_32_20 = 410
ROCRAND_RNG_PSEUDO_THREEFRY4_64_20 = 411
ROCRAND_RNG_QUASI_DEFAULT = 500
ROCRAND_RNG_QUASI_SOBOL32 = 501
ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
ROCRAND_RNG_QUASI_SOBOL64 = 504
ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 505

ROCRAND_STATUS_SUCCESS = 0
ROCRAND_STATUS_VERSION_MISMATCH = 100
ROCRAND_STATUS_NOT_CREATED = 101
ROCRAND_STATUS_ALLOCATION_FAILED = 102
ROCRAND_STATUS_TYPE_ERROR = 103
ROCRAND_STATUS_OUT_OF_RANGE = 104
ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105
ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
ROCRAND_STATUS_LAUNCH_FAILURE = 107
ROCRAND_STATUS_INTERNAL_ERROR = 108

ROCRAND_STATUS = {
    ROCRAND_STATUS_SUCCESS: (
        "ROCRAND_STATUS_SUCCESS",
        "Success"),
    ROCRAND_STATUS_VERSION_MISMATCH: (
        "ROCRAND_STATUS_VERSION_MISMATCH",
        "Header file and linked library version do not match"),
    ROCRAND_STATUS_NOT_CREATED: (
        "ROCRAND_STATUS_NOT_CREATED",
        "Generator was not created using rocrand_create_generator"),
    ROCRAND_STATUS_ALLOCATION_FAILED: (
        "ROCRAND_STATUS_ALLOCATION_FAILED",
        "Memory allocation failed during execution"),
    ROCRAND_STATUS_TYPE_ERROR: (
        "ROCRAND_STATUS_TYPE_ERROR",
        "Generator type is wrong"),
    ROCRAND_STATUS_OUT_OF_RANGE: (
        "ROCRAND_STATUS_OUT_OF_RANGE",
        "Argument out of range"),
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: (
        "ROCRAND_STATUS_LENGTH_NOT_MULTIPLE",
        "Length requested is not a multiple of dimension"),
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED: (
        "ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
        "GPU does not have double precision"),
    ROCRAND_STATUS_LAUNCH_FAILURE: (
        "ROCRAND_STATUS_LAUNCH_FAILURE",
        "Kernel launch failure"),
    ROCRAND_STATUS_INTERNAL_ERROR: (
        "ROCRAND_STATUS_INTERNAL_ERROR",
        "Internal library error")
}

def check_rocrand(status):
    if status != ROCRAND_STATUS_SUCCESS:
        raise RocRandError(status)


class RocRandError(Exception):
    """Run-time rocRAND error."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value in ROCRAND_STATUS:
            v, s = ROCRAND_STATUS[self.value]
        else:
            v, s = str(self.value), "Unknown error"
        return "{} ({})".format(s, v)


class RNG(object):
    """Random number generator base class."""

    def __init__(self, rngtype, offset=None, stream=None, is_host=False):
        self._gen = c_void_p()
        create_fun = rocrand.rocrand_create_generator_host if is_host else rocrand.rocrand_create_generator
        check_rocrand(create_fun(byref(self._gen), rngtype))
        finalize(self, RNG._finalize, self._gen)

        self._offset = 0
        if offset is not None:
            self.offset = offset

        self._stream = None
        if stream is not None:
            self.stream = stream
        self.is_host = is_host

    @classmethod
    def _finalize(cls, gen):
        check_rocrand(rocrand.rocrand_destroy_generator(gen))

    @property
    def offset(self):
        """Mutable attribute of the offset of random numbers sequence.

        Setting this attribute resets the sequence.
        """
        return self._offset

    @offset.setter
    def offset(self, offset):
        """Mutable attribute of HIP stream for all kernel launches of the generator.

        All functions will use this stream.
        *None* means default stream.
        """
        check_rocrand(rocrand.rocrand_set_offset(self._gen, c_ulonglong(offset)))
        self._offset = offset

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        check_rocrand(rocrand.rocrand_set_stream(self._gen, stream))
        self._stream = stream

    def _generate_device(self, gen_func, ary, size, *args):
        if size is None:
            size = ary.size
        elif size > ary.size:
            raise ValueError("requested size is greater than ary")

        if isinstance(ary, np.ndarray):
            dary, needs_conversion = empty(size, ary.dtype), True
        elif isinstance(ary, DeviceNDArray):
            dary, needs_conversion = ary, False
        else:
            raise TypeError("unsupported type {}".format(type(ary)))

        check_rocrand(gen_func(self._gen, device_pointer(dary), c_size_t(size), *args))

        if needs_conversion:
            dary.copy_to_host(ary)

    def _generate_host(self, gen_func, ary, size, *args):
        if size is None:
            size = ary.size
        elif size > ary.size:
            raise ValueError("requested size is greater than ary")

        if isinstance(ary, DeviceNDArray):
            raise TypeError("Generate called with a device-side array on a host-side generator. "
                            "For device arrays, instantiate a device-side generator")
        elif not isinstance(ary, np.ndarray):
            raise TypeError("unsupported type {}".format(type(ary)))

        check_rocrand(gen_func(self._gen, ctypes.c_void_p(ary.ctypes.data), c_size_t(size), *args))
        # If we're generating on the host, synchronize the kernel to match the
        # behavior of a device-side generator being called with a host array (_generate_device).
        stream_synchronize(self.stream)

    def _generate(self, gen_func, ary, size, *args):
        if self.is_host:
            self._generate_host(gen_func, ary, size, *args)
        else:
            self._generate_device(gen_func, ary, size, *args)

    def generate(self, ary, size=None):
        """Generates uniformly distributed integers.

        Generates **size** (if present) or **ary.size** uniformly distributed
        integers and saves them to **ary**.

        Supported **dtype** of **ary**: :class:`numpy.uint32`, :class:`numpy.int32`.

        :param ary:  NumPy array (:class:`numpy.ndarray`) or
                     HIP device-side array (:class:`DeviceNDArray`)
        :param size: Number of samples to generate, default to **ary.size**
        """
        if ary.dtype in (np.uint32, np.int32):
            self._generate(
                rocrand.rocrand_generate,
                ary, size)
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    def uniform(self, ary, size=None):
        """Generates uniformly distributed floats.

        Generates **size** (if present) or **ary.size** uniformly distributed
        floats and saves them to **ary**.

        Supported **dtype** of **ary**: :class:`numpy.float32`, :class:`numpy.float64`.

        Generated numbers are between 0.0 and 1.0, excluding 0.0 and
        including 1.0.

        :param ary:  NumPy array (:class:`numpy.ndarray`) or
                     HIP device-side array (:class:`DeviceNDArray`)
        :param size: Number of samples to generate, default to **ary.size**
        """
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_uniform,
                ary, size)
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_uniform_double,
                ary, size)
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    def normal(self, ary, mean, stddev, size=None):
        """Generates normally distributed floats.

        Generates **size** (if present) or **ary.size** normally distributed
        floats and saves them to **ary**.

        Supported **dtype** of **ary**: :class:`numpy.float32`, :class:`numpy.float64`.

        :param ary:    NumPy array (:class:`numpy.ndarray`) or
                       HIP device-side array (:class:`DeviceNDArray`)
        :param mean:   Mean value of normal distribution
        :param stddev: Standard deviation value of normal distribution
        :param size:   Number of samples to generate, default to **ary.size**
        """
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_normal,
                ary, size,
                c_float(mean), c_float(stddev))
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_normal_double,
                ary, size,
                c_double(mean), c_double(stddev))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    def lognormal(self, ary, mean, stddev, size=None):
        """Generates log-normally distributed floats.

        Generates **size** (if present) or **ary.size** log-normally distributed
        floats and saves them to **ary**.

        Supported **dtype** of **ary**: :class:`numpy.float32`, :class:`numpy.float64`.

        :param ary:    NumPy array (:class:`numpy.ndarray`) or
                       HIP device-side array (:class:`DeviceNDArray`)
        :param mean:   Mean value of log normal distribution
        :param stddev: Standard deviation value of log normal distribution
        :param size:   Number of samples to generate, default to **ary.size**
        """
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_log_normal,
                ary, size,
                c_float(mean), c_float(stddev))
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_log_normal_double,
                ary, size,
                c_double(mean), c_double(stddev))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    def poisson(self, ary, lmbd, size=None):
        """Generates Poisson-distributed integers.

        Generates **size** (if present) or **ary.size** Poisson-distributed
        integers and saves them to **ary**.

        Supported **dtype** of **ary**: :class:`numpy.uint32`, :class:`numpy.int32`.

        :param ary:   NumPy array (:class:`numpy.ndarray`) or
                      HIP device-side array (:class:`DeviceNDArray`)
        :param lmbd:  lambda for the Poisson distribution
        :param size:  Number of samples to generate, default to **ary.size**
        """
        if ary.dtype in (np.uint32, np.int32):
            self._generate(
                rocrand.rocrand_generate_poisson,
                ary, size,
                c_double(lmbd))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))


class PRNG(RNG):
    DEFAULT         = ROCRAND_RNG_PSEUDO_DEFAULT
    """Default pseudo-random generator type, :const:`XORWOW`"""
    XORWOW          = ROCRAND_RNG_PSEUDO_XORWOW
    """XORWOW pseudo-random generator type"""
    MRG31K3P        = ROCRAND_RNG_PSEUDO_MRG31K3P
    """MRG31k3p pseudo-random generator type"""
    MRG32K3A        = ROCRAND_RNG_PSEUDO_MRG32K3A
    """MRG32k3a pseudo-random generator type"""
    MTGP32          = ROCRAND_RNG_PSEUDO_MTGP32
    """Mersenne Twister MTGP32 pseudo-random generator type"""
    MT19937         = ROCRAND_RNG_PSEUDO_MT19937
    """Mersenne Twister pseudo-random generator type"""
    PHILOX4_32_10   = ROCRAND_RNG_PSEUDO_PHILOX4_32_10
    """PHILOX_4x32 (10 rounds) pseudo-random generator type"""
    LFSR113         = ROCRAND_RNG_PSEUDO_LFSR113
    """LFSR113 pseudo-random generator type"""
    THREEFRY2_32_20 = ROCRAND_RNG_PSEUDO_THREEFRY2_32_20
    """THREEFRY2_32_20 pseudo-random generator type"""
    THREEFRY2_64_20 = ROCRAND_RNG_PSEUDO_THREEFRY2_64_20
    """THREEFRY2_64_20 pseudo-random generator type"""
    THREEFRY4_32_20 = ROCRAND_RNG_PSEUDO_THREEFRY4_32_20
    """THREEFRY4_32_20 pseudo-random generator type"""
    THREEFRY4_64_20 = ROCRAND_RNG_PSEUDO_THREEFRY4_64_20
    """THREEFRY4_64_20 pseudo-random generator type"""

    def __init__(self, rngtype=DEFAULT, seed=None, offset=None, stream=None, is_host=False):
        """Creates a new pseudo-random number generator.

        A new pseudo-random number generator of type **rngtype** is initialized
        with given **seed**, **offset** and **stream**. If **is_host** is set to true, the generator
        operates on the host CPU and memory instead on the device. Note, that not all generators
        may support the host option.

        Values of **rngtype**:

        * :const:`DEFAULT`
        * :const:`XORWOW`
        * :const:`MRG31K3P`
        * :const:`MRG32K3A`
        * :const:`MTGP32`
        * :const:`MT19937`
        * :const:`PHILOX4_32_10`
        * :const:`LFSR113`
        * :const:`THREEFRY2_32_20`
        * :const:`THREEFRY2_64_20`
        * :const:`THREEFRY4_32_20`
        * :const:`THREEFRY4_64_20`

        :param rngtype: Type of pseudo-random number generator to create
        :param seed:    Initial seed value
        :param offset:  Initial offset of random numbers sequence
        :param stream:  HIP stream for all kernel launches of the generator
        :param is_host:    If True, the generator operates on the host CPU, not on the device

        Example::

            import rocrand
            import numpy as np

            gen = rocrand.PRNG(rocrand.PRNG.PHILOX4_32_10, seed=123456)
            a = np.empty(1000, np.int32)
            gen.poisson(a, 10.0)
            print(a)
        """
        super(PRNG, self).__init__(rngtype, offset=offset, stream=stream, is_host=is_host)

        self._seed = None
        if seed is not None:
            self.seed = seed

    @property
    def seed(self):
        """Mutable attribute of the seed of random numbers sequence.

        Setting this attribute resets the sequence.
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        check_rocrand(rocrand.rocrand_set_seed(self._gen, c_ulonglong(seed)))
        self._seed = seed


class QRNG(RNG):
    DEFAULT           = ROCRAND_RNG_QUASI_DEFAULT
    """Default quasi-random generator type, :const:`SOBOL32`"""
    SOBOL32           = ROCRAND_RNG_QUASI_SOBOL32
    """Sobol32 quasi-random generator type"""
    SCRAMBLED_SOBOL32           = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
    """Scrambled Sobol32 quasi-random generator type"""
    SOBOL64           = ROCRAND_RNG_QUASI_SOBOL64
    """Sobol64 quasi-random generator type"""
    SCRAMBLED_SOBOL64           = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64
    """Scrambled Sobol64 quasi-random generator type"""

    def __init__(self, rngtype=DEFAULT, ndim=None, offset=None, stream=None, is_host=False):
        """Creates a new quasi-random number generator.

        A new quasi-random number generator of type **rngtype** is initialized
        with given **ndim**, **offset** and **stream**. If **is_host** is set to true, the generator
        operates on the host CPU and memory instead on the device. Note, that not all generators
        may support the host option.

        Values of **rngtype**:

        * :const:`DEFAULT`
        * :const:`SOBOL32`
        * :const:`SCRAMBLED_SOBOL32`
        * :const:`SOBOL64`
        * :const:`SCRAMBLED_SOBOL64`

        Values of **ndim** are 1 to 20000.

        :param rngtype: Type of quasi-random number generator to create
        :param ndim:    Number of dimensions
        :param offset:  Initial offset of random numbers sequence
        :param stream:  HIP stream for all kernel launches of the generator
        :param is_host:    If True, the generator operates on the host CPU, not on the device

        Example::

            import rocrand
            import numpy as np

            gen = rocrand.QRNG(rocrand.QRNG.SOBOL32, ndim=4)
            a = np.empty(1000, np.float32)
            gen.normal(a, 0.0, 1.0)
            print(a)
        """

        super(QRNG, self).__init__(rngtype, offset=offset, stream=stream, is_host=is_host)

        self._ndim = 1
        if ndim is not None:
            self.ndim = ndim

    @property
    def ndim(self):
        """Mutable attribute of the number of dimensions of random numbers sequence.

        Supported values are 1 to 20000.
        Setting this attribute resets the sequence.
        """
        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        check_rocrand(rocrand.rocrand_set_quasi_random_generator_dimensions(self._gen, c_uint(ndim)))
        self._ndim = ndim


def get_version():
    """Returns the version number of the rocRAND library."""
    version = c_int(0)
    check_rocrand(rocrand.rocrand_get_version(byref(version)))
    return version.value
