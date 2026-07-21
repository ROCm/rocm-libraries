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

"""Minimal HIP wrapper"""

import os
import ctypes
import ctypes.util
from ctypes import *

import numbers
import numpy as np

from .utils import find_library

# Finalize is only supported by python >= 3.4
try:
  from weakref import finalize
except ImportError:
  from backports.weakref import finalize


class HipError(Exception):
    """Run-time HIP error."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

hipSuccess = 0
hipMemcpyDeviceToHost = 2

def check_hip(status):
    if status != hipSuccess:
        raise HipError(status)

hip = None

HIP_PATHS = [
    os.getenv("ROCM_PATH"),
    os.getenv("HIP_PATH"),
    "/opt/rocm",
    "/opt/rocm/hip"]

CUDA_PATHS = [
    os.getenv("CUDA_PATH"),
    "/opt/cuda"]

def load_hip():
    global hip

    loading_errors = []
    if hip is None:
        try:
            hip = CDLL(find_library(HIP_PATHS, "libamdhip64.so"), mode=RTLD_GLOBAL)
        except OSError as e:
            loading_errors.append(str(e))

    if hip is None:
        try:
            cuda = CDLL(find_library(CUDA_PATHS, "libcudart.so"), mode=RTLD_GLOBAL)
        except OSError as e:
            loading_errors.append(str(e))
        else:
            hip = cuda
            # Aliases
            hip.hipMalloc = cuda.cudaMalloc
            hip.hipFree = cuda.cudaFree
            hip.hipMemcpy = cuda.cudaMemcpy
            hip.hipStreamSynchronize = cuda.cudaStreamSynchronize

    if hip is None:
        raise ImportError("both libcudart.so and libamdhip64.so cannot be loaded: " +
                ", ".join(loading_errors))

class MemoryPointer(object):
    def __init__(self, nbytes):
        self.ptr = c_void_p()
        check_hip(hip.hipMalloc(byref(self.ptr), c_size_t(nbytes)))
        finalize(self, MemoryPointer._finalize, self.ptr)

    @classmethod
    def _finalize(cls, ptr):
        check_hip(hip.hipFree(ptr))

def device_pointer(dary):
    return dary.data.ptr

class DeviceNDArray(object):
    """Device-side array.

    This class is a limited version of :class:`numpy.ndarray` for device-side
    arrays.

    See :func:`empty`
    """

    def __init__(self, shape, dtype, data=None):
        """Create an empty device-side array.

        :param shape: Shape of the array (see :attr:`numpy.ndarray.shape`)
        :param dtype: Type of the array (see :attr:`numpy.ndarray.dtype`)
        :param data:  existing HIP device-side memory pointer
        """

        dtype = np.dtype(dtype)

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        shape = tuple(shape)

        size = np.prod(shape)

        self.shape = shape
        self.dtype = dtype
        self.size = size
        self.nbytes = self.dtype.itemsize * self.size

        if data is None:
            self.data = MemoryPointer(self.nbytes)
        else:
            self.data = data

    def copy_to_host(self, ary=None):
        """Copy from data device memory to host memory.

        If **ary** is passed then **ary** must have the same **dtype**
        and greater or equal **size** as **self** has.

        If **ary** is not passed then a new :class:`numpy.ndarray` will be
        created.

        :param ary: NumPy array (:class:`numpy.ndarray`)

        :returns: a new array or **ary**
        """
        if ary is None:
            ary = np.empty(self.shape, self.dtype)
        else:
            if self.dtype != ary.dtype:
                raise TypeError("self and ary must have the same dtype")
            if self.size > ary.size:
                raise ValueError("size of self must be less than size of ary")

        dst = ary.ctypes.data_as(c_void_p)
        src = device_pointer(self)
        check_hip(hip.hipMemcpy(dst, src, c_size_t(self.nbytes), hipMemcpyDeviceToHost))

        return ary

def empty(shape, dtype):
    """Create a new device-side array of given shape and type, without initializing entries.

    This function is a limited version of :func:`numpy.empty` for device-side
    arrays.

    Example::

        import hiprand
        import numpy as np

        gen = hiprand.QRNG(ndim=5)
        d_a = hiprand.empty((5, 10000), dtype=np.float32)
        gen.uniform(d_a)
        a = d_a.copy_to_host()
        print(a)

    See :class:`DeviceNDArray`

    :param shape: Shape of the array (see :attr:`numpy.ndarray.shape`)
    :type shape: int or tuple of int
    :param dtype: Type of the array (see :attr:`numpy.ndarray.dtype`)
    """
    return DeviceNDArray(shape, dtype)

def stream_synchronize(stream):
    """Blocks until all previously queued operations on the provided stream complete.

    :param stream: The HIP stream to synchronize.
    """
    check_hip(hip.hipStreamSynchronize(stream))
