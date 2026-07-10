# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level Python bindings for the hipBLASLt GEMM API."""

from . import _core
import numpy as _np

__all__ = ["_core"]
__version__ = _core.__version__

# Minimal numpy-native dtype map; extended with ml_dtypes in Phase 4.
_DTYPE_TO_NP = {
    _core.DataType.R_32F: _np.float32,
    _core.DataType.R_64F: _np.float64,
    _core.DataType.R_16F: _np.float16,
    _core.DataType.R_32I: _np.int32,
    _core.DataType.R_8I: _np.int8,
}


def _device_array_to_numpy(self):
    np_dtype = _DTYPE_TO_NP.get(self.dtype)
    if np_dtype is None:
        raise TypeError(
            f"to_numpy() does not support dtype {self.dtype!r}; "
            "use copy_to_host() with a uint8 buffer for narrow types"
        )
    out = _np.empty(tuple(self.shape), dtype=np_dtype)
    self.copy_to_host(out)
    return out


_core.DeviceArray.to_numpy = _device_array_to_numpy


# ---------------------------------------------------------------------------
# DLPack interop — escape hatch so DeviceArray can talk to torch/cupy-ROCm.
# These are monkey-patched onto the C++ class so the C++ side stays lean.
# ---------------------------------------------------------------------------

import ctypes as _ctypes
import ctypes.util as _ctu


def _load_libhip():
    name = _ctu.find_library("amdhip64") or "libamdhip64.so"
    try:
        return _ctypes.CDLL(name)
    except OSError:
        return None


_libhip = _load_libhip()


# DLPack ABI structs (v0.8) — defined once at module level so they are not
# reconstructed on every __dlpack__ call.
_kDLROCM = 10  # ROCm device type in the DLPack ABI
_kDLFloat = 2


class _DLDevice(_ctypes.Structure):
    _fields_ = [("device_type", _ctypes.c_int), ("device_id", _ctypes.c_int)]


class _DLDataType(_ctypes.Structure):
    _fields_ = [
        ("code", _ctypes.c_uint8),
        ("bits", _ctypes.c_uint8),
        ("lanes", _ctypes.c_uint16),
    ]


class _DLTensor(_ctypes.Structure):
    _fields_ = [
        ("data", _ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", _ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", _ctypes.POINTER(_ctypes.c_int64)),
        ("strides", _ctypes.POINTER(_ctypes.c_int64)),
        ("byte_offset", _ctypes.c_uint64),
    ]


class _DLManagedTensor(_ctypes.Structure):
    pass


# Deleter signature: void (*)(DLManagedTensor*)
_DeleterType = _ctypes.CFUNCTYPE(None, _ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", _ctypes.c_void_p),
    ("deleter", _DeleterType),
]


def _get_device_id(ptr_int):
    """Return the device id that owns the HIP pointer *ptr_int*.

    Uses hipPointerGetAttributes so the result reflects the pointer's actual
    device rather than the calling thread's current device (hipGetDevice).
    Falls back to hipGetDevice when hipPointerGetAttributes fails.
    """
    if _libhip is None:
        return 0

    class _HipPointerAttr(_ctypes.Structure):
        _fields_ = [
            ("memoryType", _ctypes.c_int),
            ("device", _ctypes.c_int),
            ("devicePointer", _ctypes.c_void_p),
            ("hostPointer", _ctypes.c_void_p),
            ("isManaged", _ctypes.c_int),
            ("allocationFlags", _ctypes.c_uint),
        ]

    attr = _HipPointerAttr()
    ret = _libhip.hipPointerGetAttributes(
        _ctypes.byref(attr), _ctypes.c_void_p(ptr_int)
    )
    if ret == 0:  # hipSuccess
        return attr.device
    # Fall back to hipGetDevice
    dev = _ctypes.c_int(0)
    _libhip.hipGetDevice(_ctypes.byref(dev))
    return dev.value


def _device_array_dlpack(self, stream=None):
    """Return a DLPack capsule for this DeviceArray.

    torch/cupy-ROCm consumers call ``torch.from_dlpack(da)`` which will
    invoke this method.  We build a minimal DLPack PyCapsule via ctypes so
    we do not need torch or cupy as build-time dependencies.

    Note: stream synchronisation is the caller's responsibility.  If the
    consumer passes a ``stream`` argument we ignore it here — callers that
    require strict ordering must synchronise before calling.
    """
    # Resolve the ROCm device id for the pointer (uses hipPointerGetAttributes).
    device_id = _get_device_id(self.ptr)

    # Map hipDataType → DLDataType (best-effort; covers common training types).
    _DTYPE_MAP = {
        _core.DataType.R_32F: _DLDataType(_kDLFloat, 32, 1),
        _core.DataType.R_64F: _DLDataType(_kDLFloat, 64, 1),
        _core.DataType.R_16F: _DLDataType(_kDLFloat, 16, 1),
    }
    dl_dtype = _DTYPE_MAP.get(self.dtype, _DLDataType(_kDLFloat, 32, 1))

    # Build shape array (heap-allocated; lifetime managed by DLManagedTensor).
    ndim = len(self.shape)
    shape_arr = (_ctypes.c_int64 * ndim)(*self.shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = _ctypes.c_void_p(self.ptr)
    managed.dl_tensor.device = _DLDevice(_kDLROCM, device_id)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = dl_dtype
    managed.dl_tensor.shape = _ctypes.cast(shape_arr, _ctypes.POINTER(_ctypes.c_int64))
    managed.dl_tensor.strides = None  # None → C-contiguous
    managed.dl_tensor.byte_offset = 0

    # Deleter: nothing to free — DeviceArray owns the GPU memory.
    @_DeleterType
    def _noop_deleter(ptr):
        pass  # DeviceArray.__del__ / .free() handles hipFree

    managed.deleter = _noop_deleter

    # Wrap in a PyCapsule named "dltensor" as the DLPack spec requires.
    pythonapi = _ctypes.pythonapi
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = _ctypes.py_object
    PyCapsule_New.argtypes = [_ctypes.c_void_p, _ctypes.c_char_p, _ctypes.c_void_p]

    managed_p = _ctypes.cast(_ctypes.addressof(managed), _ctypes.c_void_p)

    # Destructor: called by CPython when the capsule is finalized; removes the
    # pin entry so the ctypes objects can be collected.
    def _capsule_destructor(cap):
        _dlpack_pin.pop(id(cap), None)

    cap_destructor = _ctypes.CFUNCTYPE(None, _ctypes.py_object)(_capsule_destructor)

    capsule = PyCapsule_New(managed_p, b"dltensor", cap_destructor)

    # Pin the ctypes objects so GC does not collect them before the consumer
    # has a chance to read the capsule.  We attach them to the capsule object
    # using a side-channel dict keyed by capsule's id().  The capsule destructor
    # above removes the entry when CPython finalises the capsule.
    _dlpack_pin[id(capsule)] = (managed, shape_arr, _noop_deleter, cap_destructor)
    return capsule


# Side-channel storage: keeps ctypes objects alive until the PyCapsule is GC'd.
_dlpack_pin: dict = {}


def _device_array_dlpack_device(self):
    """Return (device_type, device_id) for the DLPack protocol.

    Returns (10, device_id) where 10 == kDLROCM.
    """
    return (_kDLROCM, _get_device_id(self.ptr))


@staticmethod
def _from_dlpack(obj):
    """Import a device tensor from an external framework.

    Accepts any object that exposes ``__cuda_array_interface__`` or
    ``__hip_array_interface__`` (torch-ROCm, cupy-ROCm) and copies the data
    into a new DeviceArray.  A true zero-copy borrow would require
    reference-counted ownership across the C++ boundary, which is deferred to
    a later task; the copy preserves correctness for now.

    Objects that expose only the pure DLPack protocol (``__dlpack__`` /
    ``__dlpack_device__``) are not yet supported — a ``NotImplementedError``
    with an explicit message is raised in that case.

    Raises ``NotImplementedError`` if neither array-interface nor DLPack is
    found on ``obj``.
    """
    import numpy as _np

    iface = getattr(obj, "__cuda_array_interface__",
                    getattr(obj, "__hip_array_interface__", None))
    if iface is None:
        # Check for pure DLPack before giving up.
        if hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"):
            raise NotImplementedError(
                "from_dlpack: pure DLPack protocol not yet supported; "
                "pass a tensor with __cuda_array_interface__ or "
                "__hip_array_interface__"
            )
        raise NotImplementedError(
            "from_dlpack: object has no __cuda_array_interface__, "
            "__hip_array_interface__, or __dlpack__.  "
            "Pass a torch-ROCm or cupy-ROCm tensor."
        )

    ptr = iface["data"][0]          # (pointer, read_only)
    shape = list(iface["shape"])
    typestr = iface.get("typestr", "<f4")

    _TYPESTR_TO_DTYPE = {
        "<f4": _core.DataType.R_32F,
        ">f4": _core.DataType.R_32F,
        "<f2": _core.DataType.R_16F,
        ">f2": _core.DataType.R_16F,
        "<f8": _core.DataType.R_64F,
        ">f8": _core.DataType.R_64F,
        "<i4": _core.DataType.R_32I,
        ">i4": _core.DataType.R_32I,
        "|i1": _core.DataType.R_8I,
        "|u1": _core.DataType.R_8I,
    }
    dtype = _TYPESTR_TO_DTYPE.get(typestr, _core.DataType.R_32F)

    # Compute nbytes using the actual element size from the typestr.
    _TYPESTR_ITEMSIZE = {
        "<f4": 4, ">f4": 4,
        "<f2": 2, ">f2": 2,
        "<f8": 8, ">f8": 8,
        "<i4": 4, ">i4": 4,
        "|i1": 1, "|u1": 1,
    }
    itemsize = _TYPESTR_ITEMSIZE.get(typestr, 4)
    nelems = int(_np.prod(shape)) if shape else 1
    nbytes = nelems * itemsize

    da = _core.DeviceArray._alloc(nbytes, dtype, [int(s) for s in shape])
    if _libhip is None:
        da.free()
        raise RuntimeError(
            "libamdhip64 not found; cannot copy device memory"
        )
    # hipMemcpyKind: hipMemcpyDeviceToDevice = 3
    rc = _libhip.hipMemcpy(
        _ctypes.c_void_p(da.ptr),
        _ctypes.c_void_p(ptr),
        _ctypes.c_size_t(nbytes),
        _ctypes.c_int(3),
    )
    if rc != 0:
        da.free()
        raise RuntimeError(f"hipMemcpy (D2D) failed with code {rc}")
    return da


_core.DeviceArray.__dlpack__ = _device_array_dlpack
_core.DeviceArray.__dlpack_device__ = _device_array_dlpack_device
_core.DeviceArray.from_dlpack = _from_dlpack
