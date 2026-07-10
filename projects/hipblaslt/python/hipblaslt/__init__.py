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

def _device_array_dlpack(self, stream=None):
    """Return a DLPack capsule for this DeviceArray.

    torch/cupy-ROCm consumers call ``torch.from_dlpack(da)`` which will
    invoke this method.  We build a minimal DLPack PyCapsule via ctypes so
    we do not need torch or cupy as build-time dependencies.

    Note: stream synchronisation is the caller's responsibility.  If the
    consumer passes a ``stream`` argument we ignore it here — callers that
    require strict ordering must synchronise before calling.
    """
    import ctypes

    # ----- dlpack ABI structs (v0.8) ----------------------------------------
    # kDLROCM = 10  (ROCm device type in the DLPack ABI)
    kDLROCM = 10
    kDLFloat = 2

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        pass

    # Deleter signature: void (*)(DLManagedTensor*)
    _DeleterType = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

    DLManagedTensor._fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DeleterType),
    ]

    # Resolve the ROCm device id for the pointer.
    device_id = ctypes.c_int(0)
    try:
        hip = ctypes.CDLL("libamdhip64.so.6")
        hip.hipGetDevice(ctypes.byref(device_id))
    except OSError:
        pass  # stay at 0

    # Map hipDataType → DLDataType (best-effort; covers common training types).
    _DTYPE_MAP = {
        _core.DataType.R_32F: DLDataType(kDLFloat, 32, 1),
        _core.DataType.R_64F: DLDataType(kDLFloat, 64, 1),
        _core.DataType.R_16F: DLDataType(kDLFloat, 16, 1),
    }
    dl_dtype = _DTYPE_MAP.get(self.dtype, DLDataType(kDLFloat, 32, 1))

    # Build shape array (heap-allocated; lifetime managed by DLManagedTensor).
    ndim = len(self.shape)
    shape_arr = (ctypes.c_int64 * ndim)(*self.shape)

    managed = DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(self.ptr)
    managed.dl_tensor.device = DLDevice(kDLROCM, device_id.value)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = dl_dtype
    managed.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None  # None → C-contiguous
    managed.dl_tensor.byte_offset = 0

    # Deleter: nothing to free — DeviceArray owns the GPU memory.
    @_DeleterType
    def _noop_deleter(ptr):
        pass  # DeviceArray.__del__ / .free() handles hipFree

    managed.deleter = _noop_deleter

    # Wrap in a PyCapsule named "dltensor" as the DLPack spec requires.
    pythonapi = ctypes.pythonapi
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    # Keep managed and shape_arr alive as long as the capsule exists by
    # stashing them on the capsule object via a closure held in _noop_deleter.
    # We cannot attach attributes to a PyCapsule, so we keep a module-level
    # registry keyed by capsule id.  This is safe because the DLPack consumer
    # is expected to consume the capsule exactly once (rename to "used_dltensor"
    # after consumption is the framework's job).
    managed_p = ctypes.cast(ctypes.addressof(managed), ctypes.c_void_p)
    capsule = PyCapsule_New(managed_p, b"dltensor", None)

    # Pin the ctypes objects so GC does not collect them before the consumer
    # has a chance to read the capsule.  We attach them to the capsule object
    # using a side-channel dict keyed by capsule's id().
    _dlpack_pin[id(capsule)] = (managed, shape_arr, _noop_deleter)
    return capsule


# Side-channel storage: keeps ctypes objects alive until the PyCapsule is GC'd.
_dlpack_pin: dict = {}


def _device_array_dlpack_device(self):
    """Return (device_type, device_id) for the DLPack protocol.

    Returns (10, device_id) where 10 == kDLROCM.
    """
    import ctypes
    device_id = ctypes.c_int(0)
    try:
        hip = ctypes.CDLL("libamdhip64.so.6")
        hip.hipGetDevice(ctypes.byref(device_id))
    except OSError:
        pass
    return (10, device_id.value)  # 10 = kDLROCM


@staticmethod
def _from_dlpack(obj):
    """Import a device tensor from an external framework.

    Accepts any object that exposes ``__cuda_array_interface__`` (torch-ROCm,
    cupy-ROCm) and copies the data into a new DeviceArray.  A true zero-copy
    borrow would require reference-counted ownership across the C++ boundary,
    which is deferred to a later task; the copy preserves correctness for now.

    Raises ``NotImplementedError`` if neither ``__cuda_array_interface__``
    nor ``__hip_array_interface__`` is found on ``obj``.
    """
    import ctypes
    import numpy as _np

    iface = getattr(obj, "__cuda_array_interface__",
                    getattr(obj, "__hip_array_interface__", None))
    if iface is None:
        raise NotImplementedError(
            "from_dlpack: object has no __cuda_array_interface__ or "
            "__hip_array_interface__.  Pass a torch-ROCm or cupy-ROCm tensor."
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
    try:
        hip = ctypes.CDLL("libamdhip64.so.6")
        # hipMemcpyKind: hipMemcpyDeviceToDevice = 3
        rc = hip.hipMemcpy(
            ctypes.c_void_p(da.ptr),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(3),
        )
        if rc != 0:
            da.free()
            raise RuntimeError(f"hipMemcpy (D2D) failed with code {rc}")
    except OSError as exc:
        da.free()
        raise RuntimeError("libamdhip64.so.6 not found; cannot copy device memory") from exc
    return da


_core.DeviceArray.__dlpack__ = _device_array_dlpack
_core.DeviceArray.__dlpack_device__ = _device_array_dlpack_device
_core.DeviceArray.from_dlpack = _from_dlpack
