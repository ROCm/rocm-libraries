# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch / AOT-cache helpers of the read-once (persistent grid) batchnorm kernels."""

from __future__ import annotations

import ctypes
import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rocke.dispatch.families.batchnorm import (  # noqa: F401
    RO_BLOCKS,
    RO_REGS,
    BatchNormRoCandidate,
)
from rocke.instances.common.batchnorm2d_ro import batchnorm2d_ro_sync_ints
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

MAX_SPIN = 500_000


def ro_label(cand, plan) -> str:
    return f"{cand.name}_sn{plan.sn}_sh{plan.sh}_g{plan.grid}"


def compile_ro_cached(
    cand: BatchNormRoCandidate, *, dtype: str, arch: str, cache_dir: str
):
    """Load / build the kernel of a read-once candidate (shape- and plan-free key)."""
    from rocke.helpers.compile import compile_kernel

    dtype = {"fp16": "f16", "fp32": "f32"}.get(dtype, dtype)
    spec = cand.spec(dtype, arch)
    kname = spec.kernel_name("bwd" if cand.backward else "train")
    raw = f"batchnorm2d_ro|dynamic|{dtype}|{arch}|{cand.name}|{kname}"
    path = Path(cache_dir) / (
        "batchnorm2d_ro_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24] + ".pkl"
    )
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    if os.environ.get("ROCKE_NORM_CACHE_ONLY") == "1":
        raise RuntimeError(
            f"AOT cache miss (ROCKE_NORM_CACHE_ONLY=1): {kname} -> {path}"
        )
    artifact = compile_kernel(cand.build(spec), arch=arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".pkl.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return artifact


_occ_cache: dict = {}


def resident_ctas(
    launcher: KernelLauncher, block_size: int, device: Optional[int] = None
) -> int:
    """CTAs of this kernel that can be co-resident on the device (occupancy x CUs)."""
    from rocke.runtime.hip_module import get_device_num_cus

    if device is None:
        from rocke.runtime.hip_module import _resolve_hip as _rh

        cur = ctypes.c_int(0)
        if _rh().hipGetDevice(ctypes.byref(cur)) != 0:
            raise RuntimeError("hipGetDevice failed")
        device = cur.value
    key = (launcher.kernel_name, block_size, device)
    if key in _occ_cache:
        return _occ_cache[key]
    from rocke.runtime.hip_module import _resolve_hip

    lib = _resolve_hip()
    fn = lib.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    nb = ctypes.c_int(0)
    rc = fn(ctypes.byref(nb), launcher._live_fn().p, block_size, 0)
    if rc != 0 or nb.value < 1:
        raise RuntimeError(f"occupancy query failed rc={rc} blocks={nb.value}")
    cus = get_device_num_cus(device) or 0
    if cus <= 0:
        raise RuntimeError("CU count unavailable")
    _occ_cache[key] = int(nb.value) * cus
    return _occ_cache[key]


def new_sync(c: int):
    """Zeroed barrier words (``sync``) and a one-element error counter (``err``) on the device."""
    from rocke.runtime._norm_buffer import NormBuffer

    return (
        NormBuffer((batchnorm2d_ro_sync_ints(c) + 64,), "i32"),
        NormBuffer((1,), "i32"),
    )


class RoRun:
    """Launches one compiled read-once kernel with a fixed plan on fixed buffers.

    ``bufs`` as for the two-launch runners (train: ``x gamma beta y rm rv mean
    invstd``; backward: ``x dy scale mean invstd dx dscale dbias``). ``sync`` is
    a zero-initialised int32 buffer of at least ``batchnorm2d_ro_sync_ints(c)``
    elements, ``err`` a one-element int32 buffer. ``memset=True`` re-zeroes the
    sync buffer before every launch (epoch stays 0); otherwise the epoch grows.
    """

    def __init__(
        self,
        cand,
        plan,
        art,
        *,
        n,
        c,
        h,
        w,
        bufs,
        sync,
        err,
        eps=1e-5,
        momentum=0.1,
        memset=False,
    ):
        self._keep = (bufs, sync, err)
        self.cand, self.plan, self.sync, self.err, self.memset = (
            cand,
            plan,
            sync,
            err,
            memset,
        )
        assert sync.numel() >= batchnorm2d_ro_sync_ints(c)
        self.launcher = KernelLauncher.from_artifact(art)
        cap = resident_ctas(self.launcher, cand.block_size)
        if plan.grid > cap:
            raise RuntimeError(
                f"plan grid {plan.grid} exceeds co-resident capacity {cap}: refused"
            )
        self.capacity = cap
        p = lambda k: bufs[k].data_ptr()  # noqa: E731
        self.v = {
            "SyncI": sync.data_ptr(),
            "SyncF": sync.data_ptr(),
            "Err": err.data_ptr(),
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "NChunk": plan.n_chunk,
            "HwChunk": plan.hw_chunk,
            "SplitHw": plan.sh,
            "SplitN": plan.sn,
            "Grid": plan.grid,
            "MaxSpin": MAX_SPIN,
        }
        if cand.backward:
            self.v.update(
                {
                    "X": p("x"),
                    "dY": p("dy"),
                    "Scale": p("scale"),
                    "Mean": p("mean"),
                    "InvStd": p("invstd"),
                    "dX": p("dx"),
                    "dScale": p("dscale"),
                    "dBias": p("dbias"),
                }
            )
        else:
            self.v.update(
                {
                    "X": p("x"),
                    "Gamma": p("gamma"),
                    "Beta": p("beta"),
                    "Y": p("y"),
                    "RunningMean": p("rm"),
                    "RunningVar": p("rv"),
                    "eps": eps,
                    "momentum": momentum,
                }
            )
            if "mean" in bufs:
                self.v.update({"Mean": p("mean"), "InvStd": p("invstd")})
        self.cfg = LaunchConfig(grid=(plan.grid, 1, 1), block=(cand.block_size, 1, 1))
        self.epoch = 0

    def __call__(self) -> None:
        if self.memset:
            self.sync.zero_()
            self.v["Epoch"] = 0
        else:
            self.v["Epoch"] = self.epoch
            self.epoch += 1
        self.launcher(self.v, config=self.cfg)
