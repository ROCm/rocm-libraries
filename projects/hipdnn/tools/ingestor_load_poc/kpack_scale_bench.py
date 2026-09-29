# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Synthetic kpack scaling bench: writer (rocm_kpack Python, as hkp_pack calls it) and
reader (ROCm librocm_kpack C runtime, as hip-kernel-provider calls it) at N entries.

Blobs are the real gfx950 code objects from a built hip-kernel-provider kpack, cycled
to N distinct toc keys, so per-entry size matches production.

Usage:
  python kpack_scale_bench.py write <src.kpack> <N> <out.kpack>   # one N per process
  python kpack_scale_bench.py read  <archive.kpack> <libkpack.so>  # one open per process
Each prints one JSON line; peak RSS is the process's VmHWM.
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path


def rss_mib():
    # VmHWM is per-mm and resets at exec; ru_maxrss survives exec and would report
    # the launching process's high-water mark instead of this one's.
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmHWM missing")


def cmd_write(src, n, out):
    from rocm_kpack import compression as comp
    from rocm_kpack import kpack as kpack_mod

    src_archive = kpack_mod.PackedKernelArchive.read(src)
    arch = "gfx950"
    blobs = []
    for binary in src_archive.toc:
        if binary.startswith("_") or not isinstance(src_archive.toc[binary], dict):
            continue
        if arch in src_archive.toc[binary]:
            blobs.append(src_archive.get_kernel(binary, arch))
    assert blobs, "no gfx950 blobs in source archive"
    rss_after_src = rss_mib()

    t0 = time.perf_counter()
    archive = kpack_mod.PackedKernelArchive(
        group_name="scale",
        gfx_arch_family=arch,
        gfx_arches=[arch],
        compressor=comp.ZstdCompressor(compression_level=3),
    )
    for i in range(n):
        vk = f"variant_{i:07d}"
        prepared = archive.prepare_kernel(
            relative_path=vk,
            gfx_arch=arch,
            hsaco_data=blobs[i % len(blobs)],
            metadata={"variant_key": vk},
        )
        archive.add_kernel(prepared)
    t_add = time.perf_counter()
    archive.finalize_archive()
    t_fin = time.perf_counter()
    archive.write(out)
    t_write = time.perf_counter()

    print(
        json.dumps(
            {
                "mode": "write",
                "n": n,
                "src_blobs": len(blobs),
                "mean_uncompressed_bytes": sum(len(b) for b in blobs) / len(blobs),
                "add_s": round(t_add - t0, 3),
                "finalize_s": round(t_fin - t_add, 3),
                "write_s": round(t_write - t_fin, 3),
                "total_s": round(t_write - t0, 3),
                "archive_bytes": os.path.getsize(out),
                "peak_rss_mib": round(rss_mib(), 1),
                "rss_before_build_mib": round(rss_after_src, 1),
            }
        )
    )


def cmd_read(path, lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.kpack_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.kpack_open.restype = ctypes.c_int
    lib.kpack_get_binary_count.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.kpack_get_binary_count.restype = ctypes.c_int
    lib.kpack_get_binary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.kpack_get_binary.restype = ctypes.c_int
    lib.kpack_get_kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.kpack_get_kernel.restype = ctypes.c_int
    lib.kpack_free_kernel.argtypes = [ctypes.c_void_p]
    lib.kpack_close.argtypes = [ctypes.c_void_p]

    rss0 = rss_mib()
    h = ctypes.c_void_p()
    t0 = time.perf_counter()
    rc = lib.kpack_open(path.encode(), ctypes.byref(h))
    t_open = time.perf_counter()
    result = {
        "mode": "read",
        "archive": path,
        "open_rc": rc,
        "open_ms": round((t_open - t0) * 1e3, 2),
    }
    if rc == 0:
        rss_open = rss_mib()
        count = ctypes.c_size_t()
        lib.kpack_get_binary_count(h, ctypes.byref(count))
        name = ctypes.c_char_p()
        lib.kpack_get_binary(h, count.value - 1, ctypes.byref(name))
        data, size = ctypes.c_void_p(), ctypes.c_size_t()
        t1 = time.perf_counter()
        rc2 = lib.kpack_get_kernel(
            h, name.value, b"gfx950", ctypes.byref(data), ctypes.byref(size)
        )
        t2 = time.perf_counter()
        lib.kpack_free_kernel(data)
        lib.kpack_close(h)
        result.update(
            {
                "entries": count.value,
                "get_rc": rc2,
                "get_one_ms": round((t2 - t1) * 1e3, 3),
                "kernel_bytes": size.value,
                "rss_open_delta_mib": round(rss_open - rss0, 1),
                "peak_rss_mib": round(rss_mib(), 1),
            }
        )
    print(json.dumps(result))


if __name__ == "__main__":
    if sys.argv[1] == "write":
        cmd_write(Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        cmd_read(sys.argv[2], sys.argv[3])
