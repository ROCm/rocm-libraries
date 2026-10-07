#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# launch_from_bundle.py -- take a CBOR bundle all the way to a verified GPU
# launch, with every launch decision coming from the C engine.
#
# This is the demonstration behind rocke/recipe_launch.h. The pure-C path used
# to stop at .ll: a client could get a correct kernel out of a bundle and still
# not know what to launch it with, because the grid lived in host Python. This
# driver proves that gap is closed, by refusing to let Python answer any of the
# questions a shipped client would have to answer for itself.
#
# WHAT IS AND IS NOT BEING PROVED
# -------------------------------
# Python is still in this process, and honesty about why matters more than the
# demo does. It is here as the HARNESS -- comgr and the HIP runtime are reached
# through ctypes because rocke has no C++ wrapper for either (see
# dsl_docs/runtime/comgr_and_hipmodule.md). Those are thin bindings over
# libamd_comgr and libamdhip64, which a C++ client such as hipDNN already links
# for itself.
#
# What Python is NOT allowed to do here is supply METADATA. The kernel name, the
# grid, the block, the kernarg offsets and the buffer size all come from
# rocke_bundle_plan_launch_cbor. Nothing is imported from the kernel family that
# authored the recipe, no elementwise_grid, no elementwise_signature. If the
# bundle did not carry enough to launch, this driver could not run -- which is
# the whole point, and why it packs the kernarg buffer by hand at the offsets
# the C engine reported instead of going through KernelLauncher.
#
#   python3 -m rocke.portable_ir.drivers.launch_from_bundle
#   python3 -m rocke.portable_ir.drivers.launch_from_bundle --n 5000 --op mul

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from typing import Any, Dict

import numpy as np

ARCH = "gfx950"
KEY = "elementwise"


def author(op: str, dtype: str) -> tuple:
    """Record the kernel and stamp its launch geometry into the recipe.

    The geometry is the same ceiling division the family's own dispatch does,
    but written as an intexpr over N so it travels with the artifact instead of
    staying behind in Python. `chunk` is baked, because a recorded recipe has
    fixed block_size and vec; N is the free one, supplied per call."""
    from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise
    from rocke.portable_ir.src import launch
    from rocke.portable_ir.src.recipe_bundle import build_bundle, cbor_encode
    from rocke.portable_ir.src.recording_builder import record_kernel

    spec = ElementwiseSpec(op=op, dtype=dtype)
    kernel, recipe = record_kernel(lambda: build_elementwise(spec))
    chunk = spec.elems_per_block()
    recipe = launch.attach_launch(
        recipe,
        grid=[{"div": [{"add": [{"spec": "N"}, chunk - 1]}, chunk]}, 1, 1],
        block=[spec.block_size, 1, 1],
    )
    bundle = cbor_encode(build_bundle([{"key": KEY, "arch": ARCH, "recipe": recipe}]))
    return spec, kernel.name, bundle, chunk


def pack_kernargs(plan: Dict[str, Any], values: Dict[str, Any]) -> bytes:
    """Build the kernarg buffer from the C engine's reported layout.

    Writes each value at plan['args'][i]['offset'] rather than concatenating in
    order. Those coincide only while every argument is the same width; this
    signature is three pointers and an i32, so it is a real test of whether the
    exported offsets are usable rather than merely present."""
    buf = bytearray(plan["kernarg_size"])
    for arg in plan["args"]:
        v = values[arg["name"]]
        if arg["kind"] == "pointer":
            raw = struct.pack("<Q", int(v))
        elif arg["kind"] == "i32":
            raw = struct.pack("<i", int(v))
        elif arg["kind"] == "i64":
            raw = struct.pack("<q", int(v))
        elif arg["kind"] == "f32":
            raw = struct.pack("<f", float(v))
        else:
            raise SystemExit(f"unhandled arg kind {arg['kind']!r}")
        buf[arg["offset"] : arg["offset"] + arg["size"]] = raw
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="add")
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--n", type=int, default=0, help="0 => run a sweep")
    args = ap.parse_args()

    from rocke.core.arch import ArchTarget
    from rocke.portable_ir.src import online
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.launcher import DeviceMem, _runtime

    spec, recorded_name, bundle, chunk = author(args.op, args.dtype)
    print(f"bundle: {len(bundle)} bytes   recorded as {recorded_name}")
    print(f"elems_per_block={chunk}  (grid must be ceil(N/{chunk}))\n")

    # Sizes that are NOT multiples of the slab, so a grid that forgot to round
    # up would leave a tail unwritten and the comparison would catch it.
    sizes = [args.n] if args.n else [chunk, chunk + 1, 4 * chunk, 5 * chunk - 3, 100000]
    rt = _runtime()
    rng = np.random.default_rng(0xC0FFEE)
    npdt = {"f16": np.float16, "bf16": np.float32, "f32": np.float32}[args.dtype]
    failures = 0

    for n in sizes:
        plan = online.plan_launch(bundle, KEY, arch=ARCH, ints={"N": n})
        geom = plan["geometry"]
        if geom is None:
            print("FAIL: bundle carries no launch geometry")
            return 1

        ll, _ = online.bundle_cbor_to_llvm(bundle, KEY, arch=ARCH, ints={"N": n})
        hsaco, _ = build_hsaco_from_llvm_ir(
            ll, isa=ArchTarget.from_gfx(ARCH).isa_triple, options=["-O3"]
        )

        A = rng.standard_normal(n).astype(npdt)
        B = rng.standard_normal(n).astype(npdt)
        C = np.zeros_like(A)
        dev = {}
        for name, arr in (("A", A), ("B", B), ("C", C)):
            mem = DeviceMem(arr.nbytes)
            rt.memcpy_h2d(
                mem.ptr(), (ctypes.c_ubyte * arr.nbytes).from_buffer(arr), arr.nbytes
            )
            dev[name] = mem

        kernargs = pack_kernargs(
            plan,
            {"A": dev["A"].ptr(), "B": dev["B"].ptr(), "C": dev["C"].ptr(), "N": n},
        )
        mod = rt.load_module(hsaco)
        fn = mod.get_function(plan["kernel_name"])
        rt.launch(
            fn,
            geom["grid"],
            geom["block"],
            kernargs,
            shared_bytes=geom["lds_bytes"],
        )
        rt.sync()
        rt.memcpy_d2h(
            (ctypes.c_ubyte * C.nbytes).from_buffer(C), dev["C"].ptr(), C.nbytes
        )

        ref = {"add": A + B, "mul": A * B, "sub": A - B}[args.op]
        bad = int(np.sum(C.astype(np.float32) != ref.astype(np.float32)))
        failures += bad != 0
        print(
            f"N={n:<8} grid={geom['grid']} block={geom['block']} "
            f"kernarg={plan['kernarg_size']}B  "
            f"{'OK' if bad == 0 else f'MISMATCH ({bad} elems)'}"
        )

    print()
    if failures:
        print("FAIL: the bundle's own geometry did not produce correct results")
        return 1
    print(
        "PASS: name, grid, block and kernarg layout all came from the C engine;\n"
        "      no kernel-family Python was imported to launch these."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
