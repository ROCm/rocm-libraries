# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare native XF32 input preparation with independent NumPy references."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import struct

import numpy as np

from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.gfx942.tf32_mma_probe import (
    PREPARATIONS,
    Tf32MmaProbeSpec,
    build_tf32_mma_probe,
)


def rne_bits(bits):
    """Independent integer quotient/remainder reference; quiet NaNs."""
    u = np.asarray(bits, dtype=np.uint32)
    sign = u & np.uint32(0x80000000)
    magnitude = u & np.uint32(0x7FFFFFFF)
    quotient = magnitude.astype(np.uint64) // 8192
    remainder = magnitude % 8192
    rounded = (
        quotient + ((remainder > 4096) | ((remainder == 4096) & (quotient % 2 == 1)))
    ) * 8192
    finite = rounded.astype(np.uint32) | sign
    special = u & np.uint32(0xFFFFE000)
    special = np.where(magnitude > 0x7F800000, special | np.uint32(0x400000), special)
    return np.where(magnitude < 0x7F800000, finite, special).astype(np.uint32)


def corpus(m):
    k = 128 // m
    # Exact values, ties (even/odd), either side, carry, normals, subnormals,
    # infinities, signed zeros, signaling and quiet NaNs, both signs.
    bits = np.array(
        [
            0,
            1,
            0x1000,
            0x7FFFFF,
            0x800000,
            0x801001,
            0x3F800000,
            0x3F800001,
            0x3F800FFF,
            0x3F801000,
            0x3F801001,
            0x3F802FFF,
            0x3F803000,
            0x3F803001,
            0x3FFFFFFF,
            0x7F7FFFFF,
            0x7F800000,
            0x7F800001,
            0x7FC00001,
        ],
        dtype=np.uint32,
    )
    bits = np.concatenate((bits, bits | np.uint32(0x80000000)))
    rng = np.random.default_rng(1714)
    finite = rng.uniform(-8, 8, 1024).astype(np.float32).view(np.uint32)
    bits = np.concatenate((bits, finite))
    single = len(bits) * 2
    a = np.zeros((single + 32, m, k), dtype=np.float32)
    b = np.zeros_like(a)
    c = np.zeros((len(a), m, m), dtype=np.float32)
    a[: len(bits), :, 0] = bits.view(np.float32)[:, None]
    b[: len(bits), :, 0] = 1
    a[len(bits) : single, :, 0] = 1
    b[len(bits) : single, :, 0] = bits.view(np.float32)[:, None]
    a[single:] = rng.uniform(-2, 2, a[single:].shape)
    b[single:] = rng.uniform(-2, 2, b[single:].shape)
    c[single:] = rng.uniform(-1, 1, c[single:].shape)
    # Asymmetric dense inputs and cancellation with a nonzero accumulator.
    a[-1, :, 1::2] = -a[-1, :, ::2]
    b[-1, :, 1::2] = b[-1, :, ::2]
    return a, b, c, bits, single


def same_values(a, b):
    # Compare every finite bit (including zero sign); NaN payloads are not an
    # arithmetic-output contract. Prepared input payloads are checked separately.
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def launch(blob, name, a, b, c):
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [
        a.copy(),
        b.copy(),
        c.copy(),
        np.full_like(c, np.nan),
        np.full(a.shape, 0xDEADBEEF, np.uint32),
        np.full(b.shape, 0xDEADBEEF, np.uint32),
    ]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        runtime.launch_blocking(
            module.get_function(name),
            (len(a), 1, 1),
            (64, 1, 1),
            struct.pack("6Q", *ptrs),
        )
        for array, ptr in zip(arrays[3:], ptrs[3:]):
            runtime.memcpy_d2h(
                (ctypes.c_ubyte * array.nbytes).from_buffer(array), ptr, array.nbytes
            )
        return arrays[3:]
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()


def run(output_dir, backend="both", shapes=(16, 32)):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir, resolved_lib_path
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch()
    if not arch or arch.split(":")[0] != "gfx942":
        raise RuntimeError(f"TF32 numerical validation requires gfx942, got {arch}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engines = ("python", "cpp") if backend == "both" else (backend,)
    native = None
    if "cpp" in engines:
        import rocke_engine as native  # Required: no fallback.
    summary = {
        "arch": arch,
        "numpy": np.__version__,
        "flavor": os.getenv("ROCKE_LLVM_FLAVOR"),
        "results": [],
    }
    for m in shapes:
        a, b, c, bits, single = corpus(m)
        np.savez(output_dir / f"inputs_{m}.npz", a=a, b=b, c=c)
        all_results = {}
        for engine in engines:
            results = {}
            for variant in (*PREPARATIONS, "masked"):
                mode = "prepacked" if variant == "masked" else variant
                spec = Tf32MmaProbeSpec(m, mode)
                kernel = build_tf32_mma_probe(spec)
                ir = serialize(kernel)
                if engine == "cpp":
                    ir_native = native.tf32_mma_probe_serialize_ir(m, mode)
                    assert ir == ir_native, (m, mode, "builder parity")
                    ll = native.lower_serialized_ir(
                        ir_native,
                        arch="gfx942",
                        flavor=os.getenv("ROCKE_LLVM_FLAVOR", "llvm23"),
                    )
                    assert ll == _lower_kernel_to_llvm_python(kernel, arch="gfx942"), (
                        m,
                        mode,
                        "lowerer parity",
                    )
                else:
                    ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
                stem = output_dir / f"{engine}_{m}_{variant}"
                stem.with_suffix(".ll").write_text(ll)
                stem.with_suffix(".hip").write_text(
                    lower_kernel_to_hip(kernel, arch="gfx942")
                )
                blob, _ = build_hsaco_from_llvm_ir(ll, isa="amdgcn-amd-amdhsa--gfx942")
                stem.with_suffix(".hsaco").write_bytes(blob)
                aa, bb = a, b
                if variant == "prepacked":
                    aa, bb = rne_bits(a.view(np.uint32)), rne_bits(b.view(np.uint32))
                elif variant == "masked":
                    aa, bb = a.view(np.uint32) & np.uint32(0xFFFFE000), b.view(
                        np.uint32
                    ) & np.uint32(0xFFFFE000)
                d, pa, pb = launch(blob, spec.name, aa, bb, c)
                expected_a = (
                    rne_bits(a.view(np.uint32))
                    if variant in ("rne", "prepacked")
                    else aa.view(np.uint32)
                )
                expected_b = (
                    rne_bits(b.view(np.uint32))
                    if variant in ("rne", "prepacked")
                    else bb.view(np.uint32)
                )
                assert np.array_equal(pa, expected_a), (
                    engine,
                    m,
                    variant,
                    "A preparation",
                )
                assert np.array_equal(pb, expected_b), (
                    engine,
                    m,
                    variant,
                    "B preparation",
                )
                np.savez(stem.with_suffix(".npz"), d=d, pa=pa, pb=pb)
                results[variant] = d
                # Dense reference uses float64 scalar products and a conservative
                # FP32 accumulation bound, proportional to sum(abs(products))+abs(C).
                ra, rb = expected_a, expected_b
                if variant in ("raw", "carrier", "masked"):
                    ra, rb = ra & np.uint32(0xFFFFE000), rb & np.uint32(0xFFFFE000)
                da, db = ra.view(np.float32)[single:].astype(np.float64), rb.view(
                    np.float32
                )[single:].astype(np.float64)
                ref = np.einsum("bik,bjk->bij", da, db) + c[single:].astype(np.float64)
                scale = np.einsum("bik,bjk->bij", abs(da), abs(db)) + abs(c[single:])
                error = abs(d[single:].astype(np.float64) - ref)
                assert np.all(
                    error <= 4 * (128 // m) * np.finfo(np.float32).eps * scale + 1e-7
                ), (engine, m, variant, "dense reference", float(error.max()))
                # Exact single-product normal results avoid reduction-order noise.
                input_bits = np.concatenate((bits, bits))
                expected = (
                    input_bits
                    if variant == "fp32"
                    else (
                        rne_bits(input_bits)
                        if variant in ("rne", "prepacked")
                        else input_bits & np.uint32(0xFFFFE000)
                    )
                )
                # +0 accumulation turns a negative-zero product into +0.
                expected = np.where(
                    (expected & 0x7FFFFFFF) == 0, np.uint32(0), expected
                ).astype(np.uint32)
                assert np.all(
                    same_values(
                        d[:single],
                        np.broadcast_to(
                            expected.view(np.float32)[:, None, None], d[:single].shape
                        ),
                    )
                ), (engine, m, variant, "single product")
                summary["results"].append(
                    {
                        "engine": engine,
                        "m": m,
                        "variant": variant,
                        "batches": len(a),
                        "prepared_words": int(pa.size + pb.size),
                        "dense_max_abs_error": float(error.max()),
                        "status": "pass",
                    }
                )
            for left, right in (
                ("raw", "carrier"),
                ("raw", "masked"),
                ("rne", "prepacked"),
            ):
                assert np.all(same_values(results[left], results[right])), (
                    engine,
                    m,
                    left,
                    right,
                )
            # Preserve all boundary observations, including denormal and NaN behavior.
            observations = [
                {
                    "input_bits": f"{int(u):08x}",
                    "rne_bits": f"{int(rne_bits(u)):08x}",
                    **{
                        v: f"{int(d[i,0,0].view(np.uint32)):08x}"
                        for v, d in results.items()
                    },
                }
                for i, u in enumerate(bits)
            ]
            for row in observations:
                original = np.array(int(row["input_bits"], 16), dtype=np.uint32).view(
                    np.float32
                )
                row["input_value"] = str(original)
                for variant in ("raw", "rne"):
                    actual_bits = int(row[variant], 16)
                    actual = np.array(actual_bits, dtype=np.uint32).view(np.float32)
                    if np.isfinite(original) and np.isfinite(actual):
                        row[f"{variant}_abs_error_vs_fp32"] = abs(
                            float(actual) - float(original)
                        )
                        # Within one sign, IEEE encoding order gives ULP distance.
                        if (actual_bits ^ int(row["input_bits"], 16)) < 0x80000000:
                            row[f"{variant}_ulp_error_vs_fp32"] = abs(
                                actual_bits - int(row["input_bits"], 16)
                            )
            (output_dir / f"{engine}_{m}_boundaries.json").write_text(
                json.dumps(observations, indent=2) + "\n"
            )
            counterexample = next(
                row
                for row in observations
                if row["raw"] != row["rne"] and row["input_bits"] == "3f801001"
            )
            print(engine, m, json.dumps(counterexample), flush=True)
            all_results[engine] = results
        if backend == "both":
            for variant in all_results["python"]:
                assert np.all(
                    same_values(
                        all_results["python"][variant], all_results["cpp"][variant]
                    )
                )
    summary["comgr"] = resolved_lib_path()
    summary["status"] = "pass"
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "cpp", "both"), default="both")
    parser.add_argument("--shape", choices=("16", "32", "all"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps({"status": "running"}) + "\n")
    try:
        result = run(
            args.output_dir,
            args.backend,
            (16, 32) if args.shape == "all" else (int(args.shape),),
        )
    except Exception as error:
        result_path.write_text(
            json.dumps({"status": "fail", "error": str(error)}) + "\n"
        )
        raise
    print(json.dumps({"status": result["status"], "variants": len(result["results"])}))


if __name__ == "__main__":
    main()
