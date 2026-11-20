# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit UniversalGemm compile-and-launch regression gate, not a quick unit test."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import struct
import sys

from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def tile_copy_spec(trait: TraitSpec) -> UniversalGemmSpec:
    return UniversalGemmSpec(
        name="universal_gemm_tile_copy",
        tile=TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=trait,
    )


def tile_copy_cases():
    return (
        ("coalesced", TraitSpec(pipeline="mem", epilogue="default"), "single", (96,)),
        (
            "clipped_tail",
            TraitSpec(pipeline="mem", epilogue="default", pad_k=True),
            "single",
            (65, 91),
        ),
        (
            "preshuffled_ping_pong",
            TraitSpec(pipeline="compv4", epilogue="default", preshuffle_b=True),
            "sync",
            (96,),
        ),
        (
            "preshuffled_tail",
            TraitSpec(
                pipeline="mem", epilogue="default", pad_k=True, preshuffle_b=True
            ),
            "single",
            (91,),
        ),
        (
            "sync_ping_pong",
            TraitSpec(pipeline="compv4", epilogue="default"),
            "sync",
            (32, 64, 96),
        ),
        (
            "dtl_ping_pong",
            TraitSpec(
                pipeline="compv4",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=True,
            ),
            "async",
            (32, 64, 96),
        ),
        (
            "sync_swizzled",
            TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
            "sync",
            (96,),
        ),
        (
            "dtl_swizzled",
            TraitSpec(
                pipeline="compv4",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=True,
                lds_swizzle=True,
            ),
            "async",
            (96,),
        ),
        (
            "split_k_edges",
            TraitSpec(
                pipeline="mem", epilogue="default", split_k=2, pad_m=True, pad_n=True
            ),
            "single",
            (128,),
        ),
        (
            "cshuffle_full",
            TraitSpec(pipeline="mem", epilogue="cshuffle"),
            "single",
            (96,),
        ),
        (
            "cshuffle_edges",
            TraitSpec(pipeline="mem", epilogue="cshuffle", pad_m=True, pad_n=True),
            "single",
            (96,),
        ),
    )


def _launch_and_compare(rt, spec, artifact, rng, label, k):
    import numpy as np

    split_k = spec.trait.split_k
    m = 67 if spec.trait.pad_m else 128
    n = 131 if spec.trait.pad_n else 192
    edges = spec.trait.pad_m or spec.trait.pad_n
    tile_m, tile_n = spec.tile.tile_m, spec.tile.tile_n
    gm, gn = (m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n
    storage_dtype = np.float64 if spec.data.dtype_a == "fp64" else np.float16
    a = rng.integers(-3, 4, (m, k), dtype=np.int16).astype(storage_dtype)
    b = rng.integers(-3, 4, (n, k), dtype=np.int16).astype(storage_dtype)
    a_storage, b_storage = a, b
    output_dtype = np.float32 if split_k > 1 else storage_dtype
    initial = 0 if split_k > 1 else np.nan
    if edges:
        # Pad physical input rows but keep the logical M/N bounds. Nonzero
        # padding makes a missing store/atomic guard corrupt C or the canary.
        a_storage = np.full((gm * tile_m, k), 3, dtype=a.dtype)
        b_storage = np.full((gn * tile_n, k), 5, dtype=b.dtype)
        a_storage[:m], b_storage[:n] = a, b
        output = np.full(gm * tile_m * gn * tile_n, 137, dtype=output_dtype)
        c = output[: m * n].reshape(m, n)
        c.fill(initial)
    else:
        c = output = np.full((m, n), initial, dtype=output_dtype)
    if spec.trait.preshuffle_b:
        tile_n, tile_k = spec.tile.tile_n, spec.tile.tile_k
        k_tiles = (k + tile_k - 1) // tile_k
        # Fill the physically allocated but logically clipped K tail with
        # nonzero data so a missed clip changes the result.
        padded = np.full((n, k_tiles * tile_k), 17, dtype=b.dtype)
        padded[:, :k] = b
        b_storage = np.ascontiguousarray(
            padded.reshape(n // tile_n, tile_n, k_tiles, tile_k).transpose(2, 0, 1, 3)
        )
    _run_buffers(
        rt,
        artifact,
        a_storage,
        b_storage,
        output,
        (gn, gm, split_k),
        spec.block_size,
        (m, n, k),
    )

    ref_dtype = np.float64 if c.dtype == np.float64 else np.float32
    ref = (a.astype(ref_dtype) @ b.astype(ref_dtype).T).astype(c.dtype)
    np.testing.assert_array_equal(c, ref)
    if edges:
        np.testing.assert_array_equal(output[m * n :], 137)
    _numeric_pass(label, k, c)


def _numeric_pass(label, k, result):
    print(
        json.dumps(
            {"case": label, "k": k, "phase": "numeric_pass", "elements": result.size}
        )
    )


def _upload(rt, stack, arrays):
    from rocke.runtime.host_buffers import as_u8_buffer

    ptrs = []
    for array in arrays:
        ptr = rt.alloc(array.nbytes)
        stack.callback(rt.free, ptr)
        rt.memcpy_h2d(ptr, as_u8_buffer(array), array.nbytes)
        ptrs.append(ptr)
    return ptrs


def _launch_buffers(rt, artifact, arrays, grid, block_size, pack):
    from rocke.runtime.host_buffers import as_u8_buffer

    output = arrays[2]
    with ExitStack() as stack:
        mod = rt.load_module(artifact.hsaco)
        stack.callback(mod.unload)
        ptrs = _upload(rt, stack, arrays)
        rt.launch_blocking(
            mod.get_function(artifact.kernel_name), grid, (block_size, 1, 1), pack(ptrs)
        )
        rt.memcpy_d2h(as_u8_buffer(output), ptrs[2], output.nbytes)


def _run_buffers(rt, artifact, a, b, output, grid, block_size, scalars):
    def pack(ptrs):
        return struct.pack("<QQQ" + "i" * len(scalars), *ptrs, *scalars)

    _launch_buffers(rt, artifact, (a, b, output), grid, block_size, pack)


def _compile(kernel, arch):
    from rocke.helpers.compile import compile_kernel

    artifact = compile_kernel(kernel, arch=arch, backend="python")
    print(json.dumps({"case": kernel.name, "phase": "compiled", "isa": artifact.isa}))
    return artifact


def _fp16_arrays(rng, m, n, k, batch=()):
    import numpy as np

    a = rng.integers(-3, 4, (*batch, m, k), dtype=np.int16).astype(np.float16)
    b = rng.integers(-3, 4, (*batch, n, k), dtype=np.int16).astype(np.float16)
    c = np.full((*batch, m, n), np.nan, dtype=np.float16)
    return a, b, c


def _check_fp16(label, a, b, c):
    import numpy as np

    ref = (a.astype(np.float32) @ b.astype(np.float32).swapaxes(-1, -2)).astype(c.dtype)
    np.testing.assert_array_equal(c, ref)
    _numeric_pass(label, a.shape[-1], c)


def _grouped_gfx950(rt, rng):
    from rocke.instances.common.grouped_gemm import (
        GroupedGemmLauncher,
        GroupedGemmProblem,
        GroupedGemmSpec,
        build_grouped_gemm,
    )
    from rocke.runtime.host_buffers import as_u8_buffer

    spec = GroupedGemmSpec(
        name="grouped_swizzled",
        tile=TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle", lds_swizzle=True),
    )
    artifact = _compile(build_grouped_gemm(spec, arch="gfx950"), "gfx950")
    launch = GroupedGemmLauncher(hsaco=artifact.hsaco, spec=spec)
    with ExitStack() as stack:
        problems, outputs = [], []
        for m, n, k in ((128, 256, 64), (256, 128, 96)):
            a, b, c = _fp16_arrays(rng, m, n, k)
            pa, pb, pc = _upload(rt, stack, (a, b, c))
            problems.append(GroupedGemmProblem(m, n, k, pa, pb, pc))
            outputs.append((a, b, c, pc))
        if launch(problems).launches != len(problems):
            raise RuntimeError("grouped GEMM did not launch all problems")
        rt.sync()
        for a, b, c, pc in outputs:
            rt.memcpy_d2h(as_u8_buffer(c), pc, c.nbytes)
            _check_fp16(spec.name, a, b, c)
    return len(outputs)


def _batched_gfx950(rt, rng):
    from rocke.instances.common.batched_gemm import (
        BatchedGemmSpec,
        batched_gemm_grid,
        build_batched_gemm,
    )

    spec = BatchedGemmSpec(
        name="batched_swizzled",
        tile=TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
        trait=TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
    )
    artifact = _compile(build_batched_gemm(spec, arch="gfx950"), "gfx950")
    batch, m, n, k = 2, 128, 256, 96
    a, b, c = _fp16_arrays(rng, m, n, k, (batch,))
    _run_buffers(
        rt,
        artifact,
        a,
        b,
        c,
        batched_gemm_grid(batch, m, n, spec),
        spec.block_size,
        (m, n, k, m * k, n * k, m * n),
    )
    _check_fp16(spec.name, a, b, c)
    return 1


def _multi_abd_gfx950(rt, rng):
    import numpy as np

    from rocke.instances.common.gemm_multi_abd import (
        GemmMultiAbdSpec,
        build_gemm_multi_abd,
        gemm_multi_abd_grid,
        gemm_multi_abd_signature,
    )
    from rocke.runtime.packing import pack_args

    label = "multi_abd_swizzled"
    spec = GemmMultiAbdSpec(
        base=UniversalGemmSpec(
            name=label,
            tile=TileSpec(64, 128, 32, 1, 2),
            trait=TraitSpec(pipeline="mem", epilogue="cshuffle", lds_swizzle=True),
        ),
        d_operands=(("D0", "add"),),
        d_load_kind="stock",
    )
    artifact = _compile(build_gemm_multi_abd(spec, arch="gfx950"), "gfx950")
    signature = gemm_multi_abd_signature(spec)
    m, n, k = 128, 256, 96
    a, b, c = _fp16_arrays(rng, m, n, k)
    # Use a nonzero, varying addend so omitting the fusion fails independently
    # of the GEMM. Small integer inputs keep the reference exactly representable.
    d0 = rng.integers(1, 8, (m, n), dtype=np.int16).astype(np.float16)

    def pack(ptrs):
        pa, pb, pc, pd = ptrs
        return pack_args(signature, dict(A=pa, B=pb, C=pc, M=m, N=n, K=k, D0=pd))

    _launch_buffers(
        rt,
        artifact,
        (a, b, c, d0),
        gemm_multi_abd_grid(spec, m, n),
        spec.base.block_size,
        pack,
    )
    ref = a.astype(np.float32) @ b.astype(np.float32).T + d0.astype(np.float32)
    np.testing.assert_array_equal(c, ref.astype(c.dtype), err_msg=label)
    _numeric_pass(label, k, c)
    return 1


def _gfx950(rt, rng):
    count = 0
    for label, trait, _, sizes in tile_copy_cases():
        spec = tile_copy_spec(trait)
        artifact = _compile(build_universal_gemm(spec, arch="gfx950"), "gfx950")
        for k in sizes:
            _launch_and_compare(rt, spec, artifact, rng, label, k)
            count += 1
    return (
        count
        + _grouped_gfx950(rt, rng)
        + _batched_gfx950(rt, rng)
        + _multi_abd_gfx950(rt, rng)
    )


def _gfx942(rt, rng):
    count = 0
    for swizzle in (False, True):
        spec = UniversalGemmSpec(
            name="fp64_dtl_sub_element",
            tile=TileSpec(64, 64, 16, 2, 2, 1, 16, 16, 4),
            trait=TraitSpec(
                pipeline="compv4",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=True,
                lds_swizzle=swizzle,
            ),
            data=DataSpec("fp64", "fp64", "fp64", "fp64"),
        )
        artifact = _compile(build_universal_gemm(spec, arch="gfx942"), "gfx942")
        label = "fp64_dtl_swizzled" if swizzle else "fp64_dtl"
        for k in (16, 48):
            _launch_and_compare(rt, spec, artifact, rng, label, k)
            count += 1
    return count


def _gfx1250(rt, rng):
    import numpy as np

    # Include both a K-only tail and a tail on each operand's row axis.
    aligned = (256, 384)
    cases = (
        ("single", 1, 0, False, False, aligned, (32, 80)),
        ("ping_pong", 2, 8, False, False, aligned, (32, 64, 96, 80)),
        ("ring", 3, 8, False, False, aligned, (32, 80, 128)),
        ("split_ring", 3, 8, True, False, aligned, (80, 128)),
        ("early_refill", 2, 8, True, True, aligned, (64, 80, 96)),
        ("ragged_mn", 1, 0, False, False, (259, 387), (80,)),
    )
    count = 0
    tile = TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32)
    for name, depth, pad, split_barrier, early, (m, n), sizes in cases:
        pad_m, pad_n = m % tile.tile_m != 0, n % tile.tile_n != 0
        gm = (m + tile.tile_m - 1) // tile.tile_m
        gn = (n + tile.tile_n - 1) // tile.tile_n
        spec = UniversalGemmSpec(
            name=f"tdm_tile_loader_{name}",
            tile=tile,
            wave_size=32,
            trait=TraitSpec(
                pipeline="mem",
                epilogue="default",
                pad_m=pad_m,
                pad_n=pad_n,
                tdm=True,
                tdm_depth=depth,
                lds_k_pad=pad,
                split_barrier=split_barrier,
                tdm_alternate_issuers=early,
                tdm_early_refill=early,
            ),
        )
        artifact = _compile(build_universal_gemm(spec, arch="gfx1250"), "gfx1250")
        if "call void @llvm.amdgcn.tensor.load.to.lds" not in artifact.llvm_text:
            raise RuntimeError(f"{name} did not emit TDM")
        for k in sizes:
            a, b, c = _fp16_arrays(rng, m, n, k)
            output = c
            if pad_m or pad_n:
                # Keep A and B exact-size. Guard all output addresses a full
                # tile could touch, without changing the logical C row stride.
                output = np.full(
                    gm * tile.tile_m * gn * tile.tile_n, 137, dtype=c.dtype
                )
                c = output[: m * n].reshape(m, n)
                c.fill(np.nan)
            _run_buffers(
                rt, artifact, a, b, output, (gn, gm, 1), spec.block_size, (m, n, k)
            )
            if pad_m or pad_n:
                np.testing.assert_array_equal(
                    output[m * n :], 137, err_msg="TDM output canary was overwritten"
                )
            _check_fp16(name, a, b, c)
            count += 1
    return count


_GPU_CASES = {"gfx950": _gfx950, "gfx942": _gfx942, "gfx1250": _gfx1250}
SKIP = 77


def main(argv=None) -> int:
    import numpy as np

    from rocke.runtime.comgr import loaded_compiler_info
    from rocke.runtime.hip_module import Runtime
    from rocke.tests.gpu import probe_device_arch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=tuple(_GPU_CASES))
    args = parser.parse_args(argv)
    required = f"ROCKE_REQUIRE_{args.arch.upper()}"
    arch, absence = probe_device_arch()
    if arch != args.arch:
        print(
            f"UniversalGemm gate needs {args.arch}; found {arch or absence}",
            file=sys.stderr,
        )
        return 1 if os.environ.get(required) == "1" else SKIP
    compiler = loaded_compiler_info()
    if compiler is None:
        print(f"UniversalGemm {arch} gate needs COMGR", file=sys.stderr)
        return 1
    print(f"arch={arch} numpy={np.__version__} compiler={compiler}")
    count = _GPU_CASES[arch](Runtime(), np.random.default_rng(2301))
    if count <= 0:
        print(f"UniversalGemm {arch} gate ran no comparisons", file=sys.stderr)
        return 1
    print(json.dumps({"arch": arch, "comparisons": count, "phase": "summary"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
