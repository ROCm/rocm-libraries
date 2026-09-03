# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""M1 end-to-end demo: a GEMM authored ENTIRELY through the ``rocke.helpers.tiling`` surface.

A runnable proof of the API on real hardware. It follows rocke's normal flow -- a ``Spec`` of
knobs/levers feeds a ``build_*`` builder: :class:`TilingGemmSpec` fixes the tile size (and,
optionally, the MMA atom); ``build_tiling_gemm`` binds a target, ``TileMma`` resolves the
intrinsic + layouts (IR-free, oracle-proven), and the body threads ptr-free ``TensorDesc`` +
dtype-free ``TileDesc`` through the generic ``*_fragment`` verbs and the ``TileMma`` driver.
Every address comes from our encodings and bottoms out only at raw ``IRBuilder`` ops.
**No rocke ``mfma_gemm_inner`` helpers are used.**

The only rocke-substrate dependency is the ``IRBuilder`` (the lowering vessel) + element types +
compile/launch; those are imported lazily inside the run functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import (
    TileMma,
    Tiling,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
    transform_fragment,
)


@dataclass(frozen=True)
class TilingGemmSpec:
    """Knobs/levers for one dense-RCR tiling GEMM instance -- the top of the spec->builder
    flow. TARGET-AGNOSTIC: the arch binds at build time (``build_tiling_gemm(spec, ..., arch=)``).

    Levers:
      * ``tile`` -- the per-CTA (M, N, K) wave-tile size. REQUIRED; the primary lever.
      * ``atom`` -- OPTIONAL MMA-atom knob (``Tiling.atom_shape``): a shape ``(M, N, K)`` tuple
        (target-agnostic) OR an explicit intrinsic name ``str`` (escape hatch). ``None`` means
        "single MMA" (atom == tile).
      * ``order`` -- the M/N/K subtile loop-nest order (``Tiling.order``; a permutation of
        ``"MNK"``).
      * ``a_dtype`` / ``b_dtype`` / ``c_dtype`` -- element types (SOT tokens).
      * ``name`` -- kernel-name stem.
    """

    tile: tuple[int, int, int]
    atom: Optional[tuple[int, int, int] | str] = None
    order: str = "MNK"
    a_dtype: str = "f16"
    b_dtype: str = "f16"
    c_dtype: str = "f32"
    name: str = "tiling_gemm_demo"

    def __post_init__(self) -> None:
        if len(self.tile) != 3 or any(
            not isinstance(d, int) or d <= 0 for d in self.tile
        ):
            raise ValueError(f"tile must be 3 positive ints -- tile={self.tile!r}")
        if self.atom is not None and not isinstance(self.atom, str):
            if len(self.atom) != 3 or any(
                not isinstance(d, int) or d <= 0 for d in self.atom
            ):
                raise ValueError(f"atom must be 3 positive ints -- atom={self.atom!r}")


def is_valid_spec(spec: TilingGemmSpec, arch: str = "gfx90a") -> tuple[bool, str]:
    """Fail-fast spec check against a bound target: does the atom resolve to an intrinsic on
    ``arch`` and is the wave tile a clean multiple of it? Returns ``(ok, reason)`` (rocke's
    spec-validation contract)."""
    try:
        TileMma(
            spec.tile,
            a=spec.a_dtype, b=spec.b_dtype, c=spec.c_dtype, target=arch,
            tiling=Tiling(atom_shape=spec.atom, order=spec.order),
        )
    except (ValueError, NotImplementedError) as exc:
        return False, str(exc)
    return True, "ok"


def build_tiling_gemm(
    spec: TilingGemmSpec, M_LEN: int, N_LEN: int, K_LEN: int, *, arch: str = "gfx90a",
    lda: int | None = None, ldb: int | None = None, ldc: int | None = None,
    interleave_a: bool = False,
):
    """Build a dense-RCR f16->f32(->f16 store) GEMM KernelDef from a spec.

    RCR: A row-major (M,K), B row-major (K,N), C row-major (M,N); one wave-tile per CTA
    (wave64). ``M_LEN/N_LEN/K_LEN`` are the compile-time tensor extents.
    Returns ``(kernel_def, mma)``.
    """
    from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid tiling_gemm spec for {arch!r} -- {why}")

    # The MMA object owns the wave tile: shape = the per-CTA tile, atom_shape = the knob.
    # It resolves the intrinsic from the atom and iterates the K subtiles internally.
    mma = TileMma(
        spec.tile,
        a=spec.a_dtype, b=spec.b_dtype, c=spec.c_dtype, target=arch,
        tiling=Tiling(atom_shape=spec.atom, order=spec.order),
    )
    TILE_M, TILE_N, TILE_K = spec.tile

    b = IRBuilder(f"{spec.name}_{M_LEN}x{N_LEN}x{K_LEN}_{spec.a_dtype}_{arch}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size

    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    lane = b.thread_id_x()
    m_tile_base = b.mul(b.block_id_y(), b.const_i32(TILE_M))
    n_tile_base = b.mul(b.block_id_x(), b.const_i32(TILE_N))

    # Leading dims: the tensor's physical row stride, SEPARATE from the compute extent
    # (M/N/K). When ld > the dim, rows past the compute region are still valid, in-bounds data
    # -- so the clip must exclude them by COORDINATE, not by hitting the tensor edge.
    lda = lda if lda is not None else K_LEN
    ldb = ldb if ldb is not None else N_LEN
    ldc = ldc if ldc is not None else N_LEN

    # Memory descriptors (ptr-free): lengths = the VALID (compute) extent = the default clip;
    # strides = the physical layout (leading dim). Operand difference is HERE, in the strides +
    # the layout -- never in the verb. RCR: A(M,K) row-major, B stored so coords (n,k) walk
    # strides (1, ld), C(M,N) row-major. The ptr binds at load/store, not on the descriptor.
    a_td = make_tensor_desc((M_LEN, K_LEN), (lda, 1), F16)
    b_td = make_tensor_desc((N_LEN, K_LEN), (1, ldb), F16)
    c_td = make_tensor_desc((M_LEN, N_LEN), (ldc, 1), F16)

    # The MMA object hands back the per-operand TileDescs directly (wave shape + resolved
    # layout, K subtiles folded in). One load fills the whole tile; mma() walks the atoms.
    # Clipping (Part C): each window is (origin, upper bound) = the COMPUTE extent M/N/K (not
    # the physical ld). load/store clip any element whose position reaches that bound (zero-pad
    # on load, drop on store); tile-aligned bounds are skipped at build time (byte-identical).
    accumulator = make_fragment(mma.c_desc, F32)
    fill_fragment(b, accumulator, 0)
    for tile_k_base in range(0, K_LEN, TILE_K):
        k_base = b.const_i32(tile_k_base)
        a_win = make_window(a_td, (m_tile_base, k_base))  # clip defaults to the desc lengths
        b_win = make_window(b_td, (n_tile_base, k_base))

        a_fragment = load_fragment(b, a_ptr, a_win, mma.a_desc(), lane)
        b_fragment = load_fragment(b, b_ptr, b_win, mma.b_desc(), lane)
        accumulator = mma(b, a_fragment, b_fragment, accumulator)

    c_win = make_window(c_td, (m_tile_base, n_tile_base))
    store_fragment(b, c_ptr, c_win, accumulator, lane)
    b.ret()
    return b.kernel, mma


def _run_gemm_numpy(launcher, config, arrays: dict, scalars: dict) -> None:
    """Torch-free device I/O: upload each numpy host array to a `DeviceMem` (RAII over
    ``hipMalloc``), launch with those device pointers, then copy every buffer back into its host
    array in place. The rocke runtime's numpy path -- no torch tensors; `pack_args` reads each
    ``DeviceMem.ptr()`` and each scalar `int` straight from the values dict."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, synchronize_and_release

    rt = Runtime()
    device = {name: DeviceMem(host.nbytes) for name, host in arrays.items()}
    for name, host in arrays.items():
        rt.memcpy_h2d(device[name].ptr(), as_u8_buffer(host), host.nbytes)
    launcher({**device, **scalars}, config=config)
    synchronize_and_release()
    for name, host in arrays.items():
        rt.memcpy_d2h(as_u8_buffer(host), device[name].ptr(), host.nbytes)


def run_and_verify(
    M_LEN: int = 256, N_LEN: int = 256, K_LEN: int = 256,
    *, spec: Optional[TilingGemmSpec] = None, arch: str = "gfx90a", interleave_a: bool = False,
) -> dict:
    """Compile, launch on the GPU, and verify C = A @ B against a NUMPY golden reference.

    torch-free: numpy host arrays + `DeviceMem` device buffers. ``spec`` defaults to a 16x16x16
    single-atom tile (the M1 config). ``interleave_a=True`` loads A in the interleaved (AOS) layout
    and `transform_fragment`s it to the MMA form -- proving the register-reorder path bit-exact."""
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if spec is None:
        spec = TilingGemmSpec(tile=(16, 16, 16))
    kernel, mma = build_tiling_gemm(spec, M_LEN, N_LEN, K_LEN, arch=arch, interleave_a=interleave_a)
    TILE_M, TILE_N, _TILE_K = spec.tile
    artifact = compile_kernel(kernel, arch=arch)
    signature = (
        SignatureBuilder()
        .ptr("A", "f16").ptr("B", "f16").ptr("C", "f16")
        .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
        .build()
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )
    # Integer inputs in [-3, 3]: exact in f16, f32 accumulation is exact regardless of
    # order, and the result fits f16 exactly -> the correct kernel is BIT-EXACT (tol 0),
    # isolating correctness from f16 float-rounding noise (rocke's CDNA verify convention).
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(M_LEN, K_LEN)).astype(np.float16)
    b = rng.integers(-3, 4, size=(K_LEN, N_LEN)).astype(np.float16)
    c = np.zeros((M_LEN, N_LEN), dtype=np.float16)
    # ceil_div: edge CTAs launch for a ragged matrix and clip internally (Part C).
    grid = (-(-N_LEN // TILE_N), -(-M_LEN // TILE_M), 1)
    _run_gemm_numpy(
        launcher,
        LaunchConfig(grid=grid, block=(mma.wave_size, 1, 1)),
        {"A": a, "B": b, "C": c},
        {"M": M_LEN, "N": N_LEN, "K": K_LEN},
    )
    reference = a.astype(np.float32) @ b.astype(np.float32)
    max_abs_diff = float(np.abs(c.astype(np.float32) - reference).max())
    return {
        "op_id": mma.op_id,
        "target": arch,
        "tile": spec.tile,
        "shape": (M_LEN, N_LEN, K_LEN),
        "max_abs_diff": max_abs_diff,
        "bit_exact": max_abs_diff == 0.0,
        "reference": "C = A @ B (RCR), integer inputs",
    }


def run_and_verify_within_valid_space(
    compute: int = 250, alloc: int = 256, *, spec: Optional[TilingGemmSpec] = None,
    arch: str = "gfx90a",
) -> dict:
    """Isolate the clip INSIDE a valid space (not the OOB tensor edge).

    Compute a `compute`^3 GEMM into tensors ALLOCATED `alloc`-wide (leading dim = alloc >
    compute). Rows/cols `compute..alloc` therefore hold VALID, in-bounds data -- the clip must
    exclude them by COORDINATE (a broken mask would leak them into the result), and the store
    must leave the C tail there UNTOUCHED (a NaN sentinel proves no OOB write).
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if spec is None:
        spec = TilingGemmSpec(tile=(16, 16, 16))
    kernel, mma = build_tiling_gemm(
        spec, compute, compute, compute, arch=arch, lda=alloc, ldb=alloc, ldc=alloc
    )
    TILE_M, TILE_N, _TILE_K = spec.tile
    artifact = compile_kernel(kernel, arch=arch)
    signature = (
        SignatureBuilder()
        .ptr("A", "f16").ptr("B", "f16").ptr("C", "f16")
        .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
        .build()
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )
    # Full alloc x alloc arrays of VALID data; C prefilled with NaN so any untouched cell
    # stays NaN (uploaded as-is -- memset can't write NaN). Compute region is only [:compute, :compute].
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(alloc, alloc)).astype(np.float16)
    b = rng.integers(-3, 4, size=(alloc, alloc)).astype(np.float16)
    c = np.full((alloc, alloc), np.nan, dtype=np.float16)
    grid = (-(-compute // TILE_N), -(-compute // TILE_M), 1)
    _run_gemm_numpy(
        launcher,
        LaunchConfig(grid=grid, block=(mma.wave_size, 1, 1)),
        {"A": a, "B": b, "C": c},
        {"M": compute, "N": compute, "K": compute},
    )
    reference = a[:compute, :compute].astype(np.float32) @ b[:compute, :compute].astype(np.float32)
    computed_diff = float(
        np.abs(c[:compute, :compute].astype(np.float32) - reference).max()
    )
    tail_untouched = bool(
        np.isnan(c[compute:, :]).all() and np.isnan(c[:, compute:]).all()
    )
    return {
        "compute": compute,
        "alloc": alloc,
        "tile": spec.tile,
        "computed_max_abs_diff": computed_diff,
        "computed_bit_exact": computed_diff == 0.0,
        "tail_untouched": tail_untouched,
    }


if __name__ == "__main__":
    print(run_and_verify())
