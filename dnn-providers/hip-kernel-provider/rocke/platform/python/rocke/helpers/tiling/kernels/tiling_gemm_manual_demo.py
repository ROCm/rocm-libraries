# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lowest-level MANUAL GEMM -- a TEACHING demo showing the PURE CUSTOMIZATION path of the
tiling engine.

This is the FIRST (lowest) layer of the layered authoring API. EVERYTHING is written by hand with
the primitive surface -- nothing is derived for you. Its job is to teach, precisely, how a
register/thread distribution is authored from human geometric quantities, with no MMA calculator
and no conveniences standing between the author and the hardware layout.

Where this sits in the layered API:
  * the DRIVER demo (tiling_gemm_demo) asks `TileMma` to DERIVE the A/B/C operand distributions
    from the atom -- convenient, but the layout is hidden inside the calculator; whereas
  * THIS demo authors the A/B/C distributions MANUALLY via the quantity-major `make_tile_desc`
    surface, so you can see and control exactly which lane + register holds which element. This is
    the escape hatch for a custom / non-standard layout the calculator would not give you.

What is hand-authored here:
  * pointers + `make_tensor_desc` for the memory (lengths + strides, rightmost = stride-1);
  * `TileMma` is configured ONLY to hand back the backend `mma` op + wave_size -- NO layout comes
    from it (`a_desc`/`b_desc`/`c_desc` are authored by us, not by the calculator);
  * the A / B / Acc / C register distributions are authored MANUALLY with `make_tile_desc`;
  * the MMA loop is hand-iterated -- fragments plug straight into the raw `b.mma(op, ...)`.

Layout is RCC (A row-major, B col-major, C col-major). Single 16x16x16 atom, one wave (wave64).

KEY TEACHING POINT -- two INDEPENDENT mappings that this demo keeps deliberately separate:
  * the DESCRIPTOR maps a logical element (e.g. (m, k)) -> a MEMORY ADDRESS, via strides. Pure
    memory layout; it knows nothing about lanes or registers.
  * the DISTRIBUTION maps (lane, register) -> a logical element. Pure hardware placement; it knows
    nothing about memory addresses.
  They never constrain each other -- `load_fragment` simply COMPOSES them (ask the distribution
  "which element?", then ask the descriptor "which address?"). That separation is the whole point
  of the primitive surface, and it is why the same distribution can front any memory layout.

And their AXIS ORDER differs by design: the TENSOR descriptor is authored in PHYSICAL order
(fastest stride right-most, like numpy / CK / CuTe); the TILE descriptor is in LOGICAL matrix
order. A col-major operand bridges the two with a `.permute([...])` VIEW on the tensor descriptor
-- an explicit axis-index reorder (a pure re-labeling of axes -- same bytes); the 2-D swap is
`.permute([1, 0])`.
"""

from __future__ import annotations

import numpy as np

from .. import (
    TileMma,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)


def build_manual_gemm(
    M_LEN: int, N_LEN: int, K_LEN: int, *, arch: str = "gfx90a",
    lda: int | None = None, ldb: int | None = None, ldc: int | None = None,
):
    """Build the RCC f16->f32(->f16) GEMM from primitives only. One 16x16x16 atom per CTA."""
    from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

    TILE_M = TILE_N = TILE_K = 16  # single universal atom

    # Configure the MMA ONLY to resolve the backend op + wave_size. `atom_override` names the exact
    # intrinsic; TileMma then DERIVES the M/N/K and dtypes from that atom's traits, so we don't
    # restate them. Crucially we never touch `mma.a_desc`/`b_desc`/`c_desc`: NO layout comes from
    # the calculator in this demo -- we author every distribution by hand below.
    mma = TileMma(target=arch, atom_override="mfma_f32_16x16x16f16")
    op = mma.emit_op()

    b = IRBuilder(f"tiling_gemm_manual_{M_LEN}x{N_LEN}x{K_LEN}_{arch}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size

    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    lane_id = b.thread_id_x()
    m_tile_base = b.mul(b.block_id_y(), b.const_i32(TILE_M))
    n_tile_base = b.mul(b.block_id_x(), b.const_i32(TILE_N))

    lda = lda if lda is not None else K_LEN   # A physical (M, K): M-stride (row length K)
    ldb = ldb if ldb is not None else K_LEN   # B physical (N, K): N-stride (column length K)
    ldc = ldc if ldc is not None else M_LEN   # C physical (N, M): N-stride (column length M)

    # ---- MEMORY (tensor descriptors) ----------------------------------------------------------
    # TRADITIONAL memory layout: lengths + strides with the FASTEST (stride-1) axis RIGHT-MOST,
    # the same convention as numpy / CK / CuTe. The axis ORDER here is PHYSICAL -- it follows
    # memory, NOT the logical matrix. This demo is RCC, so the stride-1 rightmost axis is K for A
    # (row-major) and K / M for B / C (col-major -- hence their physical shapes are (N, K) and
    # (N, M)). Each is handed to its tile in LOGICAL order via `.permute([1, 0])` at the window (a
    # pure view, same bytes): a col-major operand's physical (N, K) then reads as logical (K, N).
    a_td = make_tensor_desc((M_LEN, K_LEN), (lda, 1), F16)   # physical (M, K): K contiguous (stride 1)
    b_td = make_tensor_desc((N_LEN, K_LEN), (ldb, 1), F16)   # physical (N, K): K contiguous (stride 1)
    c_td = make_tensor_desc((N_LEN, M_LEN), (ldc, 1), F16)   # physical (N, M): M contiguous (stride 1)

    # ---- DISTRIBUTION (the heart of this demo: PURE CUSTOMIZATION) -----------------------------
    # `make_tile_desc` is the quantity-major surface: one axes-ordered list per geometric quantity.
    # Its COLUMNS are the LOGICAL matrix axes -- A is (M, K), B is (K, N), C is (M, N) -- NOT the
    # physical memory order; the physical tensor descriptor above is lined up to this logical order
    # by the window's `.permute([1, 0])`. It takes the tile `shape` too, so it hands back a ready
    # `TileDesc` (shape + layout) in ONE call.
    #   * shape            -- the tile size per axis.
    #   * thread_tile      -- contiguous elements each lane holds per axis (register run; stride-1 vector).
    #   * thread_dist      -- how the wave's lanes spread over each axis (product == wave_size).
    #   * thread_order     -- lane-carrying axes, FASTEST-moving axis RIGHT-MOST (default = axis order).
    #   * thread_broadcast -- duplicate the tile across lanes (1 = none).
    #   * block_repeat     -- the whole lane tile stamped as strided registers (1 = none).
    #   * wave_dist        -- how the block's waves spread over each axis (1 = single wave here).
    #   * wave_order       -- wave axes, fastest right-most (unused here -- single wave).
    #   * wave_broadcast   -- duplicate the tile across waves (1 = none).
    #
    # The 16x16x16 MFMA atom fixes the lane<->element wiring: it wants the CONTRACTION axis as the
    # MAJOR (slowest) lane -- K for A/B, M for C. In logical coords that axis is column 0 for B (K, N)
    # and C (M, N), so plain column order already puts it major; for A (M, K) it is column 1, so A's
    # thread_order lists K first. EVERY argument is spelled out below for teaching -- the trivial ones
    # (block_repeat, thread_broadcast, wave_*) sit at their no-op values.
    a_desc = make_tile_desc(
        shape=[TILE_M, TILE_K],   # (M, K)
        thread_tile=[1, 4],       # K: 4 contiguous per lane (k_inner) -- the vector
        thread_dist=[16, 4],      # M -> 16 lanes, K -> 4 lanes (k_outer)
        thread_order=[1, 0],      # M fastest, K major -- the atom's wiring (K is column 1, so A overrides)
        thread_broadcast=1,       # no lane duplication
        block_repeat=[1, 1],      # no stamped repeats
        wave_dist=[1, 1],         # single wave
        wave_order=None,          # single wave -- nothing to order
        wave_broadcast=1,         # no wave duplication
        wave_size=64,
    )  # A operand
    b_desc = make_tile_desc(
        shape=[TILE_K, TILE_N],   # (K, N)
        thread_tile=[4, 1],       # K: 4 contiguous per lane (k_inner) -- the vector
        thread_dist=[4, 16],      # K -> 4 lanes (k_outer), N -> 16 lanes
        thread_order=[0, 1],      # N fastest, K major -- column order (K already column 0)
        thread_broadcast=1,       # no lane duplication
        block_repeat=[1, 1],      # no stamped repeats
        wave_dist=[1, 1],         # single wave
        wave_order=None,          # single wave -- nothing to order
        wave_broadcast=1,         # no wave duplication
        wave_size=64,
    )  # B operand
    c_desc = make_tile_desc(
        shape=[TILE_M, TILE_N],   # (M, N)
        thread_tile=[4, 1],       # M: 4 contiguous per lane (m_inner) -- the vector
        thread_dist=[4, 16],      # M -> 4 lanes (m_outer), N -> 16 lanes
        thread_order=[0, 1],      # N fastest, M major -- column order (M already column 0)
        thread_broadcast=1,       # no lane duplication
        block_repeat=[1, 1],      # no stamped repeats
        wave_dist=[1, 1],         # single wave
        wave_order=None,          # single wave -- nothing to order
        wave_broadcast=1,         # no wave duplication
        wave_size=64,
    )  # C accumulator

    # ---- MANUAL MMA loop ----------------------------------------------------------------------
    # Accumulate over the K tiles, one 16x16x16 atom each, plugging the fragments straight into the
    # raw `b.mma(op, ...)` -- no TileMma driver, no calculator-provided descs.
    accumulator = make_fragment(c_desc, F32)
    fill_fragment(b, accumulator, 0)
    for tile_k_base in range(0, K_LEN, TILE_K):
        k_base = b.const_i32(tile_k_base)
        # Positioned windows at this K tile; origin order matches each descriptor's axis order.
        a_win = make_window(a_td, (m_tile_base, k_base))                  # A physical (M,K) == logical (M,K)
        b_win = make_window(b_td.permute([1, 0]), (k_base, n_tile_base))  # B physical (N,K) -> logical (K,N)
        a_fragment = load_fragment(b, a_ptr, a_win, a_desc, lane_id)
        b_fragment = load_fragment(b, b_ptr, b_win, b_desc, lane_id)
        accumulator.value = b.mma(
            op, a_fragment.value, b_fragment.value, accumulator.value
        )

    # Store through the hand-authored C distribution; C physical (N,M) -> logical (M,N) via permute.
    c_win = make_window(c_td.permute([1, 0]), (m_tile_base, n_tile_base))  # C physical (N,M) -> logical (M,N)
    store_fragment(b, c_ptr, c_win, accumulator, lane_id)
    b.ret()
    return b.kernel, mma


def run_and_verify_manual(
    M_LEN: int = 256, N_LEN: int = 256, K_LEN: int = 256, *, arch: str = "gfx90a"
) -> dict:
    """Compile, launch, and verify the MANUAL RCC GEMM is bit-exact vs a NUMPY golden reference
    (integer inputs). torch-free: numpy host arrays + `DeviceMem` device buffers."""
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    kernel, mma = build_manual_gemm(M_LEN, N_LEN, K_LEN, arch=arch)
    TILE_M = TILE_N = 16
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
    # RCC buffers, each laid out to match its descriptor exactly:
    #   A (M, K) row-major   -> a_buf[m, k] = A[m, k]
    #   B (N, K) col-major   -> b_buf[n, k] = B[k, n]   (i.e. the logical B is b_buf transposed)
    #   C (N, M) col-major   -> c_buf[n, m] = C[m, n]
    rng = np.random.default_rng(0)
    a_buf = rng.integers(-3, 4, size=(M_LEN, K_LEN)).astype(np.float16)
    b_buf = rng.integers(-3, 4, size=(N_LEN, K_LEN)).astype(np.float16)
    c_buf = np.zeros((N_LEN, M_LEN), dtype=np.float16)
    grid = (-(-N_LEN // TILE_N), -(-M_LEN // TILE_M), 1)
    # Torch-free device I/O: upload the host arrays to DeviceMem buffers, launch with those
    # device pointers, copy C back. pack_args reads each DeviceMem.ptr() from the values dict.
    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_buf, b_buf, c_buf))
    rt.memcpy_h2d(a_dev.ptr(), as_u8_buffer(a_buf), a_buf.nbytes)
    rt.memcpy_h2d(b_dev.ptr(), as_u8_buffer(b_buf), b_buf.nbytes)
    rt.memcpy_h2d(c_dev.ptr(), as_u8_buffer(c_buf), c_buf.nbytes)
    launcher(
        {"A": a_dev, "B": b_dev, "C": c_dev,
         "M": M_LEN, "N": N_LEN, "K": K_LEN},
        config=LaunchConfig(grid=grid, block=(mma.wave_size, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(c_buf), c_dev.ptr(), c_buf.nbytes)
    # logical: C[m, n] = sum_k A[m, k] * B[k, n] = sum_k a_buf[m, k] * b_buf[n, k]
    reference = a_buf.astype(np.float32) @ b_buf.astype(np.float32).T   # (M, K) @ (K, N) = (M, N)
    result = c_buf.astype(np.float32).T                                # (N, M) buffer -> logical (M, N) C
    max_abs_diff = float(np.abs(result - reference).max())
    return {
        "shape": (M_LEN, N_LEN, K_LEN),
        "op_id": mma.op_id,
        "max_abs_diff": max_abs_diff,
        "bit_exact": max_abs_diff == 0.0,
    }


if __name__ == "__main__":
    print(run_and_verify_manual())
