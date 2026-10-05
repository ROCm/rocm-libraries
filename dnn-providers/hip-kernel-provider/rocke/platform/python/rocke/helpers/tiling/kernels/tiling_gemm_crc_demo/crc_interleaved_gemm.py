# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CRC GEMM -- the free-dim-contiguous (column-major-in, column-major-out) interleaved kernel.

ISOLATED PROTOTYPE (do NOT fold into the other kernels). C = A.B, f16 input -> f32 accumulate ->
f32 OUTPUT, authored ENTIRELY through the ``rocke.helpers.tiling`` surface.

THE FIXED CRC LAYOUT (this is the whole point of the file):
  * A is (M, K) with strides (1, M)  -> M is stride-1  -> A is FREE-DIM (M) CONTIGUOUS.
  * B is (N, K) with strides (1, N)  -> N is stride-1  -> B is FREE-DIM (N) CONTIGUOUS.
  * C is (M, N) with strides (1, M)  -> M is stride-1  -> C is M-CONTIGUOUS (col-major output).

THE CRC GIFT: because the global load is already free-dim-innermost (M for A, N for B), the same
byte order the coalesced global load wants IS the free-dim-innermost LDS store order -- so the
cooperative store into LDS is an IDENTITY round-trip (no global->LDS register reorder). The K-loop
wave read then reaches its K-contiguous MMA operand via a pure ``_transpose_desc`` relabel of the
M-innermost LDS memref (no shuffle, no cross-lane). This is the mirror image of the row-major
interleaved kernel (``tiling_gemm_interleaved_demo``), which pays a register ``reorder`` on the store
because its global load is K-contiguous.

PIPELINE (double-buffered prefetch, wave64, gfx90a, atom 16x16x16):
  global free-dim-contig load -> LDS free-dim-innermost store (IDENTITY) -> K-loop wave read
  (``_transpose_desc`` relabel to K-contig) -> MMA (interleaved atom grid) -> C epilogue.

TWO PATHS selectable by ``ab_swap`` (both compute identical C, both bit-exact):
  * ``ab_swap=False`` (base): MMA built as (warp_m, warp_n, tile_k); C accumulator is the native
    (M, N); stored through the M-contiguous (col-major) C descriptor. The native-C register run is
    M-inner, so the M-contiguous store needs NO fragment reorder (correctness is address-driven).
  * ``ab_swap=True`` (crossed): MMA built as (warp_n, warp_m, tile_k) with B fed into the A-slot and
    A into the B-slot (an M<->N free relabel). The machine emits C' = C^T = (N, M); stored through
    the (N, M)-permuted C descriptor so C'[n, m] lands at addr = m + n*ldc -- the SAME col-major C
    address. In the crossed layout M is lane-major, so the M-contiguous store wave-coalesces.

Only PURE layout helpers are reused from the interleaved demo (``_wave_descs_interleaved``,
``_transpose_desc``, ``b32_swizzle``/``b64_swizzle``); the majors here are CRC-specific.
"""

from __future__ import annotations

import numpy as np

from ... import (
    TileMma,
    Tiling,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
    transform_fragment,
)
from ..tiling_gemm_interleaved_demo import (
    _transpose_desc,
    _wave_descs_interleaved,
    b32_swizzle,  # noqa: F401  (re-exported swizzle policies, usable as lds_swizzle)
    b64_swizzle,  # noqa: F401
)

_ATOM = 16  # the 16x16x16 MFMA atom

# sched_group_barrier instruction-class mask bits (AMDGPU) -- mirror the interleaved demo.
_SGB_MFMA = 0x008
_SGB_VMEM_READ = 0x020
_SGB_VMEM_WRITE = 0x040
_SGB_DS_READ = 0x100
_SGB_DS_WRITE = 0x200


def _macro_coop_descs_crc(tile_free: int, tile_k: int, n_waves: int):
    """CRC cooperative load descriptor: the free dim (M for A, N for B) is the stride-1 per-lane
    vector, MATCHING the free-dim-contiguous global layout -> the coalesced global load is already
    the free-dim-innermost order the LDS store wants, so store == load (an IDENTITY, the CRC gift).

    Returns ONE ``TileDesc`` (the same desc serves the global LOAD and the LDS STORE -- no reorder).
    Each wave loads a ``tile_free / n_waves`` band; within the band a lane owns ``dpt`` contiguous
    free-dim elements (stride-1) and ``kpt`` K elements, the two lane-grid factors spreading the rest.
    """
    coop_free = tile_free // n_waves
    per_lane = (coop_free * tile_k) // 64
    if per_lane < 1:
        raise ValueError(
            f"CRC coop band too small -- coop_free={coop_free} x tile_k={tile_k} < 64 lanes"
        )
    # Keep the free dim the contiguous vector (dpt), spread K across lanes as far as it goes, then
    # spend the rest of per_lane on more free-dim contiguity. kpt=1 keeps the whole free run stride-1.
    dpt = per_lane
    kpt = 1
    while dpt > coop_free:  # can't own more free elements than the band has
        dpt //= 2
        kpt *= 2
    if coop_free % dpt or tile_k % kpt or (coop_free // dpt) * (tile_k // kpt) != 64:
        raise ValueError(
            f"CRC coop split does not tile 64 lanes -- coop_free={coop_free}, tile_k={tile_k}, "
            f"dpt={dpt}, kpt={kpt}"
        )
    m_lanes = coop_free // dpt
    k_lanes = tile_k // kpt
    desc = make_tile_desc(
        shape=[tile_free, tile_k],
        thread_tile=[dpt, kpt],          # free dim = dpt CONTIGUOUS per lane (stride-1); K = kpt
        thread_dist=[m_lanes, k_lanes],
        thread_order=[1, 0],             # K the major lane, free the minor (matches wave read wiring)
        thread_broadcast=1,
        block_repeat=[1, 1],
        wave_dist=[n_waves, 1],          # the block's waves split the free dim
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )
    return desc


def _c_store_desc(m_sub: int, n_sub: int):
    """M-contiguous C store distribution (make_tile_desc): each lane owns a contiguous 4*m_sub x n_sub
    (M x N) block with M inner (stride-1) -> the col-major C store is coalesced/wide in M. The native
    interleaved accumulator -> this store desc is a pure register ``reorder`` (same lane ownership)."""
    return make_tile_desc(
        shape=[m_sub * _ATOM, n_sub * _ATOM],
        thread_tile=[4 * m_sub, n_sub],  # M inner (contiguous), N outer per lane
        thread_dist=[4, 16],
        thread_order=[0, 1],
        thread_broadcast=1,
        block_repeat=[1, 1],
        wave_dist=[1, 1],
        wave_order=None,
        wave_broadcast=1,
        wave_size=64,
    )


def build_crc_gemm(
    M_LEN: int,
    N_LEN: int,
    K_LEN: int,
    *,
    arch: str = "gfx90a",
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 32,
    waves_m: int = 4,
    waves_n: int = 4,
    ab_swap: bool = False,
    single_buffer: bool = False,
    mac_prio: int = 0,
    lds_swizzle=False,
    lda: int | None = None,
    ldb: int | None = None,
    ldc: int | None = None,
):
    """Build the CRC f16->f32 GEMM KernelDef. Returns ``(kernel, mma)``.

    ``ab_swap`` selects the base ((warp_m,warp_n,K), native C) or crossed ((warp_n,warp_m,K),
    C'=C^T stored through the permuted desc) path. Both are bit-exact and write the same col-major C.
    """
    from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType

    n_waves = waves_m * waves_n
    if tile_m % waves_m or tile_n % waves_n:
        raise ValueError(f"macro tile must divide by waves -- {tile_m}x{tile_n} / {waves_m}x{waves_n}")
    warp_m, warp_n = tile_m // waves_m, tile_n // waves_n
    if warp_m % _ATOM or warp_n % _ATOM or tile_k % _ATOM:
        raise ValueError(f"warp tile must be multiples of {_ATOM} -- {warp_m}x{warp_n}x{tile_k}")
    if tile_m % n_waves or tile_n % n_waves:
        raise ValueError(f"macro tile must divide by n_waves={n_waves} for the cooperative load")
    if K_LEN % tile_k:
        raise ValueError(f"K_LEN ({K_LEN}) must be a multiple of tile_k ({tile_k})")
    if (K_LEN // tile_k) < 2 and not single_buffer:
        raise ValueError(
            f"double-buffered pipeline needs >= 2 K-tiles -- K_LEN={K_LEN}, tile_k={tile_k}; "
            "use single_buffer=True for a single K-tile"
        )
    m_sub, n_sub, k_sub = warp_m // _ATOM, warp_n // _ATOM, tile_k // _ATOM

    # The MMA driver is configured with the WARP tile. ab_swap swaps the M/N roles: the machine then
    # multiplies B(fed as A) x A(fed as B) and emits C' = C^T = (N, M).
    if ab_swap:
        mma = TileMma(
            (warp_n, warp_m, tile_k), a="f16", b="f16", c="f32", target=arch,
            tiling=Tiling(atom_shape=(_ATOM, _ATOM, _ATOM)),
        )
    else:
        mma = TileMma(
            (warp_m, warp_n, tile_k), a="f16", b="f16", c="f32", target=arch,
            tiling=Tiling(atom_shape=(_ATOM, _ATOM, _ATOM)),
        )

    b = IRBuilder(
        f"tiling_gemm_crc_{M_LEN}x{N_LEN}x{K_LEN}_{tile_m}x{tile_n}x{tile_k}"
        f"_w{waves_m}x{waves_n}_{'swap' if ab_swap else 'base'}_{arch}"
    )
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size * n_waves

    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(mma.wave_size))
    wm = b.div(wave, b.const_i32(waves_n))
    wn = b.mod(wave, b.const_i32(waves_n))

    m_macro = b.mul(b.block_id_y(), b.const_i32(tile_m))
    n_macro = b.mul(b.block_id_x(), b.const_i32(tile_n))

    # CRC memory descriptors -- the free dim is stride-1 everywhere.
    lda = lda if lda is not None else M_LEN   # A (M,K): M contiguous -> strides (1, lda)
    ldb = ldb if ldb is not None else N_LEN   # B (N,K): N contiguous -> strides (1, ldb)
    ldc = ldc if ldc is not None else M_LEN   # C (M,N): M contiguous -> strides (1, ldc)
    a_td = make_tensor_desc((M_LEN, K_LEN), (1, lda), F16)
    b_td = make_tensor_desc((N_LEN, K_LEN), (1, ldb), F16)
    c_td = make_tensor_desc((M_LEN, N_LEN), (1, ldc), F32)

    # Interleaved wave-tile MMA operands (free-dim-contiguous per lane). ab_swap swaps which physical
    # operand feeds which MMA slot: the A-slot gets the N-major operand, the B-slot the M-major one.
    a_wave, b_wave, c_wave = _wave_descs_interleaved(m_sub, n_sub, k_sub)
    if ab_swap:
        # crossed: the driver's A-slot is N (warp_n), its B-slot is M (warp_m); C' = (warp_n, warp_m).
        slotA_wave, slotB_wave, c_wave = _wave_descs_interleaved(n_sub, m_sub, k_sub)
    else:
        slotA_wave, slotB_wave = a_wave, b_wave

    # Cooperative coop descs (free-dim-contiguous LOAD == LDS STORE, the CRC gift identity).
    a_coop = _macro_coop_descs_crc(tile_m, tile_k, n_waves)
    b_coop = _macro_coop_descs_crc(tile_n, tile_k, n_waves)

    _bufs = 1 if single_buffer else 2
    zero = b.const_i32(0)
    warp_m_c, warp_n_c = b.const_i32(warp_m), b.const_i32(warp_n)
    tile_m_c, tile_n_c = b.const_i32(tile_m), b.const_i32(tile_n)

    # LDS is FREE-DIM-INNERMOST ([tile_k, _bufs*free], free stride-1): the coop store is an identity
    # (store desc == load desc) and the wave read reaches K-contiguous by a _transpose_desc relabel.
    lds_a = b.smem_alloc(F16, [tile_k, _bufs * tile_m], name_hint="lds_a")
    lds_b = b.smem_alloc(F16, [tile_k, _bufs * tile_n], name_hint="lds_b")
    lds_a_td = make_tensor_desc((tile_k, _bufs * tile_m), (_bufs * tile_m, 1), F16)
    lds_b_td = make_tensor_desc((tile_k, _bufs * tile_n), (_bufs * tile_n, 1), F16)

    # The coop store/read index the (K, free) LDS memref -- the free dim is the memref innermost -> wide
    # ds_write/ds_read. The coop desc's own X-dims are (free, K); _transpose_desc gives the (K, free)
    # view the memref uses. Retagging the coop fragment to this transposed layout is register-IDENTITY
    # (same registers, swapped X-dim labels) -- the CRC gift: the store is NOT a reorder.
    a_coop_t, b_coop_t = _transpose_desc(a_coop), _transpose_desc(b_coop)
    a_rd_t, b_rd_t = _transpose_desc(slotA_wave), _transpose_desc(slotB_wave)

    def _store_buf(frags, row_a, row_b):
        # IDENTITY round-trip: relabel (free,K) -> (K,free) with NO register movement, then store wide.
        fa = make_fragment(a_coop_t, F16, frags[0].value)
        fb = make_fragment(b_coop_t, F16, frags[1].value)
        store_fragment(b, lds_a, make_window(lds_a_td, (zero, row_a)), fa, tid, lds_swizzle=lds_swizzle)
        store_fragment(b, lds_b, make_window(lds_b_td, (zero, row_b)), fb, tid, lds_swizzle=lds_swizzle)

    def _read_buf(row_a, row_b):
        # Read the wave's warp tile from LDS in the interleaved MMA-operand layout (K-contig via the
        # transposed relabel). ab_swap reads the N-band into slotA and the M-band into slotB.
        fa = load_fragment(
            b, lds_a, make_window(lds_a_td, (zero, b.add(row_a, b.mul(wm, warp_m_c)))),
            a_rd_t, tid, lds_swizzle=lds_swizzle,
        )
        fb = load_fragment(
            b, lds_b, make_window(lds_b_td, (zero, b.add(row_b, b.mul(wn, warp_n_c)))),
            b_rd_t, tid, lds_swizzle=lds_swizzle,
        )
        return make_fragment(slotA_wave, F16, fa.value), make_fragment(slotB_wave, F16, fb.value)

    def _prefetch_load(kb):
        # Global load uses the coop desc in its native (free, K) X-order matching a_td=(M,K)/b_td=(N,K).
        return (
            load_fragment(b, a_ptr, make_window(a_td, (m_macro, kb)), a_coop, tid),
            load_fragment(b, b_ptr, make_window(b_td, (n_macro, kb)), b_coop, tid),
        )

    def _mma_prio(a_fr, b_fr, acc_v):
        if mac_prio:
            b.s_setprio(mac_prio)
        # ab_swap: the A-slot carries the N operand, the B-slot the M operand (roles crossed).
        if ab_swap:
            out = mma(b, b_fr, a_fr, make_fragment(c_wave, F32, acc_v))
        else:
            out = mma(b, a_fr, b_fr, make_fragment(c_wave, F32, acc_v))
        if mac_prio:
            b.s_setprio(0)
        return out

    accumulator = make_fragment(c_wave, F32)
    fill_fragment(b, accumulator, 0)

    n_tiles = K_LEN // tile_k
    tk_c = b.const_i32(tile_k)
    last_k = b.const_i32(K_LEN - tile_k)

    # sched_group_barrier cadence (per trip, per wave; VEC = 8 f16 per wide mem op).
    _VEC = 8
    n_mfma = m_sub * n_sub * k_sub
    n_dsread = (warp_m * tile_k + warp_n * tile_k) // mma.wave_size // _VEC
    n_vmem = (tile_m * tile_k + tile_n * tile_k) // (mma.wave_size * n_waves) // _VEC
    n_dswrite = n_vmem

    def _pin():
        b.sched_group_barrier(_SGB_DS_READ, max(n_dsread, 1), 0)
        b.sched_group_barrier(_SGB_VMEM_READ, max(n_vmem, 1), 0)
        b.sched_group_barrier(_SGB_MFMA, max(n_mfma, 1), 0)
        b.sched_group_barrier(_SGB_DS_WRITE, max(n_dswrite, 1), 0)

    if single_buffer:
        _store_buf(_prefetch_load(zero), zero, zero)
        b.sync_lds_only()
        outer = b.scf_for_iter(
            b.const_i32(0), b.const_i32(K_LEN), tk_c, [("acc", accumulator.value)], iv_name="k",
        )
        with outer as (kiv, (acc_val,)):
            a_frag, b_frag = _read_buf(zero, zero)
            pf = _prefetch_load(b.smin(b.add(kiv, tk_c), last_k))
            acc = _mma_prio(a_frag, b_frag, acc_val)
            b.sync_lds_only()
            _store_buf(pf, zero, zero)
            _pin()
            b.sync_lds_only()
            b.scf_yield(acc.value)
        accumulator = make_fragment(c_wave, F32, outer.results[0])
    else:
        _store_buf(_prefetch_load(zero), zero, zero)  # prologue: tile 0 -> buffer 0
        b.sync_lds_only()
        outer = b.scf_for_iter(
            b.const_i32(0), b.const_i32(K_LEN - tile_k), tk_c, [("acc", accumulator.value)], iv_name="k",
        )
        with outer as (kiv, (acc_val,)):
            ki = b.div(kiv, tk_c)
            cur = b.mod(ki, b.const_i32(2))
            oth = b.sub(b.const_i32(1), cur)
            cur_ra, cur_rb = b.mul(cur, tile_m_c), b.mul(cur, tile_n_c)
            oth_ra, oth_rb = b.mul(oth, tile_m_c), b.mul(oth, tile_n_c)
            pf = _prefetch_load(b.add(kiv, tk_c))  # prefetch tile ki+1 (always in range)
            a_frag, b_frag = _read_buf(cur_ra, cur_rb)  # read tile ki
            acc = _mma_prio(a_frag, b_frag, acc_val)
            _store_buf(pf, oth_ra, oth_rb)  # store tile ki+1 -> OTHER buffer
            _pin()
            b.sync_lds_only()
            b.scf_yield(acc.value)
        acc_val = outer.results[0]
        last_cur = (n_tiles - 1) % 2  # process the last prefetched tile (N-1)
        last_ra, last_rb = b.const_i32(last_cur * tile_m), b.const_i32(last_cur * tile_n)
        a_frag, b_frag = _read_buf(last_ra, last_rb)
        accumulator = _mma_prio(a_frag, b_frag, acc_val)

    # C epilogue. The accumulator is the native interleaved C (shape (warp_m,warp_n) base, or
    # (warp_n,warp_m) crossed). Reorder to an M-inner store distribution (a pure register reorder,
    # same lane ownership) so the col-major store is wide, then write through the C descriptor.
    if ab_swap:
        # C' = C^T = (warp_n, warp_m). Store the NATIVE crossed accumulator directly through the (N, M)-
        # permuted C view: each register's crossed coord (nc, mc) lands at addr = mc*1 + nc*ldc -- the
        # SAME col-major C address as the base path. In the crossed layout M is lane-major, so this
        # M-contiguous store wave-coalesces WITHOUT any epilogue reorder (the ab_swap payoff).
        c_view = c_td.permute([1, 0])                       # (N, M): addr(n, m) = m*1 + n*ldc
        c_win = make_window(
            c_view,
            (b.add(n_macro, b.mul(wn, warp_n_c)), b.add(m_macro, b.mul(wm, warp_m_c))),
        )
    else:
        c_store = _c_store_desc(m_sub, n_sub)               # over (warp_m, warp_n); axis0 = M (inner)
        accumulator = transform_fragment(b, accumulator, c_store)
        c_view = c_td                                       # (M, N): addr(m, n) = m*1 + n*ldc
        c_win = make_window(
            c_view,
            (b.add(m_macro, b.mul(wm, warp_m_c)), b.add(n_macro, b.mul(wn, warp_n_c))),
        )
    store_fragment(b, c_ptr, c_win, accumulator, tid)
    b.ret()
    return b.kernel, mma


def _compile_launcher(kernel, arch: str):
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher

    artifact = compile_kernel(kernel, arch=arch)
    signature = (
        SignatureBuilder()
        .ptr("A", "f16").ptr("B", "f16").ptr("C", "f32")
        .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
        .build()
    )
    return KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )


def _crc_host_arrays(M_LEN, N_LEN, K_LEN, *, zeros=False):
    """Host arrays in CRC byte order. A col-major (M,K) has the SAME bytes as a C-contiguous (K,M);
    likewise B->(K,N), C->(N,M). We upload those C-contiguous transposes (as_u8_buffer needs
    C-contiguity) and reinterpret on readback. Returns (A_logical, B_logical, A_dev, B_dev, C_dev)."""
    if zeros:
        a = np.zeros((M_LEN, K_LEN), dtype=np.float16)
        bm = np.zeros((N_LEN, K_LEN), dtype=np.float16)
    else:
        rng = np.random.default_rng(0)
        a = rng.integers(-3, 4, size=(M_LEN, K_LEN)).astype(np.float16)
        bm = rng.integers(-3, 4, size=(N_LEN, K_LEN)).astype(np.float16)
    a_dev = np.ascontiguousarray(a.T)          # (K,M) c-contig == A (M,K) col-major bytes
    b_dev = np.ascontiguousarray(bm.T)         # (K,N) c-contig == B (N,K) col-major bytes
    c_dev = np.zeros((N_LEN, M_LEN), dtype=np.float32)  # (N,M) c-contig == C (M,N) col-major bytes
    return a, bm, a_dev, b_dev, c_dev


def run_and_verify_crc(
    M_LEN: int = 256,
    N_LEN: int = 256,
    K_LEN: int = 64,
    *,
    arch: str = "gfx90a",
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 32,
    waves_m: int = 4,
    waves_n: int = 4,
    ab_swap: bool = False,
    single_buffer: bool = False,
    mac_prio: int = 0,
    lds_swizzle=False,
) -> dict:
    """Compile, launch, verify bit-exact (max_abs_diff == 0.0) vs numpy ``a @ b`` (integer inputs).
    Gated on ``get_device_arch(0)``. Torch-free: numpy host arrays + DeviceMem."""
    from rocke.runtime.hip_module import Runtime, get_device_arch
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig, synchronize_and_release

    dev_arch = get_device_arch(0)
    if dev_arch != arch:
        raise RuntimeError(f"device arch {dev_arch!r} != requested {arch!r}; run on a {arch} GPU")

    kernel, mma = build_crc_gemm(
        M_LEN, N_LEN, K_LEN, arch=arch, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        waves_m=waves_m, waves_n=waves_n, ab_swap=ab_swap, single_buffer=single_buffer,
        mac_prio=mac_prio, lds_swizzle=lds_swizzle,
    )
    launcher = _compile_launcher(kernel, arch)
    block = (mma.wave_size * waves_m * waves_n, 1, 1)

    a, bm, a_dev_h, b_dev_h, c_dev_h = _crc_host_arrays(M_LEN, N_LEN, K_LEN)
    grid = (-(-N_LEN // tile_n), -(-M_LEN // tile_m), 1)

    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_dev_h, b_dev_h, c_dev_h))
    rt.memcpy_h2d(a_dev.ptr(), as_u8_buffer(a_dev_h), a_dev_h.nbytes)
    rt.memcpy_h2d(b_dev.ptr(), as_u8_buffer(b_dev_h), b_dev_h.nbytes)
    rt.memcpy_h2d(c_dev.ptr(), as_u8_buffer(c_dev_h), c_dev_h.nbytes)
    launcher(
        {"A": a_dev, "B": b_dev, "C": c_dev, "M": M_LEN, "N": N_LEN, "K": K_LEN},
        config=LaunchConfig(grid=grid, block=block),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(c_dev_h), c_dev.ptr(), c_dev_h.nbytes)

    reference = a.astype(np.float32) @ bm.astype(np.float32).T   # logical (M, N)
    result = c_dev_h.T.astype(np.float32)                        # (N,M) c-contig -> (M,N) col-major
    max_abs_diff = float(np.abs(result - reference).max())
    return {
        "shape": (M_LEN, N_LEN, K_LEN),
        "tile": (tile_m, tile_n, tile_k),
        "waves": (waves_m, waves_n),
        "ab_swap": ab_swap,
        "op_id": mma.op_id,
        "max_abs_diff": max_abs_diff,
        "bit_exact": max_abs_diff == 0.0,
    }


def benchmark_crc(
    M_LEN: int = 256,
    N_LEN: int = 256,
    K_LEN: int = 256,
    *,
    arch: str = "gfx90a",
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 32,
    waves_m: int = 4,
    waves_n: int = 4,
    ab_swap: bool = False,
    single_buffer: bool = False,
    mac_prio: int = 0,
    lds_swizzle=False,
    cold_niters: int = 5,
    nrepeat: int = 20,
) -> dict:
    """hipEvent-timed wall-time + TFLOPS (2*M*N*K / time). Zero inputs (timing only)."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem, StreamConfig, launch_kernel, make_kernel, wait_stream_and_release,
    )

    kernel, mma = build_crc_gemm(
        M_LEN, N_LEN, K_LEN, arch=arch, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        waves_m=waves_m, waves_n=waves_n, ab_swap=ab_swap, single_buffer=single_buffer,
        mac_prio=mac_prio, lds_swizzle=lds_swizzle,
    )
    launcher = _compile_launcher(kernel, arch)

    _, _, a_dev_h, b_dev_h, c_dev_h = _crc_host_arrays(M_LEN, N_LEN, K_LEN, zeros=True)
    grid = (-(-N_LEN // tile_n), -(-M_LEN // tile_m), 1)

    rt = Runtime()
    a_dev, b_dev, c_dev = (DeviceMem(x.nbytes) for x in (a_dev_h, b_dev_h, c_dev_h))
    for dev, host in ((a_dev, a_dev_h), (b_dev, b_dev_h), (c_dev, c_dev_h)):
        rt.memcpy_h2d(dev.ptr(), as_u8_buffer(host), host.nbytes)

    closure = make_kernel(
        launcher,
        {"A": a_dev, "B": b_dev, "C": c_dev, "M": M_LEN, "N": N_LEN, "K": K_LEN},
        grid, (mma.wave_size * waves_m * waves_n, 1, 1),
    )
    ms = launch_kernel(
        StreamConfig(time_kernel=True, cold_niters=cold_niters, nrepeat=nrepeat, is_gpu_timer=True),
        closure,
    )
    wait_stream_and_release()
    tflops = (2.0 * M_LEN * N_LEN * K_LEN) / (ms * 1e-3) / 1e12
    return {
        "shape": (M_LEN, N_LEN, K_LEN),
        "tile": (tile_m, tile_n, tile_k),
        "ab_swap": ab_swap,
        "ms": ms,
        "tflops": tflops,
    }


if __name__ == "__main__":
    print(run_and_verify_crc(256, 256, 64, ab_swap=False))
    print(run_and_verify_crc(256, 256, 64, ab_swap=True))
    print(run_and_verify_crc(256, 128, 64, ab_swap=False))
    print(run_and_verify_crc(256, 128, 64, ab_swap=True))
