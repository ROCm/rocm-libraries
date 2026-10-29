# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense direct wgrad on wave32 WMMA (gfx1250), streamed row by row.

    dW[k][r][s][c] = sum_{n, y, x} dY[n][y + PAD - r][x + PAD - s][k] * X[n][y][x][c]

for 3x3-class filters at stride 1, with the channel counts at runtime. Per
input row ``y`` and 32 output columns ``q0 ..`` (the WMMA K), a wave owning
``tiles_k x tiles_c`` 16x16 tiles of dW per tap computes

    A[k][j] = dY[y + PAD - r][q0 + j][k]          (M = 16 dY channels)
    B[j][c] = X[y][q0 + j + s - PAD][c]           (N = 16 X channels)
    acc[r][s][tk][tc] += A B

for every tap. Both operands want a lane per channel and 16 consecutive
pixels per lane: the rows are staged in LDS as ``[column][channel]`` and
``ds_load_tr16_b128`` transposes on the read; the tap shift ``s`` is a
whole-pixel offset of the read address. The dY fragments of the ``KH - 1``
previous rows stay in a register ring, so a row reads ``tiles_k`` dY and
``KW * tiles_c`` X fragments for ``KH * KW * tiles_k * tiles_c`` WMMAs.

A block owns one ``block_k x block_c`` dW tile and walks a flat list of row
steps: unit ``u = qc * N + n`` (column chunk ``qc``, image ``n``) holds
``H + KH - 1`` steps, the first ``KH - 1`` of which only fill the dY ring (their
input rows lie above the image and load as zeros). The list is cut into
``n_split`` equal runs, one per block along grid z, so the block count is free
of the image count; a run that starts inside a unit first replays ``KH - 1``
steps with its X rows masked, refilling the ring without counting any (row,
tap) pair twice. The copies never drain between units.

Copies are per-lane ``global_load_async_to_lds`` into a ring of ``stages``
row buffers. Every lane issues every copy (a padding lane copies element 0
into a dummy slot past the buffer and zero-fills its own slot with an LDS
store), so each wave counts the same copies per step and
``s_wait_asynccnt`` can leave the younger steps in flight.

The block adds its tile into the zeroed fp32 dW with cascade atomics.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import CPOL_ATOMIC_CASCADE_DEV, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    dilation_reason,
    emit_direct_params,
)

_WAVE = 32
_TILE = 16  # WMMA M = N
_KSTEP = 32  # output columns per WMMA (WMMA K)
_PAD_CH = 8  # LDS row padding (channels) against bank conflicts
_DUMMY = 8  # elements past each buffer: the padding lanes' copy target
_VGPR_BUDGET = 480  # accumulators + fragments a wave may hold (512-VGPR file)


@dataclass(frozen=True)
class DirectWgradWmmaSpec:
    problem: DirectConvProblem
    name: str = "direct_wgrad_wmma"
    waves_k: int = 4
    waves_c: int = 2
    tiles_k: int = 2  # 16x16 dW tiles per tap along K, per wave
    tiles_c: int = 2  # ... along C
    q_steps: int = 1  # 32-column WMMA steps per column chunk
    stages: int = 3  # LDS row buffers (steps in flight + 1)
    target_blocks: int = 256
    reuse: bool = False  # WMMA operand-reuse hints
    prefetch: bool = False  # read step g + 1's fragments during step g's WMMAs
    pad: int = _PAD_CH  # LDS row padding (channels), against bank conflicts
    sched: bool = False  # fence the copies and reads ahead of the WMMAs; setprio
    fast_copy: bool = False  # unmasked copies on interior steps (uniform branch)
    xcd: int = 0  # XCDs to group the runs by (1-D grid); 0: tile x run grid
    tdm: bool = False  # rows move by TDM (wave 0: X, wave 1: dY), not per-lane copies
    pingpong: bool = False  # two wave halves offset by one barrier phase
    rows: int = 1  # input rows staged per barrier (a stage holds ``rows`` rows)
    # > 1: two-stage. The step list is cut into ws_replicas runs, run r stores
    # its tile into fp32 scratch slab r (plain stores, no atomics, no zeroing)
    # and conv_wgrad_workspace_reduce folds the slabs into dW.
    ws_replicas: int = 1

    @property
    def threads(self) -> int:
        return self.waves_k * self.waves_c * _WAVE

    @property
    def block_k(self) -> int:
        return _TILE * self.tiles_k * self.waves_k

    @property
    def block_c(self) -> int:
        return _TILE * self.tiles_c * self.waves_c

    @property
    def block_q(self) -> int:
        return _KSTEP * self.q_steps

    def x_cols(self) -> int:
        return self.block_q + self.problem.KW - 1

    def pitch_c(self) -> int:
        return self.block_c + self.pad

    def pitch_k(self) -> int:
        return self.block_k + self.pad

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        if self.reuse:
            flags["reuse"] = True
        if self.prefetch:
            flags["pf"] = True
        if self.sched:
            flags["sched"] = True
        if self.fast_copy:
            flags["fc"] = True
        if self.xcd:
            flags[f"xcd{self.xcd}"] = True
        if self.tdm:
            flags["tdm"] = True
        if self.pingpong:
            flags["pp"] = True
        if self.rows > 1:
            flags[f"g{self.rows}"] = True
        if self.ws_replicas > 1:
            flags[f"wsr{self.ws_replicas}"] = True
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}",
            f"w{self.waves_k}x{self.waves_c}",
            f"t{self.tiles_k}x{self.tiles_c}",
            f"qs{self.q_steps}",
            f"st{self.stages}",
            f"tb{self.target_blocks}",
            *([f"pad{self.pad}"] if self.pad != _PAD_CH else []),
            flags=flags,
        )

    def n_k_tiles(self) -> int:
        return ceil_div(self.problem.total_k, self.block_k)

    def n_c_tiles(self) -> int:
        return ceil_div(self.problem.total_c, self.block_c)

    def total_steps(self) -> int:
        p = self.problem
        return ceil_div(p.Wo, self.block_q) * p.N * (p.H + p.KH - 1)

    def n_split(self) -> int:
        if self.ws_replicas > 1:
            return self.ws_replicas
        planes = self.n_k_tiles() * self.n_c_tiles()
        # Nearest whole number of runs: the grid lands closest to
        # target_blocks (a multiple of the CU count), not just above it.
        runs = (self.target_blocks + planes // 2) // planes
        return max(1, min(self.total_steps(), runs))

    def launch_geometry(self):
        planes = self.n_c_tiles() * self.n_k_tiles()
        if self.xcd:
            return (planes * self.n_split(), 1, 1), (self.threads, 1, 1)
        return (planes, 1, self.n_split()), (self.threads, 1, 1)

    def vgprs(self) -> int:
        p = self.problem
        frags = self.tiles_k * self.q_steps * p.KH + self.tiles_c
        if self.prefetch:  # a whole step's fragments in flight
            frags += self.q_steps * (self.tiles_k + p.KW * self.tiles_c)
        return 8 * (p.KH * p.KW * self.tiles_k * self.tiles_c + frags)

    def lds_bytes(self) -> int:
        x_rows = self.x_cols() + (self.problem.PAD if self.tdm else 0)
        per_stage = x_rows * self.pitch_c() + self.block_q * self.pitch_k()
        return self.stages * self.rows * (per_stage + 2 * _DUMMY) * 2


def is_valid_direct_wgrad_wmma_spec(spec: DirectWgradWmmaSpec, arch: str = "gfx1250"):
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if arch != "gfx1250":
        return False, "ds_load_tr16_b128 and wmma 16x16x32 need gfx1250"
    if p.groups != 1:
        return False, "groups == 1 only"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if p.stride != 1:
        return False, "stride 1 only"
    if p.PAD > p.KH - 1 or p.PAD > p.KW - 1:
        return False, "PAD <= K - 1"
    if p.total_c % 8 or p.total_k % 8:
        return False, "C and K must be multiples of 8 (16-byte chunks)"
    if spec.waves_k * spec.waves_c not in (2, 4, 8, 16):
        return False, "2, 4, 8 or 16 waves"
    if spec.tiles_k not in (1, 2, 4) or spec.tiles_c not in (1, 2, 4):
        return False, "tiles_k / tiles_c in (1, 2, 4)"
    if spec.q_steps not in (1, 2):
        return False, "q_steps in (1, 2)"
    nw = spec.waves_k * spec.waves_c
    if spec.tdm and 2 * spec.rows > nw and (2 * spec.rows) % nw:
        return False, "tdm: 2 * rows transfers must spread evenly over the waves"
    if spec.rows < 1 or (spec.rows > 1 and spec.pingpong):
        return False, "rows >= 1 (1 with pingpong)"
    if spec.rows > 1 and spec.prefetch and spec.stages < 3:
        return False, "rows > 1 with prefetch needs stages >= 3"
    if spec.pingpong and (spec.prefetch or (spec.waves_k * spec.waves_c) % 8):
        return False, "pingpong: a multiple of 8 waves, no prefetch"
    if spec.pad < 0 or spec.pad % 8:
        return False, "pad: a multiple of 8 channels (16-byte rows)"
    if not 2 <= spec.stages <= 8:
        return False, "stages in 2..8"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    if spec.ws_replicas < 1:
        return False, "ws_replicas >= 1"
    if spec.vgprs() > _VGPR_BUDGET:
        return False, f"~{spec.vgprs()} VGPRs of accumulators and fragments"
    if spec.lds_bytes() > 300 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 300 KiB"
    return True, "ok"


def build_direct_wgrad_wmma(
    spec: DirectWgradWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    ok, why = is_valid_direct_wgrad_wmma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DirectWgradWmmaSpec: {why}")
    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    TK, TC, QS = spec.tiles_k, spec.tiles_c, spec.q_steps
    BK, BC, BQ = spec.block_k, spec.block_c, spec.block_q
    XC = spec.x_cols()
    PC, PK = spec.pitch_c(), spec.pitch_k()
    T = spec.threads
    D = spec.stages
    G = spec.rows
    NS = D * G  # row slots: slot d * G + g holds row g of stage d
    PRE = KH - 1  # ring steps before a unit's first input row
    io_type = _io_type(p.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1 = ci(0), ci(1)
    p_N = params["p_N"]
    p_C = params["p_total_c"]
    p_K = params["p_total_k"]
    p_H, p_W = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    sdy_n, sdy_h, sdy_w = (
        params["p_dY_stride_n"],
        params["p_dY_stride_ho"],
        params["p_dY_stride_wo"],
    )
    sx_n, sx_h, sx_w = (
        params["p_X_stride_n"],
        params["p_X_stride_hi"],
        params["p_X_stride_wi"],
    )
    dy_ptr, x_ptr = params["A"], params["B"]

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    wave_k = b.mod(wave, ci(spec.waves_k))
    wave_c = b.div(wave, ci(spec.waves_k))

    # ---- grid: tile = k_tile * n_c_tiles + c_tile, run of the step list ----
    n_ct = b.div(b.add(p_C, ci(BC - 1)), ci(BC))
    n_kt = b.div(b.add(p_K, ci(BK - 1)), ci(BK))
    n_qc = b.div(b.add(p_Wo, ci(BQ - 1)), ci(BQ))
    spu = b.add(p_H, ci(PRE))  # steps per unit
    total = b.mul(b.mul(n_qc, p_N), spu)
    # Same formula as DirectWgradWmmaSpec.n_split, which sizes the grid.
    planes = b.mul(n_ct, n_kt)
    if spec.ws_replicas > 1:
        n_split = ci(spec.ws_replicas)
    else:
        runs = b.div(b.add(ci(spec.target_blocks), b.div(planes, ci(2))), planes)
        n_split = b.smax(b.smin(total, runs), c1)
    if spec.xcd:
        # 1-D grid. Workgroups go to the XCDs round-robin, so hardware block
        # h runs on XCD h % xcd: renumber the blocks so that each XCD gets a
        # contiguous range of logical blocks, i.e. every tile of a few runs.
        # The tiles of one run read the same X and dY rows, which then cross
        # into that XCD's L2 once instead of into every XCD's.
        hw = b.block_id_x()
        n_blocks = b.mul(planes, n_split)
        per_xcd = b.div(n_blocks, ci(spec.xcd))
        even = b.cmp_eq(b.mul(per_xcd, ci(spec.xcd)), n_blocks)
        logical = b.select(
            even,
            b.add(b.mul(b.mod(hw, ci(spec.xcd)), per_xcd), b.div(hw, ci(spec.xcd))),
            hw,
        )
        tile = b.mod(logical, planes)
        run = b.div(logical, planes)
    else:
        tile, run = b.block_id_x(), b.block_id_z()
    c_blk0 = b.mul(b.mod(tile, n_ct), ci(BC))
    k_blk0 = b.mul(b.div(tile, n_ct), ci(BK))
    per_run = b.div(b.add(total, b.sub(n_split, c1)), n_split)
    g0 = b.mul(run, per_run)
    g1 = b.smin(b.add(g0, per_run), total)
    g_first = b.sub(g0, ci(PRE))  # replay PRE ring steps before the run

    # ---- LDS ---------------------------------------------------------------
    X_DUMMY, Y_DUMMY = XC * PC, BQ * PK
    # TDM lands the X row PAD rows down on the left image edge: slack rows.
    X_ELEMS = (XC + PAD) * PC if spec.tdm else X_DUMMY + _DUMMY
    x_stage = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xs{i}") for i in range(NS)
    ]
    y_stage = [
        b.smem_alloc(io_type, [1, Y_DUMMY + _DUMMY], name_hint=f"ys{i}")
        for i in range(NS)
    ]

    # ---- copy slots: slot = i * T + tid -> (column, 8-channel chunk) -------
    def copy_slots(cols: int, chans: int, pitch: int, ch0: Value, limit: Value):
        out = []
        ch8 = chans // 8
        for i in range(ceil_div(cols * ch8, T)):
            slot = b.add(ci(i * T), tid)
            col = b.div(slot, ci(ch8))
            ch = b.mul(b.mod(slot, ci(ch8)), ci(8))
            ok = b.cmp_lt(b.add(ch0, ch), limit)
            in_tile = None
            if (i + 1) * T > cols * ch8:  # the last pass has idle lanes
                in_tile = b.cmp_lt(col, ci(cols))
                ok = b.land(ok, in_tile)
            out.append((col, b.add(b.mul(col, ci(pitch)), ch), ok, in_tile, ch))
        return out

    x_slots = []
    for col, lds_i, ok, in_tile, ch in copy_slots(XC, BC, PC, c_blk0, p_C):
        x_slots.append((lds_i, ok, in_tile, col, b.add(c_blk0, ch)))
    y_slots = []
    for col, lds_i, ok, in_tile, ch in copy_slots(BQ, BK, PK, k_blk0, p_K):
        y_slots.append((lds_i, ok, in_tile, col, b.add(k_blk0, ch)))
    PER_STEP = len(x_slots) + len(y_slots)  # copies per row step, every lane

    def fast_slots(slots, dummy, stride):
        out = []
        for lds_i, _, in_tile, col, ch in slots:
            src = b.add(b.mul(col, stride), ch)
            if in_tile is not None:
                lds_i = b.select(in_tile, lds_i, ci(dummy))
                src = b.select(in_tile, src, c0)
            out.append((lds_i, src))
        return out

    # Slot offsets relative to the row's first column (fast path only).
    x_fast = fast_slots(x_slots, X_DUMMY, sx_w)
    y_fast = fast_slots(y_slots, Y_DUMMY, sdy_w)
    full_tiles = b.land(
        b.cmp_le(b.add(c_blk0, ci(BC)), p_C), b.cmp_le(b.add(k_blk0, ci(BK)), p_K)
    )
    zero8 = b.zero_vec(io_type, 8)

    def copy(src, stage, dummy, lds_i, live, in_tile, src_off):
        b.global_load_async_to_lds(
            src,
            b.select(live, src_off, c0),
            stage,
            [c0, b.select(live, lds_i, ci(dummy))],
            width_bytes=16,
        )
        pad = b.lnot(live) if in_tile is None else b.land(in_tile, b.lnot(live))
        with b.scf_if(pad):
            b.smem_store_vN(stage, [c0, lds_i], zero8, 8)

    # The copy cursor (g, y, n, q0): step g of the list is input row y of
    # image n in the column chunk starting at q0. It advances one step per
    # copy without dividing; steps before the list (g < 0) only mask.
    def cursor_at(g: Value):
        gpos = b.smax(g, c0)
        unit = b.div(gpos, spu)
        y = b.sub(b.sub(b.mod(gpos, spu), ci(PRE)), b.sub(gpos, g))
        return g, y, b.mod(unit, p_N), b.mul(b.div(unit, p_N), ci(BQ))

    def advance(cur):
        g, y, n, q0 = cur
        y1 = b.add(y, c1)
        wrap = b.cmp_eq(y1, p_H)
        n1 = b.select(wrap, b.add(n, c1), n)
        wrap_n = b.cmp_eq(n1, p_N)
        return (
            b.add(g, c1),
            b.select(wrap, ci(-PRE), y1),
            b.select(wrap_n, c0, n1),
            b.select(wrap_n, b.add(q0, ci(BQ)), q0),
        )

    def issue(buf: int, cur):
        """Copy the cursor's step: X row y (masked before the run) and dY row y + PAD."""
        g, y, n_img, q0 = cur
        live_g = b.land(b.cmp_lt(g, g1), b.cmp_ge(g, c0))
        x_ok = b.land(
            b.land(live_g, b.cmp_ge(g, g0)),
            b.land(b.cmp_ge(y, c0), b.cmp_lt(y, p_H)),
        )
        x_row = b.add(b.mul(n_img, sx_n), b.mul(y, sx_h))
        pr = b.add(y, ci(PAD))
        y_ok = b.land(live_g, b.land(b.cmp_ge(pr, c0), b.cmp_lt(pr, p_Ho)))
        y_row = b.add(b.mul(n_img, sdy_n), b.mul(pr, sdy_h))
        if spec.tdm:
            tdm_copies(buf, n_img, y, q0, x_ok, pr, y_ok)
            return
        if not spec.fast_copy:
            masked_copies(buf, q0, x_ok, x_row, y_ok, y_row)
            return
        # Interior step: both rows inside the image, the column chunk and
        # its halo inside the row, whole channel tiles. Then no copy needs a
        # mask: one uniform branch picks the unmasked copies.
        w0 = b.sub(q0, ci(PAD))
        interior = b.land(
            b.land(b.land(x_ok, y_ok), full_tiles),
            b.land(
                b.land(b.cmp_ge(w0, c0), b.cmp_le(b.add(w0, ci(XC)), p_W)),
                b.cmp_le(b.add(q0, ci(BQ)), p_Wo),
            ),
        )
        with b.scf_if_else(interior) as (fast, edge):
            with fast:
                xb = b.add(x_row, b.mul(w0, sx_w))
                for lds_f, src_f in x_fast:
                    b.global_load_async_to_lds(
                        x_ptr,
                        b.add(xb, src_f),
                        x_stage[buf],
                        [c0, lds_f],
                        width_bytes=16,
                    )
                yb = b.add(y_row, b.mul(q0, sdy_w))
                for lds_f, src_f in y_fast:
                    b.global_load_async_to_lds(
                        dy_ptr,
                        b.add(yb, src_f),
                        y_stage[buf],
                        [c0, lds_f],
                        width_bytes=16,
                    )
            with edge:
                masked_copies(buf, q0, x_ok, x_row, y_ok, y_row)

    def wait_copies(stages_in_flight: int):
        """Wait until at most ``stages_in_flight`` stages' copies are outstanding."""
        if spec.tdm:
            # One transfer per row per issuing wave.
            b.s_wait_tensorcnt(stages_in_flight * TDM_PER_WAVE)
        else:
            b.s_wait_asynccnt(stages_in_flight * G * PER_STEP)

    if spec.tdm:
        from rocke.core.ir import I32

        x_lds = [b.trunc(b.smem_addr_of(st), I32) for st in x_stage]
        y_lds = [b.trunc(b.smem_addr_of(st), I32) for st in y_stage]
        c_ext = b.smax(b.smin(b.sub(p_C, c_blk0), ci(BC)), c0)
        k_ext = b.smax(b.smin(b.sub(p_K, k_blk0), ci(BK)), c0)
    # A stage's 2 * G transfers (X and dY of each row) go to waves 0 .. 2G-1
    # (mod the wave count), so each issuing wave counts the same transfers
    # per stage: a wave's TDM transfers are issued one after the other.
    NW = spec.waves_k * spec.waves_c
    TDM_PER_WAVE = max(1, (2 * G) // NW)

    def tdm_copies(buf, n_img, y, q0, x_ok, pr, y_ok):
        """One TDM transfer per operand. The hardware zero-fills what lies
        past the clipped extents: rows outside the image (extent 0), columns
        past the row, channels past the tile (the pitch's pad included). A
        transfer cannot start left of column 0, so on the left image edge
        the X row starts at column 0 and lands ``-w0`` rows down; the rows
        above it are zeroed with LDS stores."""
        w0 = b.sub(q0, ci(PAD))
        w_start = b.smax(w0, c0)
        shift = b.sub(w_start, w0)
        g_row = buf % G
        with b.scf_if(b.cmp_eq(wave, ci((2 * g_row) % NW))):
            x_cols = b.smax(b.smin(b.sub(p_W, w_start), ci(XC)), c0)
            off = b.add(
                b.add(b.mul(n_img, sx_n), b.mul(b.select(x_ok, y, c0), sx_h)),
                b.add(b.mul(w_start, sx_w), c_blk0),
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(x_ptr, b.mul(off, ci(2))),
                    b.add(x_lds[buf], b.mul(b.mul(shift, ci(PC)), ci(2))),
                    elem_bytes=2,
                    tensor_dim0=c_ext,
                    tensor_dim1=b.select(x_ok, x_cols, c0),
                    row_stride=sx_w,
                    tile_dim0=PC,
                    tile_dim1=XC,
                )
            )
        with b.scf_if(b.cmp_eq(wave, ci((2 * g_row + 1) % NW))):
            y_cols = b.smax(b.smin(b.sub(p_Wo, q0), ci(BQ)), c0)
            off = b.add(
                b.add(b.mul(n_img, sdy_n), b.mul(b.select(y_ok, pr, c0), sdy_h)),
                b.add(b.mul(q0, sdy_w), k_blk0),
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(dy_ptr, b.mul(off, ci(2))),
                    y_lds[buf],
                    elem_bytes=2,
                    tensor_dim0=k_ext,
                    tensor_dim1=b.select(y_ok, y_cols, c0),
                    row_stride=sdy_w,
                    tile_dim0=PK,
                    tile_dim1=BQ,
                )
            )
        if PAD:
            chunks = PAD * PC // 8
            with b.scf_if(b.cmp_gt(shift, c0)):
                for j in range(0, chunks, T):
                    idx = b.add(tid, ci(j))
                    with b.scf_if(b.cmp_lt(idx, b.div(b.mul(shift, ci(PC)), ci(8)))):
                        b.smem_store_vN(x_stage[buf], [c0, b.mul(idx, ci(8))], zero8, 8)

    def masked_copies(buf, q0, x_ok, x_row, y_ok, y_row):
        for lds_i, ok, in_tile, col, cc in x_slots:
            wi = b.add(b.sub(q0, ci(PAD)), col)
            live = b.land(b.land(ok, x_ok), b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)))
            off = b.add(x_row, b.add(b.mul(wi, sx_w), cc))
            copy(x_ptr, x_stage[buf], X_DUMMY, lds_i, live, in_tile, off)
        for lds_i, ok, in_tile, col, kk in y_slots:
            wo = b.add(q0, col)
            live = b.land(b.land(ok, y_ok), b.cmp_lt(wo, p_Wo))
            off = b.add(y_row, b.add(b.mul(wo, sdy_w), kk))
            copy(dy_ptr, y_stage[buf], Y_DUMMY, lds_i, live, in_tile, off)

    # ---- fragment reads: ds_load_tr16_b128 of a 16 (column) x 16 (channel)
    # block of [column][channel]: lane l reads column (l // 16) * 16 + l % 8
    # (+ 8 for the high half) at channels (l % 16) // 8 * 8 ..; after the
    # transpose it holds channel l % 16 at columns (l // 16) * 16 + 0 .. 15.
    l16 = b.mod(lane, ci(16))
    lhalf = b.div(lane, ci(16))
    tr_row = b.add(b.mul(lhalf, ci(16)), b.mod(lane, ci(8)))
    tr_ch = b.mul(b.div(l16, ci(8)), ci(8))
    y_base = b.add(b.mul(tr_row, ci(PK)), b.add(b.mul(wave_k, ci(TK * _TILE)), tr_ch))
    x_base = b.add(b.mul(tr_row, ci(PC)), b.add(b.mul(wave_c, ci(TC * _TILE)), tr_ch))

    def frag(stage, base: Value, pitch: int, col: int, ch: int) -> Value:
        addr = b.add(base, ci(col * pitch + ch))
        lo = b.ds_read_tr16_b128(stage, c0, addr, dtype=io_type)
        hi = b.ds_read_tr16_b128(stage, c0, b.add(addr, ci(8 * pitch)), dtype=io_type)
        return b.vec_concat(lo, hi)

    def wmma(a: Value, bb: Value, acc: Value, reuse_a: bool) -> Value:
        attrs = {"reuse_a": True} if (spec.reuse and reuse_a) else None
        if p.dtype == "bf16":
            op = "wmma_gfx1250_f32_16x16x32_bf16"
        else:
            op = "wmma_gfx1250_f32_16x16x32_f16"
        return b.mma(op, a, bb, acc, attrs=attrs)

    def acc_i(r, s, tk, tc):
        return ((r * KW + s) * TK + tk) * TC + tc

    n_acc = KH * KW * TK * TC
    n_ring = (KH - 1) * TK * QS  # dY fragments of the KH - 1 previous rows
    zero_acc = b.zero_vec_f32(8)
    zero_frag = b.zero_vec(io_type, 16)

    def pack(accs, ring):
        return list(accs) + [v for row in ring for v in row]

    def unpack(vals):
        accs = list(vals[:n_acc])
        flat = list(vals[n_acc:])
        per = TK * QS
        return accs, [flat[j * per : (j + 1) * per] for j in range(KH - 1)]

    N_DY = TK * QS  # newest dY fragments of a step: [tk * QS + q]
    N_X = QS * KW * TC  # X fragments of a step: [(q * KW + s) * TC + tc]

    def read_frags(buf: int):
        dy = [
            frag(y_stage[buf], y_base, PK, _KSTEP * q, _TILE * tk)
            for tk in range(TK)
            for q in range(QS)
        ]
        xs = [
            frag(x_stage[buf], x_base, PC, _KSTEP * q + s, _TILE * tc)
            for q in range(QS)
            for s in range(KW)
            for tc in range(TC)
        ]
        return dy + xs

    def multiply(accs, ring, frags):
        newest, xs = frags[:N_DY], frags[N_DY:]
        full = [newest] + list(ring)  # full[r]: the dY row of tap r
        accs = list(accs)
        for q in range(QS):
            for tk in range(TK):
                for r in range(KH):
                    a = full[r][tk * QS + q]
                    group = [(s, tc) for s in range(KW) for tc in range(TC)]
                    for j, (s, tc) in enumerate(group):
                        i = acc_i(r, s, tk, tc)
                        bb = xs[(q * KW + s) * TC + tc]
                        last = j == len(group) - 1
                        accs[i] = wmma(a, bb, accs[i], reuse_a=not last)
        return accs, full[: KH - 1]

    def issue_stage(d: int, cur):
        for g in range(G):
            issue(d * G + g, cur)
            cur = advance(cur)
        return cur

    def mult(accs, ring, frags):
        if spec.sched:
            # Keep the copies and the reads ahead of the WMMAs: the reads
            # then land during the WMMAs instead of before the next barrier.
            b.sched_barrier(0)
            b.s_setprio(1)
        accs, ring = multiply(accs, ring, frags)
        if spec.sched:
            b.s_setprio(0)
        return accs, ring

    def stage(cur, d: int, accs, ring, frags):
        """Stage j (``d = j % D``): ``G`` rows behind one barrier."""
        if spec.pingpong:
            # L phase: stage j's fragments, the cursor's copy (stage j + D - 1,
            # into the buffer stage j - 1 left), then stage j + 1's copies and
            # every LDS access done before the barrier. M phase: WMMAs only.
            # One half runs a barrier phase behind the other, so on each SIMD
            # one wave's M phase overlaps the other's L phase.
            cur_frags = read_frags(d)
            cur = issue_stage((d + D - 1) % D, cur)
            wait_copies(D - 2)
            b.sync_lds_only()
            b.sched_barrier(0)
            b.s_setprio(1)
            accs, ring = multiply(accs, ring, cur_frags)
            b.s_setprio(0)
            b.sched_barrier(0)
            b.s_barrier_bare()
            b.sched_barrier(0)
            return accs, ring, [], cur
        if not spec.prefetch:
            # Stages j .. j + D - 2 are in flight: wait for stage j's copies;
            # the barrier publishes them (and the zero fills) and frees stage
            # j - 1's buffer, which the cursor's stage j + D - 1 refills.
            wait_copies(D - 2)
            b.sync_lds_only()
            cur = issue_stage((d + D - 1) % D, cur)
            for g in range(G):
                accs, ring = mult(accs, ring, read_frags(d * G + g))
            return accs, ring, [], cur
        if G == 1:
            # Stage j's fragments are already in registers. Stages j + 1 ..
            # j + D - 1 are in flight: wait for stage j + 1's copies; the
            # barrier publishes them, and (with the LDS drain before it) stage
            # j's reads are done, so the cursor's stage j + D refills buffer
            # ``d``. Stage j + 1's reads then overlap stage j's WMMAs.
            wait_copies(D - 2)
            b.sync_lds_only()
            cur = issue_stage(d, cur)
            nxt = read_frags((d + 1) % D)
            accs, ring = mult(accs, ring, frags)
            return accs, ring, nxt, cur
        # G rows: each row reads the next row's fragments during its WMMAs,
        # the last one row 0 of stage j + 1, which this barrier publishes:
        # wait for stage j + 1 (stages j + 2 .. j + D - 2 may fly). Stage
        # j - 1's buffer is free (its reads drained before the barrier).
        wait_copies(D - 3)
        b.sync_lds_only()
        cur = issue_stage((d + D - 1) % D, cur)
        for g in range(G):
            nslot = d * G + g + 1 if g + 1 < G else ((d + 1) % D) * G
            nxt = read_frags(nslot)
            accs, ring = mult(accs, ring, frags)
            frags = nxt
        return accs, ring, frags, cur

    cur = cursor_at(g_first)
    n_pre = D if (spec.prefetch and G == 1) else D - 1
    for d in range(n_pre):
        cur = issue_stage(d, cur)
    frags = []
    if spec.prefetch:
        wait_copies(n_pre - 1)
        b.sync_lds_only()
        frags = read_frags(0)
    half = b.div(wave, ci((spec.waves_k * spec.waves_c) // 2))
    if spec.pingpong:
        # Stage 0's copies published to all; then the second half drops one
        # barrier phase behind the first.
        wait_copies(D - 2)
        b.sync_lds_only()
        with b.scf_if(b.cmp_eq(half, c1)):
            b.s_barrier_bare()
        b.sched_barrier(0)
    n_frags = len(frags)
    accs = [zero_acc] * n_acc
    ring = [[zero_frag] * (TK * QS) for _ in range(KH - 1)]
    trips = b.div(b.add(b.sub(g1, g_first), ci(NS - 1)), ci(NS))
    init = pack(accs, ring) + frags + list(cur)
    loop = b.scf_for_iter(
        c0,
        trips,
        c1,
        [(f"a{i}", v) for i, v in enumerate(init)],
        iv_name="tr",
        elide_trailing_barrier=False,
    )
    with loop as (tr, carried):
        carried = list(carried)
        n_state = len(carried) - 4 - n_frags
        accs, ring = unpack(carried[:n_state])
        frags = carried[n_state : n_state + n_frags]
        cur = tuple(carried[-4:])
        for d in range(D):
            accs, ring, frags, cur = stage(cur, d, accs, ring, frags)
        b.scf_yield(*pack(accs, ring), *frags, *cur)
    if spec.pingpong:
        with b.scf_if(b.cmp_eq(half, c0)):
            b.s_barrier_bare()
    wait_copies(0)
    accs, _ = unpack(list(loop.results)[: len(init) - 4 - n_frags])

    # ---- epilogue: cascade atomics into the zeroed fp32 dW -----------------
    # Lane l, item e: dY channel k = 8 * (l // 16) + e, X channel c = l % 16.
    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])
    # Two-stage: run r's slab of the scratch (each slab is dW's layout).
    slab = b.mul(run, b.mul(b.mul(p_K, ci(KH * KW)), p_C))
    k_lane = b.add(b.add(k_blk0, b.mul(wave_k, ci(TK * _TILE))), b.mul(lhalf, ci(8)))
    c_lane = b.add(b.add(c_blk0, b.mul(wave_c, ci(TC * _TILE))), l16)
    for tc in range(TC):
        c = b.add(c_lane, ci(_TILE * tc))
        c_ok = b.cmp_lt(c, p_C)
        for tk in range(TK):
            for e in range(8):
                k = b.add(k_lane, ci(_TILE * tk + e))
                ok = b.land(c_ok, b.cmp_lt(k, p_K))
                k_row = b.mul(k, ci(KH * KW))
                with b.scf_if(ok):
                    for r in range(KH):
                        for s in range(KW):
                            off = b.add(b.mul(b.add(k_row, ci(r * KW + s)), p_C), c)
                            val = b.vec_extract(accs[acc_i(r, s, tk, tc)], e)
                            if spec.ws_replicas > 1:
                                b.global_store(
                                    params["D"], b.add(off, slab), val, align=4
                                )
                                continue
                            b.buffer_atomic_fadd_f32(
                                d_rsrc,
                                b.mul(off, ci(4)),
                                c0,
                                val,
                                cachepolicy=CPOL_ATOMIC_CASCADE_DEV,
                            )
    return b.kernel
