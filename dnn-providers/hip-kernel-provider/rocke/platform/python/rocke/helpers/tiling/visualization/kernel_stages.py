# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor -> viz adapter + PHASE recipes: turn rocKE tile descriptors into the right BASIC views for
each part of a kernel pipeline, rendered as short LINEAR strips on the tested :class:`Pipeline` spine.

Two layers, both kernel-agnostic (a specific kernel -- e.g. the CRC GEMM -- supplies its own descriptors):

- **tier-1 adapter** -- ``field_inputs``/``lds_inputs``/``coop_forward_map`` convert a ``TileDesc`` (or a raw
  ``WarpDistributionEncoding``) + LDS geometry into the exact inputs the cell-field views consume: an
  encoding, a ``{(lane,reg)->coord}`` forward map, an LDS ``addr_fn``, and the FULL cooperative (multi-wave)
  map for WG-scope views.
- **phase recipes** -- ``flow_load_phase`` / ``flow_mma_phase`` / ``flow_epilogue_phase`` pick the right
  view sequence for the load/prefetch, MMA, and epilogue phases and return a compact linear ``Pipeline``
  (or, for the MMA, the ``MmaTee``). ``scope`` is a DETAIL knob on the SAME components (axes/orientation
  unchanged): ``"wave"`` (default) shows one wave -- one atom detailed, the rest grouped; ``"wg"`` shows the
  whole workgroup by feeding each component the FULL cooperative map + ``scope="wg"`` so it collapses to one
  block per WAVE (hue = wave). The register file simply lets its tid axis grow to all lanes (banded by wave)
  and the figure auto-scales -- no bespoke per-wave maps. Epilogue branch (direct / register-shuffle / LDS
  round-trip) is auto-detected via ``classify_epilogue``.

Everything is REUSED, not re-derived: ``as_forward_map`` normalizes the layout, ``classify_transform`` prices
each hop, the LDS swizzle is replayed bit-for-bit through ``lds_conflict.NumBuilder``, and the macro tile is
enumerated via the real ``emit_tensor_coordinates``."""
from __future__ import annotations

from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.transforms import as_forward_map, classify_transform
from rocke.helpers.tiling.lds_conflict import NumBuilder
from rocke.helpers.tiling.visualization.layout_render import (
    CellGroup, FlowStage, LdsBankView, LogicalTileComponent, MmaTee, Pipeline, RegisterFileComponent,
    transactions,
)

__all__ = ["field_inputs", "lds_inputs", "coop_forward_map", "classify_epilogue",
           "flow_load_phase", "flow_mma_phase", "flow_epilogue_phase",
           "flow_lds_store_placement", "flow_lds_load_placement"]


# ================================================================ tier-1 adapter
def _as_encoding(x):
    """A ``TileDesc`` (has ``.layout``) OR a raw ``WarpDistributionEncoding`` -> the encoding. Lets the
    adapter take either a kernel descriptor or a distribution the tee already exposes (e.g. ``mma.a_layout``)."""
    return x.layout if hasattr(x, "layout") else x


def field_inputs(desc):
    """A ``TileDesc`` (or encoding) -> ``(encoding, forward_map)`` for a register / logical cell-field view.
    ``forward_map`` is ``{(lane,reg)->coord}`` via :func:`as_forward_map` (the same normalizer the tee uses)."""
    enc = _as_encoding(desc)
    return enc, as_forward_map(enc)


def lds_inputs(store_desc, *, stride, pad=0, swizzle=None):
    """A transposed LDS-store ``TileDesc`` (or encoding) + LDS geometry -> ``(store_mp, addr_fn)`` for an
    :class:`LdsBankView`. ``store_mp`` = ``{(lane,reg)->(row,col)}`` (row = the memref outer coord, col = the
    stride-1 free coord). ``addr_fn(row,col)`` = ``row*(stride+pad) + free`` in ELEMENTS, where ``free`` is
    ``col`` or -- when a kernel ``swizzle`` callable is given -- the swizzled column obtained by replaying it
    through a :class:`NumBuilder` (bit-consistent with the kernel emit; no re-derivation)."""
    store_mp = as_forward_map(_as_encoding(store_desc))
    total = stride + pad
    if swizzle is None:
        def addr_fn(row, col):
            return row * total + col
    else:
        def addr_fn(row, col):
            pos = swizzle(NumBuilder(0), [row, col])         # replay the IRBuilder swizzle as plain ints
            return row * total + pos[-1]
    return store_mp, addr_fn


def coop_forward_map(desc, *, n_waves, wave_size=64):
    """The FULL cooperative (multi-wave) forward map ``{(tid,reg)->coord}`` for a coop/macro descriptor.
    A single-wave ``RegisterMapper`` only covers ``wave_size`` lanes and mis-reads a coop encoding (its
    ``wave_dist`` collapses), so the macro tile is enumerated via the REAL emit (``emit_tensor_coordinates``
    decomposes each thread id wave-outer / lane-inner). ``tid`` spans ``n_waves*wave_size`` -- the whole
    block cooperating -- so the coords cover the entire ``[free, K]`` macro tile."""
    enc = _as_encoding(desc)
    nb = NumBuilder(0)                                        # emit takes tid explicitly; builder only evals ops
    return {(tid, reg): tuple(emit_tensor_coordinates(nb, enc, tid, reg))
            for tid in range(n_waves * wave_size) for reg in range(desc.register_count)}


# ================================================================ epilogue classifier
def classify_epilogue(c_native, c_store_desc):
    """Decide how the accumulator reaches its stored layout: ``("direct"|"reorder"|"cross_lane"|"unknown",
    note)``. ``c_native`` = the machine C register map ``{(lane,reg)->(m,n)}`` (e.g. ``MmaTee.c_mapping()``);
    ``c_store_desc`` = the store distribution (or None). identity ⇒ **direct store**; a lane-uniform register
    permutation ⇒ **register shuffle**; a cross-lane move ⇒ **LDS round-trip**. ``unknown`` (no/ambiguous
    store) ⇒ the caller should ASK the user which epilogue applies before rendering."""
    if c_store_desc is None:
        return "unknown", "no C store descriptor supplied"
    store_fm = as_forward_map(_as_encoding(c_store_desc))
    native_fm = as_forward_map(c_native)
    if native_fm == store_fm:
        return "direct", "identity (store-ready)"
    try:
        tier = classify_transform(native_fm, store_fm).tier
    except Exception:
        return "unknown", "native vs store not reconcilable by a simple transform"
    return ("reorder" if tier == "reorder" else "cross_lane"), tier


# ================================================================ phase recipes (linear Pipelines)
def _reg_file(enc_or_fm, dims, font_size, shade_map=None):
    """A register file in the NORMAL orientation (tid rows / reg cols) from an encoding OR a forward map.
    (Only the MMA *tee* transposes its B/C wings; standalone register views stay tid-vertical.) ``shade_map``
    (``{(lane,reg)->step}``) colours by vectorization TIME so one contiguous vector = one shade."""
    kw = dict(dims=dims, col_ticks_side="top", font_size=font_size, shade_map=shade_map)
    return (RegisterFileComponent(fwd_map=enc_or_fm, **kw) if isinstance(enc_or_fm, dict)
            else RegisterFileComponent(dist=enc_or_fm, **kw))


def _free_contig_addr(fwd_map):
    """A memory-order ``addr_fn`` for a FREE-DIM-CONTIGUOUS load: dim0 (the free axis, e.g. M for A, N for B)
    is stride-1, dim1 (K) strides past it. A thread's contiguous free run then reads as ONE address run ->
    ONE vectorization shade (the column-major reality), instead of one shade per register."""
    ext0 = max((c[0] for c in fwd_map.values()), default=0) + 1
    return lambda c0, c1: c0 + c1 * ext0


def flow_load_phase(*, load_desc, dims, store_desc=None, dest="lds", scope="wave", cooperative=False,
                    n_waves=1, wave_size=64, wave=0, wave_grid=None, row_coord=0, macro_note="", stride=None,
                    pad=0, swizzle=None, nbanks=32, reg_desc=None, font_size=6.0, title=""):
    """LOAD / PREFETCH phase (linear strip): **global thread-tile** -> **register file (after load)** ->
    **destination** (LDS banks, or a register-prefetch file). ``cooperative=True`` treats
    ``load_desc``/``store_desc`` as MACRO cooperative descriptors (enumerated across all ``n_waves`` via the
    real emit): the global + register + LDS stages show the band loaded by WAVE ``wave`` (default 0),
    re-keyed lane-local so it renders as a clean 64-lane wave. ``scope='wg'`` instead makes the register stage
    a per-wave :class:`WaveStrip` (all waves). ``cooperative=False`` = a per-wave load (single-wave
    descriptors; ``wave`` ignored). ``row_coord`` orients the global logical tile (0 -> dim0 rows; 1 -> dim1
    rows, the conventional B=K×N view). ``macro_note`` labels macro-tile membership. Returns a
    :class:`Pipeline`."""
    tag = f"  [{macro_note}]" if macro_note else ""
    if scope == "wg":
        # WG / MACRO overview: the SAME components as the wave view, each with scope="wg" so it collapses to
        # ONE block per WAVE (hue = wave; no inner grid / anchor). Fed the FULL cooperative map, so wave
        # ownership AND ORDER come from the real emit -- never assumed. global = macro logical tile (wave
        # sub-tiles) -> registers = ONE register file with tid grown to all n_waves*wave_size lanes, banded
        # by wave -> LDS = depth x banks coloured by the writing wave. Axes/orientation are untouched.
        lf = coop_forward_map(load_desc, n_waves=n_waves, wave_size=wave_size)
        owner = {coord: lr for lr, coord in lf.items()}      # coord -> (tid, reg): the cooperative load is 1:1
        stages = [FlowStage(f"global memory{tag}",
                            LogicalTileComponent(owner_map=owner, dims=dims, row_coord=row_coord,
                                                 label_coords="logical", scope="wg",
                                                 lanes_per_wave=wave_size, font_size=font_size)),
                  FlowStage("registers (wave bands)",
                            RegisterFileComponent(fwd_map=lf, dims=dims, scope="wg",
                                                  lanes_per_wave=wave_size, font_size=font_size),
                            transform="global load")]
        if dest == "lds" and store_desc is not None:
            _, af = lds_inputs(store_desc, stride=stride, pad=pad, swizzle=swizzle)
            sf = coop_forward_map(store_desc, n_waves=n_waves, wave_size=wave_size)
            stages.append(FlowStage("LDS (depth x banks, colour = wave)",
                                    LdsBankView(mp=sf, addr_fn=af, flow_map=lf, nbanks=nbanks, dims=dims,
                                                scope="wg", lanes_per_wave=wave_size, font_size=font_size),
                                    transform="LDS store"))
        return Pipeline(stages=tuple(stages), title=title or "WG wave-tile flow")

    def _wave_local(full):                                   # wave's 64 lanes, re-keyed to lane-local 0..63
        lo = wave * wave_size
        return {(tid - lo, reg): coord for (tid, reg), coord in full.items() if lo <= tid < lo + wave_size}

    if cooperative:
        load_sel = _wave_local(coop_forward_map(load_desc, n_waves=n_waves, wave_size=wave_size))
    else:
        _enc, load_sel = field_inputs(load_desc)
    owner0 = {c: s for s, c in load_sel.items()}
    mem_addr = _free_contig_addr(load_sel)                   # free axis stride-1: one vector = one shade
    shade = transactions(load_sel, mem_addr)[0]
    stages = [FlowStage(f"global tile{tag}",
                        LogicalTileComponent(owner_map=owner0, dims=dims, row_coord=row_coord,
                                             mode="thread_tile", label_coords="logical", addr_fn=mem_addr,
                                             detail_first=True, font_size=font_size)),
              FlowStage(f"registers (wave {wave})", _reg_file(load_sel, dims, font_size, shade_map=shade),
                        transform="global load")]
    if dest == "lds":
        _, af = lds_inputs(store_desc, stride=stride, pad=pad, swizzle=swizzle)
        if cooperative:
            # 1 WAVE detailed, rest grouped: a wide macro store occupies few of many physical LDS depths,
            # so show wave ``wave``'s slice with empty depth rows dropped (``compact_rows``) -> compact.
            store_mp = _wave_local(coop_forward_map(store_desc, n_waves=n_waves, wave_size=wave_size))
            lds = LdsBankView(mp=store_mp, addr_fn=af, flow_map=load_sel, nbanks=nbanks, dims=dims,
                              font_size=font_size, compact_rows=True)
            lname = f"LDS (wave {wave} of {n_waves}, +{n_waves - 1} identical)"
        else:
            store_mp = as_forward_map(_as_encoding(store_desc))
            lds = LdsBankView(mp=store_mp, addr_fn=af, flow_map=load_sel, nbanks=nbanks, dims=dims,
                              font_size=font_size)
            lname = "LDS (depth x banks)"
        stages.append(FlowStage(lname, lds, transform="LDS store"))
    else:                                                    # register-prefetch target (no LDS)
        rd_enc = field_inputs(reg_desc or load_desc)[0]
        stages.append(FlowStage("prefetch registers", _reg_file(rd_enc, dims, font_size),
                                transform="prefetch"))
    return Pipeline(stages=tuple(stages), title=title or f"load phase ({scope})")


def _plain(comp):
    """Label-suppressed overview: one border-only ``plain`` group over all cells (colour stays per-lane, so
    a thread's cells still trace across, but the bank grid / register block stays legible without hundreds
    of tiny labels). Used by the placement recipes."""
    comp.groups = (CellGroup(frozenset(comp._cells()), "plain", ""),)
    return comp


def _placement_pipeline(*, access_desc, flow_desc, dims, wave, n_waves, wave_size, cooperative, stride,
                        pad, swizzle, nbanks, compact, show_registers, direction, title):
    """Shared machinery for the two LDS-placement recipes: the wave's LDS bank grid (where each element
    sits, ``bank = addr mod nbanks``) + its register file, composed as a common-height 2-panel row.
    ``direction='store'`` -> registers -> LDS; ``'load'`` -> LDS -> registers (reversed). ``cooperative``
    slices one wave out of a macro (multi-wave) descriptor; else the descriptor is already a single wave."""
    if cooperative:
        lo = wave * wave_size
        wl = lambda full: {(t - lo, r): v for (t, r), v in full.items() if lo <= t < lo + wave_size}
        acc_mp = wl(coop_forward_map(access_desc, n_waves=n_waves, wave_size=wave_size))
        flow_mp = wl(coop_forward_map(flow_desc or access_desc, n_waves=n_waves, wave_size=wave_size))
    else:
        acc_mp = as_forward_map(_as_encoding(access_desc))
        flow_mp = as_forward_map(_as_encoding(flow_desc or access_desc))
    _, af = lds_inputs(access_desc, stride=stride, pad=pad, swizzle=swizzle)
    lds = _plain(LdsBankView(mp=acc_mp, addr_fn=af, flow_map=flow_mp, nbanks=nbanks, dims=dims,
                             compact_rows=compact, color_mode="full"))
    reg = _plain(RegisterFileComponent(fwd_map=flow_mp, dims=dims, color_mode="full"))
    ln = f"LDS placement (bank = addr mod {nbanks})"
    rn = f"wave {wave} registers (tid x vreg)"
    if direction == "store":
        stages = [FlowStage(rn, reg)] if show_registers else []
        stages.append(FlowStage(ln, lds, transform="LDS store" if show_registers else ""))
        default_title = f"LDS store placement (wave {wave})"
    else:
        stages = [FlowStage(ln, lds)]
        if show_registers:
            stages.append(FlowStage(rn, reg, transform="LDS read"))
        default_title = f"LDS load placement (wave {wave})"
    return Pipeline(stages=tuple(stages), title=title or default_title)


def flow_lds_store_placement(*, store_desc, load_desc=None, dims, wave=0, n_waves=1, wave_size=64,
                             cooperative=True, stride=None, pad=0, swizzle=None, nbanks=32, compact=False,
                             show_registers=True, title=""):
    """LDS STORE PLACEMENT for one wave: **registers -> LDS** -- WHERE wave ``wave``'s elements physically
    land (depth x banks, ``bank = addr mod nbanks``), coloured by lane so a thread's elements trace across.
    Pure PLACEMENT -- NO conflict verdict (that is the measured ``/bank-conflict`` path). ``compact=False``
    (default) keeps the TRUE physical depths (the store stride shows as gaps -- the rows other waves fill);
    ``compact=True`` drops empty rows (denser, but hides the stride). ``load_desc`` supplies the logical
    labels (default: ``store_desc``). Returns a :class:`Pipeline` (render via ``.render_panels(...)``)."""
    return _placement_pipeline(access_desc=store_desc, flow_desc=load_desc, dims=dims, wave=wave,
                               n_waves=n_waves, wave_size=wave_size, cooperative=cooperative, stride=stride,
                               pad=pad, swizzle=swizzle, nbanks=nbanks, compact=compact,
                               show_registers=show_registers, direction="store", title=title)


def flow_lds_load_placement(*, read_desc, flow_desc=None, dims, wave=0, n_waves=1, wave_size=64,
                            cooperative=True, stride=None, pad=0, swizzle=None, nbanks=32, compact=False,
                            show_registers=True, title=""):
    """LDS LOAD PLACEMENT for one wave -- the REVERSE of the store: **LDS -> registers**, i.e. which
    ``(depth, bank)`` each reading ``(lane, reg)`` pulls from. Same LDS geometry as the store; ``read_desc``
    is the LDS-read distribution -- pass the store descriptor for the exact round-trip inverse, or a distinct
    MMA-read descriptor for the K-vectorized read. Pure placement -- NO conflict verdict. Returns a
    :class:`Pipeline` (LDS panel -> register panel)."""
    return _placement_pipeline(access_desc=read_desc, flow_desc=flow_desc, dims=dims, wave=wave,
                               n_waves=n_waves, wave_size=wave_size, cooperative=cooperative, stride=stride,
                               pad=pad, swizzle=swizzle, nbanks=nbanks, compact=compact,
                               show_registers=show_registers, direction="load", title=title)


def flow_mma_phase(mma, *, a_enc=None, b_enc=None, dims_a=("M", "K"), dims_b=("N", "K"),
                   trace_a=None, trace_b=None, **tee_overrides):
    """MMA phase: the wave operands (read from LDS into registers) feeding the machine -> derived C. This IS
    the :class:`MmaTee` -- A/B register-file wings -> C body + C register file -- the canonical MMA view.
    Pass ``a_enc``/``b_enc`` to feed the ACTUAL (e.g. interleaved) operands (default: canonical atom). To
    show the preceding LDS->register read, compose ``flow_lds_to_register`` before this. Returns an
    ``MmaTee`` (render via ``.render(out_dir, name=...)``)."""
    ov = dict(tee_overrides)
    if a_enc is not None:
        ov["a_enc"] = a_enc
    if b_enc is not None:
        ov["b_enc"] = b_enc
    ov.setdefault("show_static", True)
    return MmaTee.from_mma(mma, dims_a=dims_a, dims_b=dims_b, trace_a=trace_a, trace_b=trace_b, **ov)


def flow_epilogue_phase(mma, *, c_store_desc=None, dims_c=("M", "N"), a_enc=None, b_enc=None,
                        stride=None, pad=0, swizzle=None, nbanks=32, font_size=6.0, title=""):
    """EPILOGUE phase: **C register file (native accumulator)** -> **{branch}** -> **final logical C tile**.
    The branch is AUTO-DETECTED by :func:`classify_epilogue`: **direct** store (identity), a **register
    shuffle** (in-register reorder), or an **LDS round-trip** (cross-lane). Returns
    ``(Pipeline | None, branch, note)``; ``branch=='unknown'`` (no/ambiguous store) returns ``None`` so the
    caller ASKS the user which epilogue applies. Set ``stride`` to include the LDS bank view on the
    cross-lane branch."""
    tee = MmaTee.from_mma(mma, **({"a_enc": a_enc} if a_enc is not None else {}),
                          **({"b_enc": b_enc} if b_enc is not None else {}))
    c_native = tee.c_mapping()                               # {(lane,reg)->(m,n)} machine accumulator
    branch, note = classify_epilogue(c_native, c_store_desc)
    if branch == "unknown":
        return None, branch, note
    stages = [FlowStage("C registers (native)", _reg_file(c_native, dims_c, font_size))]
    store_fm = as_forward_map(_as_encoding(c_store_desc))
    if branch == "cross_lane" and stride is not None:
        sm, af = lds_inputs(c_store_desc, stride=stride, pad=pad, swizzle=swizzle)
        stages.append(FlowStage("LDS round-trip",
                                LdsBankView(mp=sm, addr_fn=af, flow_map=store_fm, nbanks=nbanks, dims=dims_c,
                                            font_size=font_size),
                                transform="C-shuffle via LDS [cross-lane]"))
        stages.append(FlowStage("C registers (shuffled)", _reg_file(store_fm, dims_c, font_size),
                                transform="LDS read"))
    elif branch == "reorder":
        stages.append(FlowStage("C registers (shuffled)", _reg_file(store_fm, dims_c, font_size),
                                transform="C-shuffle [in-register reorder]"))
    store_owner = {coord: slot for slot, coord in store_fm.items()}
    stages.append(FlowStage("final C tile (stored)",
                            LogicalTileComponent(owner_map=store_owner, dims=dims_c, row_coord=0,
                                                 label_coords="logical", font_size=font_size),
                            transform="global store"))
    pipe = Pipeline(stages=tuple(stages), title=title or f"epilogue phase ({branch})")
    return pipe, branch, note
