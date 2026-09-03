# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor -> viz adapter + PHASE recipes: turn rocKE tile descriptors into the right BASIC views for
each part of a kernel pipeline, rendered as short LINEAR strips on the tested :class:`Pipeline` spine.

Two layers, both kernel-agnostic (a specific kernel -- e.g. the CRC GEMM -- supplies its own descriptors):

- **tier-1 adapter** -- ``field_inputs``/``lds_inputs``/``coop_forward_map`` convert a ``TileDesc`` (or a raw
  ``WarpDistributionEncoding``) + LDS geometry into the exact inputs the cell-field views consume: an
  encoding, a ``{(lane,reg)->coord}`` forward map, an LDS ``addr_fn``, and the FULL cooperative (multi-wave)
  map for macro-scope views.
- **phase recipes** -- ``flow_load_phase`` / ``flow_mma_phase`` / ``flow_epilogue_phase`` pick the right
  view sequence for the load/prefetch, MMA, and epilogue phases and return a compact linear ``Pipeline``
  (or, for the MMA, the ``MmaTee``). ``scope`` is a DETAIL knob on the SAME components (axes/orientation
  unchanged): ``"wave"`` (default) shows one wave -- one atom detailed, the rest grouped; ``"macro"`` shows the
  whole macro tile by feeding each component the FULL cooperative map + ``scope="macro"`` so it collapses to one
  block per WAVE (hue = wave). The register file simply lets its tid axis grow to all lanes (banded by wave)
  and the figure auto-scales -- no bespoke per-wave maps. Epilogue branch (direct / register-shuffle / LDS
  round-trip) is auto-detected via ``classify_epilogue``.

Everything is REUSED, not re-derived: ``as_forward_map`` normalizes the layout, ``classify_transform`` prices
each hop, the LDS swizzle is replayed bit-for-bit through ``lds_conflict.NumBuilder``, and the macro tile is
enumerated via the real ``emit_tensor_coordinates``."""
from __future__ import annotations

from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.transforms import (as_forward_map, classify_transform, describe_edge,
                                             reorder_between)
from rocke.helpers.tiling.lds_conflict import NumBuilder
from rocke.helpers.tiling.visualization.layout_render import (
    CellGroup, FlowStage, LdsBankView, LogicalTileComponent, MmaTee, Pipeline, RegisterFileComponent,
    transactions, vector_transactions,
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


def coop_forward_map(desc, *, n_waves, wave_size):
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
def _reg_file(enc_or_fm, dims, font_size, dtype_bits, shade_map=None):
    """A register file in the NORMAL orientation (tid rows / reg cols) from an encoding OR a forward map.
    (Only the MMA *tee* transposes its B/C wings; standalone register views stay tid-vertical.) ``dtype_bits``
    is REQUIRED (element width -- the file packs 2 f16 per 32-bit VGPR). ``shade_map`` (``{(lane,reg)->step}``)
    colours by vectorization TIME so one contiguous vector = one shade."""
    kw = dict(dims=dims, col_ticks_side="top", font_size=font_size, shade_map=shade_map, dtype_bits=dtype_bits)
    return (RegisterFileComponent(fwd_map=enc_or_fm, **kw) if isinstance(enc_or_fm, dict)
            else RegisterFileComponent(dist=enc_or_fm, **kw))


def _free_contig_addr(fwd_map):
    """A memory-order ``addr_fn`` for a FREE-DIM-CONTIGUOUS load: dim0 (the free axis, e.g. M for A, N for B)
    is stride-1, dim1 (K) strides past it. A thread's contiguous free run then reads as ONE address run ->
    ONE vectorization shade (the column-major reality), instead of one shade per register."""
    ext0 = max((c[0] for c in fwd_map.values()), default=0) + 1
    return lambda c0, c1: c0 + c1 * ext0


def _flow_edge(verb, src_enc, *, src_dims, tgt_enc=None, to_space=None):
    """Label a flow arrow by ROUTING THROUGH the classifier (never a hardcoded string), classifying by the
    SRC panel's distribution vs the DST panel's: give ``tgt_enc`` for a register-distribution destination, or
    ``to_space`` for a memory space. Same distribution -> ``identity`` (a DIRECT TRANSFER, e.g. global memory
    -> registers -- no reposition). A store into a memory space -> ``reposition`` (the datum moves register ->
    memory address); the DATA LABEL flows INVARIANT -- the store never renames the datum's axes. What changes
    is the physical STORAGE-axis alignment, which is NOT a label transpose (POSITION != LABEL): a real
    register realignment is a ``reorder``/``cross_lane`` classified by ``classify_transform``, never a
    label-space transpose stamped here. Returns ``verb -- kind\\n[why]``."""
    kind, why = describe_edge(src_enc, tgt_enc, src_dims=src_dims, to_space=to_space)
    if kind == "identity":
        return verb                                          # a direct transfer -- the verb alone says it all
    return f"{verb} -- {kind}\n[{why}]"


def flow_load_phase(*, load_desc, dims, nbanks, elem_bytes, cooperative, n_waves, wave_size,
                    lds_base_bytes=0, store_desc=None, dest="lds", scope="wave", wave=0, wave_grid=None,
                    row_coord=0, macro_note="", stride=None, load_strides=None, pad=0, swizzle=None,
                    reg_desc=None, font_size=6.0, title=""):
    """LOAD / PREFETCH phase (linear strip): **global thread-tile** -> **register file (after load)** ->
    **destination** (LDS banks, or a register-prefetch file). ``cooperative=True`` treats
    ``load_desc``/``store_desc`` as MACRO cooperative descriptors (enumerated across all ``n_waves`` via the
    real emit): the global + register + LDS stages show the band loaded by WAVE ``wave`` (default 0),
    re-keyed lane-local so it renders as a clean 64-lane wave. ``scope='macro'`` instead makes the register stage
    a per-wave :class:`WaveStrip` (all waves). ``cooperative=False`` = a per-wave load (single-wave
    descriptors; ``wave`` ignored). ``row_coord`` orients the global logical tile (0 -> dim0 rows; 1 -> dim1
    rows, the conventional B=K×N view). ``macro_note`` labels macro-tile membership. Returns a
    :class:`Pipeline`."""
    tag = f"  [{macro_note}]" if macro_note else ""
    dtype_bits = elem_bytes * 8                               # element width -> the register file packs by it
    if scope == "macro":
        # macro / MACRO overview: the SAME components as the wave view, each with scope="macro" so it collapses to
        # ONE block per WAVE (hue = wave; no inner grid / anchor). Fed the FULL cooperative map, so wave
        # ownership AND ORDER come from the real emit -- never assumed. global = macro logical tile (wave
        # sub-tiles) -> registers = ONE register file with tid grown to all n_waves*wave_size lanes, banded
        # by wave -> LDS = depth x banks coloured by the writing wave. Axes/orientation are untouched.
        lf = coop_forward_map(load_desc, n_waves=n_waves, wave_size=wave_size)
        owner = {coord: lr for lr, coord in lf.items()}      # coord -> (tid, reg): the cooperative load is 1:1
        load_enc = _as_encoding(load_desc)                   # stages render from forward maps, but each still
        store_enc = _as_encoding(store_desc) if store_desc is not None else None  # knows its OWN static dist
        stages = [FlowStage(f"global memory{tag}",
                            LogicalTileComponent(owner_map=owner, dims=dims, row_coord=row_coord,
                                                 label_coords="logical", scope="macro",
                                                 lanes_per_wave=wave_size, font_size=font_size),
                            source="load_desc", dist=load_enc),
                  FlowStage("registers (wave bands)",
                            RegisterFileComponent(fwd_map=lf, dims=dims, scope="macro",
                                                  lanes_per_wave=wave_size, dtype_bits=dtype_bits,
                                                  font_size=font_size),
                            source="load_desc",
                            transform=_flow_edge("global load", load_enc, src_dims=dims, tgt_enc=load_enc),
                            dist=load_enc)]
        if dest == "lds" and store_desc is not None:
            _, af = lds_inputs(store_desc, stride=stride, pad=pad, swizzle=swizzle)
            sf = coop_forward_map(store_desc, n_waves=n_waves, wave_size=wave_size)
            stages.append(FlowStage("LDS (depth x banks, colour = wave)",
                                    LdsBankView(mp=sf, addr_fn=af, flow_map=lf, nbanks=nbanks, dims=dims,
                                                scope="macro", lanes_per_wave=wave_size, elem_bytes=elem_bytes,
                                                lds_base_bytes=lds_base_bytes, font_size=font_size),
                                    source="store_desc",
                                    transform=_flow_edge("LDS store", load_enc, src_dims=dims, to_space="lds"),
                                    dist=store_enc))
        return Pipeline(stages=tuple(stages), title=title or "macro wave-tile flow")

    def _wave_local(full):                                   # wave's 64 lanes, re-keyed to lane-local 0..63
        lo = wave * wave_size
        return {(tid - lo, reg): coord for (tid, reg), coord in full.items() if lo <= tid < lo + wave_size}

    if cooperative:
        load_sel = _wave_local(coop_forward_map(load_desc, n_waves=n_waves, wave_size=wave_size))
    else:
        _enc, load_sel = field_inputs(load_desc)
    load_enc = _as_encoding(load_desc)                       # static distribution behind this wave's slice
    owner0 = {c: s for s, c in load_sel.items()}
    # Memory order from the RECORDED global-load descriptor STRIDES (never assumed): addr = c0*s0 + c1*s1, so the
    # stride-1 axis (whichever the descriptor says) forms the contiguous run. `_free_contig_addr` is only a
    # descriptor-only fallback (a demo with no recorded transaction). SHADE = `vector_transactions` (§9.6): one
    # shade per vectorized access, width from these strides, CAPPED at b128 -- ✗ never `transactions` (uncapped).
    mem_addr = ((lambda c0, c1, s=tuple(load_strides): c0 * s[0] + c1 * s[1]) if load_strides is not None
                else _free_contig_addr(load_sel))
    shade = vector_transactions(load_sel, mem_addr, elem_bytes * 8)[0]     # b128-capped, stride-driven
    tile_shade = {coord: shade[slot] for coord, slot in owner0.items()}    # same shade, keyed by (c0,c1)
    stages = [FlowStage(f"global tile{tag}",
                        LogicalTileComponent(owner_map=owner0, dims=dims, row_coord=row_coord,
                                             mode="thread_tile", label_coords="logical", addr_fn=mem_addr,
                                             dtype_bits=elem_bytes * 8, shade_map=tile_shade,
                                             detail_first=True, font_size=font_size),
                        source="load_desc", dist=load_enc),
              FlowStage(f"registers (wave {wave})",
                        _reg_file(load_sel, dims, font_size, dtype_bits, shade_map=shade),
                        source="load_desc",
                        transform=_flow_edge("global load", load_enc, src_dims=dims, tgt_enc=load_enc),
                        dist=load_enc)]
    if dest == "lds":
        _, af = lds_inputs(store_desc, stride=stride, pad=pad, swizzle=swizzle)
        store_enc = _as_encoding(store_desc)
        if cooperative:
            # Show this wave's store at its TRUE physical (depth, bank) -- NO row compaction, so the real
            # striping (the gaps where the OTHER waves' rows sit) is visible, exactly as in macro scope.
            store_mp = _wave_local(coop_forward_map(store_desc, n_waves=n_waves, wave_size=wave_size))
            # true-depth store: TALL + SPARSE. A single wave occupies only free/2 banks (a NARROW strip), so
            # the per-row block label must fit that width AND the thin strip height -- keep the base font (a
            # larger one overflows the strip borders); rely on the tall panel + stride gaps for legibility.
            lds = LdsBankView(mp=store_mp, addr_fn=af, flow_map=load_sel, nbanks=nbanks, dims=dims,
                              elem_bytes=elem_bytes, lds_base_bytes=lds_base_bytes,
                              font_size=font_size, compact_rows=False)
            lname = f"LDS (wave {wave} of {n_waves}, +{n_waves - 1} identical)"
        else:
            store_mp = as_forward_map(_as_encoding(store_desc))
            lds = LdsBankView(mp=store_mp, addr_fn=af, flow_map=load_sel, nbanks=nbanks, dims=dims,
                              elem_bytes=elem_bytes, lds_base_bytes=lds_base_bytes, font_size=font_size)
            lname = "LDS (depth x banks)"
        stages.append(FlowStage(lname, lds, source="store_desc",
                                transform=_flow_edge("LDS store", load_enc, src_dims=dims, to_space="lds"),
                                dist=store_enc))
    else:                                                    # register-prefetch target (no LDS)
        rd_enc = field_inputs(reg_desc or load_desc)[0]
        stages.append(FlowStage("prefetch registers", _reg_file(rd_enc, dims, font_size, dtype_bits),
                                source="reg_desc", transform="prefetch"))
    return Pipeline(stages=tuple(stages), title=title or f"load phase ({scope})")


def _plain(comp):
    """Label-suppressed overview: one border-only ``plain`` group over all cells (colour stays per-lane, so
    a thread's cells still trace across, but the bank grid / register block stays legible without hundreds
    of tiny labels). Used by the placement recipes."""
    comp.groups = (CellGroup(frozenset(comp._cells()), "plain", ""),)
    return comp


def _placement_pipeline(*, access_desc, flow_desc, dims, wave, n_waves, wave_size, cooperative, stride,
                        pad, swizzle, nbanks, elem_bytes, lds_base_bytes, compact, show_registers, direction,
                        title, reg_source, lds_source, addr_desc=None, addr_stride=None, addr_swizzle=None):
    """Shared machinery for the two LDS-placement recipes: the wave's LDS bank grid (where each element
    sits, ``bank = dword mod nbanks`` with ``4//elem_bytes`` elems packed per bank) + its register file,
    composed as a common-height 2-panel row. ``direction='store'`` -> registers -> LDS; ``'load'`` -> LDS ->
    registers (reversed). ``cooperative`` slices one wave out of a macro (multi-wave) descriptor; else the
    descriptor is already a single wave."""
    if cooperative:
        lo = wave * wave_size
        wl = lambda full: {(t - lo, r): v for (t, r), v in full.items() if lo <= t < lo + wave_size}
        acc_mp = wl(coop_forward_map(access_desc, n_waves=n_waves, wave_size=wave_size))
        flow_mp = wl(coop_forward_map(flow_desc or access_desc, n_waves=n_waves, wave_size=wave_size))
    else:
        acc_mp = as_forward_map(_as_encoding(access_desc))
        flow_mp = as_forward_map(_as_encoding(flow_desc or access_desc))
    _, af_access = lds_inputs(access_desc, stride=stride, pad=pad, swizzle=swizzle)
    if addr_desc is not None:
        # STORE AUTHORITY: the LDS layout was PHYSICALLY written by the store, so the store's map is the truth
        # for which bank a datum sits in. The READ descriptor's frame is TRANSPOSED vs that layout (K vs the
        # free dim swapped), so addressing the LDS panel / shade by the read's OWN map scrambles the banks and
        # splits a real contiguous run. Address every datum by the STORE's map; cap the shade at the READ's own
        # real contiguous run (its true load width, e.g. 4 f16 = ds_read2_b32 = b64, not the b128 ceiling).
        _, af = lds_inputs(addr_desc, stride=addr_stride if addr_stride is not None else stride, pad=pad,
                           swizzle=addr_swizzle)
        rd_ts = transactions(as_forward_map(_as_encoding(access_desc)), af_access)[0]
        l0 = min(l for l, _ in rd_ts)
        counts = {}
        for (l, _r), t in rd_ts.items():
            if l == l0:
                counts[t] = counts.get(t, 0) + 1
        # the read's real contiguous run, CAPPED at b128 (a run longer than one ds_read is hw-split; §9.6).
        max_bits = min(128, max(counts.values()) * elem_bytes * 8)
    else:
        af, max_bits = af_access, 128
    store_enc = _as_encoding(access_desc)                    # static dist behind the LDS panel
    reg_enc = _as_encoding(flow_desc or access_desc)         # static dist behind the register panel
    # Carry the FLOWED logical labels through (NOT label-suppressed): the block/grouped summaries NAME which
    # datum lands where, so INPUT logical data traces across this hop; `dist` gives each panel its static
    # distribution (Rs/Hs/Ps/Ys) in the info box instead of a bare "(no encoding)".
    lds = LdsBankView(mp=acc_mp, addr_fn=af, flow_map=flow_mp, nbanks=nbanks, elem_bytes=elem_bytes,
                      lds_base_bytes=lds_base_bytes, dims=dims, compact_rows=compact, color_mode="first8")
    # SHADE = the MECHANICS: the mem<->register TRANSACTION time order of THIS hop's OWN descriptor
    # (``acc_mp`` = the access/store or read distribution -- NOT the operand ``flow_mp``, which is a DIFFERENT
    # distribution: prefetch-store macro tile vs wave-read compute tile). Width from the STRIDES: the widest
    # LEGAL, ALIGNED access <= b128 over each stride-1 run (``vector_transactions``); a wave read whose real run
    # is 4 f16 before the M-stride jump is b64 (2 banks), matching the LDS panel + objdump, NOT the recorded emit
    # ``vw``. LABELS stay on ``flow_mp`` (WHERE each datum flows to), paired to ``acc_mp`` on the same physical
    # (lane,reg) key -- exactly as the LDS panel pairs them. The vreg axis is ordered by MEMORY ADDRESS (the
    # memory<->vreg relationship) so each transaction is a coherent adjacent block, not scattered reg indices.
    reg_shade = vector_transactions(acc_mp, af, elem_bytes * 8, max_bits=max_bits)[0]
    # RE-KEY registers into PHYSICAL (memory-address) order, index p = 0,1,2,... A strided read fills
    # NON-adjacent ENCODING reg indices (a b64 read -> r0,r8,r16,r24), so the encoding index is not the physical
    # vreg order; MEMORY order is. Re-keying to p keeps each transaction a coherent adjacent block AND keeps the
    # 32-bit-vreg tick ruler correct (p -> VGPR p//pack); reordering the encoding indices instead scrambled the
    # ruler. Label (flow_mp = WHERE it flows) + shade travel with the datum to its physical slot p.
    lanes = {l for l, _ in acc_mp}
    order = {l: sorted((r for (ll, r) in acc_mp if ll == l), key=lambda r, l=l: af(*acc_mp[(l, r)]))
             for l in lanes}
    phys_fwd = {(l, p): flow_mp[(l, order[l][p])] for l in lanes for p in range(len(order[l]))}
    phys_shade = {(l, p): reg_shade[(l, order[l][p])] for l in lanes for p in range(len(order[l]))}
    reg = RegisterFileComponent(fwd_map=phys_fwd, dims=dims, color_mode="first8", dtype_bits=elem_bytes * 8,
                                shade_map=phys_shade)
    ln = f"LDS ({dims[0]}-contig, bank = dword mod {nbanks})"
    rn = f"wave {wave} registers (tid x vreg)"
    if direction == "store":                                 # the LDS STORE hop: registers -> LDS
        stages = [FlowStage(rn, reg, source=reg_source, dist=reg_enc)] if show_registers else []
        stages.append(FlowStage(ln, lds, source=lds_source,
                                transform="LDS store" if show_registers else "", dist=store_enc))
        default_title = f"LDS store: registers -> LDS (wave {wave})"
    else:                                                    # the LDS READ hop: LDS -> registers
        stages = [FlowStage(ln, lds, source=lds_source, dist=store_enc)]
        if show_registers:
            # The register file the coalesced read LANDS (``phys_fwd`` = memory-address order). IFF that
            # landing order differs from the REQUESTED consumer order (``flow_mp``) by a within-lane register
            # reorder, the read is TWO hops: the coalesced landing, then a ``v_perm`` reorder into the requested
            # order (the honest picture -- the ``ds_read`` does NOT deliver MMA-ready registers). ``reorder_between``
            # DERIVES the reorder (never hardcoded); ``pack`` = elems/32-bit-reg decides dword vs sub-dword.
            rp = reorder_between(phys_fwd, flow_mp, pack=max(1, 32 // (elem_bytes * 8)))
            if rp is not None and rp.tier != "cross_lane":   # a within-lane reorder -> land, then reorder
                stages.append(FlowStage(rn, reg, source=reg_source, transform="LDS read", dist=reg_enc,
                                        reorder=True))       # coalesced landing: no box (box on the requested)
                # SHADE = FLAT (hue = lane%8, single shade): this panel is the RESULT of an in-register
                # reorder, NOT a memory transaction -- there is no vectorized-access time order to tint by
                # (a per-row shade gradient would be meaningless). One shade per lane.
                requested = RegisterFileComponent(fwd_map=flow_mp, dims=dims, color_mode="first8",
                                                  dtype_bits=elem_bytes * 8,
                                                  shade_map={k: 0 for k in flow_mp})
                stages.append(FlowStage(f"wave {wave} registers (requested, tid x vreg)", requested,
                                        source=reg_source, dist=reg_enc, info=(rp.cost,),
                                        transform=f"in-register reorder\n{rp.label}"))
            else:                                            # identity / cross_lane -> single register panel
                stages.append(FlowStage(rn, reg, source=reg_source, transform="LDS read", dist=reg_enc))
        default_title = f"LDS read: LDS -> registers (wave {wave})"
    return Pipeline(stages=tuple(stages), title=title or default_title)


def flow_lds_store_placement(*, store_desc, dims, nbanks, elem_bytes, n_waves, wave_size, cooperative,
                             lds_base_bytes=0, load_desc=None, wave=0, stride=None, pad=0,
                             swizzle=None, compact=False, show_registers=True, title=""):
    """LDS STORE PLACEMENT for one wave: **registers -> LDS** -- WHERE wave ``wave``'s elements physically
    land (depth x banks, ``bank = dword mod nbanks`` with ``4//elem_bytes`` elems packed per bank), coloured
    by lane so a thread's elements trace across. Pure PLACEMENT -- NO conflict verdict (that is the measured
    ``/bank-conflict`` path). ``nbanks`` + ``elem_bytes`` are REQUIRED (state arch banks + dtype width, e.g.
    gfx90a f16 -> 32, 2). ``compact=False`` (default) keeps the TRUE physical depths (the store stride shows
    as gaps -- the rows other waves fill); ``compact=True`` drops empty rows (denser, but hides the stride).
    ``load_desc`` supplies the logical labels (default: ``store_desc``). Returns a :class:`Pipeline`."""
    return _placement_pipeline(access_desc=store_desc, flow_desc=load_desc, dims=dims, wave=wave,
                               n_waves=n_waves, wave_size=wave_size, cooperative=cooperative, stride=stride,
                               pad=pad, swizzle=swizzle, nbanks=nbanks, elem_bytes=elem_bytes,
                               lds_base_bytes=lds_base_bytes, compact=compact,
                               show_registers=show_registers, direction="store", title=title,
                               lds_source="store_desc",
                               reg_source="load_desc" if load_desc is not None else "store_desc")


def flow_lds_load_placement(*, read_desc, dims, nbanks, elem_bytes, n_waves, wave_size, cooperative,
                            lds_base_bytes=0, flow_desc=None, wave=0, stride=None, pad=0,
                            swizzle=None, compact=False, show_registers=True, title="",
                            store_desc=None, store_stride=None, store_swizzle=None):
    """LDS LOAD PLACEMENT for one wave -- the REVERSE of the store: **LDS -> registers**, i.e. which
    ``(depth, bank)`` each reading ``(lane, reg)`` pulls from. Same LDS geometry as the store; ``read_desc``
    is the LDS-read distribution -- pass the store descriptor for the exact round-trip inverse, or a distinct
    MMA-read descriptor for the K-vectorized read. ``nbanks`` + ``elem_bytes`` are REQUIRED (state arch banks
    + dtype width). Pure placement -- NO conflict verdict.

    When the read frame is TRANSPOSED vs the true LDS layout (a distinct MMA-read desc), pass ``store_desc``
    (+ ``store_stride``/``store_swizzle``): the LDS banks + register shade are then addressed by the STORE
    (the layout authority) while the shade width is capped at the read's own contiguous run. Returns a
    :class:`Pipeline` (LDS -> register)."""
    return _placement_pipeline(access_desc=read_desc, flow_desc=flow_desc, dims=dims, wave=wave,
                               n_waves=n_waves, wave_size=wave_size, cooperative=cooperative, stride=stride,
                               pad=pad, swizzle=swizzle, nbanks=nbanks, elem_bytes=elem_bytes,
                               lds_base_bytes=lds_base_bytes, compact=compact,
                               show_registers=show_registers, direction="load", title=title,
                               lds_source="store_desc" if store_desc is not None else "read_desc",
                               reg_source="flow_desc" if flow_desc is not None else "read_desc",
                               addr_desc=store_desc, addr_stride=store_stride, addr_swizzle=store_swizzle)


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


def flow_epilogue_phase(mma, *, nbanks, elem_bytes, lds_base_bytes=0, c_store_desc=None, dims_c=("M", "N"),
                        a_enc=None, b_enc=None, stride=None, pad=0, swizzle=None, font_size=6.0, title=""):
    """EPILOGUE phase: **C register file (native accumulator)** -> **{branch}** -> **final logical C tile**.
    The branch is AUTO-DETECTED by :func:`classify_epilogue`: **direct** store (identity), a **register
    shuffle** (in-register reorder), or an **LDS round-trip** (cross-lane). Returns
    ``(Pipeline | None, branch, note)``; ``branch=='unknown'`` (no/ambiguous store) returns ``None`` so the
    caller ASKS the user which epilogue applies. Set ``stride`` to include the LDS bank view on the
    cross-lane branch. ``nbanks`` + ``elem_bytes`` are REQUIRED (the C accumulator is f32 -> elem_bytes=4)."""
    tee = MmaTee.from_mma(mma, **({"a_enc": a_enc} if a_enc is not None else {}),
                          **({"b_enc": b_enc} if b_enc is not None else {}))
    c_native = tee.c_mapping()                               # {(lane,reg)->(m,n)} machine accumulator
    branch, note = classify_epilogue(c_native, c_store_desc)
    if branch == "unknown":
        return None, branch, note
    stages = [FlowStage("C registers (native)", _reg_file(c_native, dims_c, font_size, elem_bytes * 8),
                        source="c_native")]
    store_fm = as_forward_map(_as_encoding(c_store_desc))
    if branch == "cross_lane" and stride is not None:
        sm, af = lds_inputs(c_store_desc, stride=stride, pad=pad, swizzle=swizzle)
        stages.append(FlowStage("LDS round-trip",
                                LdsBankView(mp=sm, addr_fn=af, flow_map=store_fm, nbanks=nbanks,
                                            elem_bytes=elem_bytes, lds_base_bytes=lds_base_bytes,
                                            dims=dims_c, font_size=font_size),
                                source="c_store_desc", transform="C-shuffle via LDS [cross-lane]"))
        stages.append(FlowStage("C registers (shuffled)", _reg_file(store_fm, dims_c, font_size, elem_bytes * 8),
                                source="c_store_desc", transform="LDS read"))
    elif branch == "reorder":
        stages.append(FlowStage("C registers (shuffled)", _reg_file(store_fm, dims_c, font_size, elem_bytes * 8),
                                source="c_store_desc", transform="C-shuffle [in-register reorder]"))
    store_owner = {coord: slot for slot, coord in store_fm.items()}
    stages.append(FlowStage("final C tile (stored)",
                            LogicalTileComponent(owner_map=store_owner, dims=dims_c, row_coord=0,
                                                 label_coords="logical", font_size=font_size),
                            source="c_store_desc", transform="global store"))
    pipe = Pipeline(stages=tuple(stages), title=title or f"epilogue phase ({branch})")
    return pipe, branch, note
