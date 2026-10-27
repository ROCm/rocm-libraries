# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite Solution -> theta. The only place TensileLite vocabulary appears."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd as _gcd

from . import traversal as geometry
from .loopir import DS_LOAD, REGISTER, SHARED, Axis, Effect, Expr
from .theta import (AgentAssignment, AgentAxisAssignment, Fragment, Global,
                    RegionLayout, Shared, Trajectory, Operand, Theta, group_labels)
from .traversal import readahead_level
from ..Components import TDMSplit as _tdm_split
from ..Components.LoopModel import Order
from ..Components.TDMFuse import liveGroups, tdmWavePartition, tdmWaveSeparated
WAVE32 = 32  # gfx1250 wave width
REG_BYTES = 4  # gfx1250 register width in bytes

def _per_operand(kernel, base, operand, preset):
    value = kernel.get("%s%s" % (base, operand))
    return preset if value is None or value < 0 else int(value)


def _k_tiles(kernel):
    mi_k = kernel["MatrixInstruction"][2]
    return max(1, kernel["DepthU"] // mi_k) * kernel["InnerUnroll"]


def _frag_elems(kernel, free_mi):
    mi_k = kernel["MatrixInstruction"][2]
    mi_b = kernel["MatrixInstruction"][3] if len(kernel["MatrixInstruction"]) > 3 else 1
    return max(1, free_mi * mi_k * mi_b // WAVE32)


def _acc_frag_elems(kernel):
    mi_b = kernel["MatrixInstruction"][3] if len(kernel["MatrixInstruction"]) > 3 else 1
    return max(1, kernel["MatrixInstruction"][0] * kernel["MatrixInstruction"][1] * mi_b // WAVE32)


def _mx_frag_elems(kernel, free_mi, mxblock):
    data = _frag_elems(kernel, free_mi)
    dup = max(1, WAVE32 // max(1, kernel["MatrixInstruction"][0]))
    return max(1, data // max(1, mxblock) * dup)


AXIS_NAMES = ["K_split", "K_inner",
              "M_split", "M_inner", "N_split", "N_inner"]

# --- build -----------------------------------------------------------------


DEFAULTS = {
    "DepthU": 128, "MatrixInstruction": [16, 16, 64, 1], "MIWaveTile": [1, 2],
    "LoopOrder": "KMN",
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "TDMInst": 3, "TDMSplit": [1, 1], "TDMSplitA": 0, "TDMSplitB": 0, "TDMIterateMode": 0,
    "WmmaFactorM": -1, "WmmaFactorN": -1, "WmmaFactorK": -1,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidth": 4,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "WaveSeparateGlobalReadA": 0, "WaveSeparateGlobalReadB": 0,
    "1LDSBuffer": 0, "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPad": 0,
    "InnerUnroll": 1, "ElemBytes": 2,
    "MXBlockA": 0, "MXBlockB": 0, "TDMFuse": 0,
    "NumWaves": 1,
    "PrefetchGlobalReadA": -1, "PrefetchGlobalReadB": -1,
    "PrefetchLocalReadA": -1, "PrefetchLocalReadB": -1,
    "ClusterLocalRead": 0, "ClusterLocalReadA": -1, "ClusterLocalReadB": -1,
    "VgprGroupA": -1, "VgprGroupB": -1, "VgprAllocA": -1, "VgprAllocB": -1,
    "VgprAllocExplicitA": False, "VgprAllocExplicitB": False,
    "LDSBufferA": -1, "LDSBufferB": -1,
    "LDSBufferMXSA": -1, "LDSBufferMXSB": -1,
    "ReadVectorElems": {},
    "ReadQuantum": {},
    "PerRegionCompletion": False,
}


@dataclass(frozen=True)
class Splits:
    """TDMSplit axis factors plus the independent physical wave-to-region span."""
    aMT: int; bMT: int; aDU: int; bDU: int
    mSplit: int; nSplit: int  # the M/N region extent = that operand's MT split
    kSplit: int; kResA: int; kResB: int  # shared K link (gcd), then each operand's residual
    waveRegionsA: int; waveRegionsB: int  # regions one wave's read addresses may occupy


@dataclass(frozen=True)
class Ord:
    """The loop nest this LoopOrder gives: the axes in order, and each operand's regions."""
    ord_: list
    M_BCAST: set; N_BCAST: set
    aRegions: tuple; bRegions: tuple
    mInner: int; nInner: int
    #: {operand: {region axis: axis values ONE of its regions spans}} -- 1 when the axis value
    #: is the region index, >1 when the shared K walk steps finer than that operand splits.
    regionSpan: dict = None
    #: {operand: the extra axes its walk steps} -- its regions plus the factors with no storage.
    factorAxes: dict = None


@dataclass(frozen=True)
class OperandFragment:
    """One operand's register fragment, the elements it holds, and the LDS ring behind it."""
    fragment: object
    frag_elems: int
    regions: int          # how many storage regions its copy writes
    lds_buffers: int


@dataclass(frozen=True)
class Fragments:
    """The fragment of each operand, by name, plus the builder the MX scale rings reuse."""
    by_operand: dict
    reg_fragment: object

    def __getitem__(self, name) -> OperandFragment:
        return self.by_operand[name]


@dataclass(frozen=True)
class MovementInputs:
    """Target movement widths, grouping, and explicit coverage."""
    grvwA: int; grvwB: int; lrvw: int
    read_elements: dict; instances_per_instruction: dict; read_coverage: dict; agent_group_sizes: dict


def _read_params(p):
    """The Solution params this decode reads, with the defaults applied."""
    kernel = dict(DEFAULTS); kernel.update(p)
    matrix_instruction = kernel["MatrixInstruction"]
    substeps = _k_tiles(kernel)  # reduction substeps per kiter (DU/MI_K x InnerUnroll)
    elem_bytes = kernel["ElemBytes"]
    fanM, fanN = kernel["MIWaveTile"][0], kernel["MIWaveTile"][1]
    copy_depth = max(0, kernel["PrefetchGlobalRead"])
    read_depth = max(0, kernel["PrefetchLocalRead"])

    return copy_depth, read_depth, elem_bytes, fanM, fanN, kernel, substeps, matrix_instruction


def _read_splits(kernel):
    """TDMSplit region extents: per-operand MT, shared DU."""
    split_factors = kernel["TDMSplit"]
    if isinstance(split_factors, bool):
        split_factors = [2, 2] if split_factors else [1, 1]
    split_factors = list(split_factors) + [1] * (4 - len(split_factors))
    aMT, bMT, aDU, bDU = (max(1, split_factors[0]), max(1, split_factors[1]), max(1, split_factors[2]), max(1, split_factors[3]))
    mSplit, nSplit = aMT, bMT  # M region (A's fan), N region (B's fan)
    _split_wave_regions = list(kernel.get("TDMSplitWaveRegions", []) or []) + [0, 0]
    waveRegionsA = max(1, min(aMT, int(_split_wave_regions[0]) or aMT))
    waveRegionsB = max(1, min(bMT, int(_split_wave_regions[1]) or bMT))

    kSplit = _gcd(aDU, bDU)  # the shared link: both operands prefix it
    kResA = aDU // max(1, kSplit)  # A's own residual beyond the shared link
    kResB = bDU // max(1, kSplit)  # B's own residual

    return Splits(aMT, bMT, aDU, bDU, mSplit, nSplit, kSplit, kResA, kResB,
                  waveRegionsA, waveRegionsB)


def _build_ord(fanM, fanN, kernel, substeps, splits):
    """The six-mode loop order and each operand's regions."""
    params = dict(kernel)
    params["LoopIters"] = substeps
    params["MIWaveTileA"] = fanM
    params["MIWaveTileB"] = fanN
    shape = Order.axes(params)
    axis_values = shape["axes"]

    extents = {name: 1 for name in AXIS_NAMES}
    extents.update({
        str(name): max(1, int(extent))
        for name, extent in axis_values
    })
    axes = {name: Axis(name, extents[name]) for name in AXIS_NAMES}
    iter_mode = Axis("iter", 0)
    axes["iter"] = iter_mode
    ord_ = [iter_mode] + [
        axes[str(name)] for name, _extent in axis_values
    ]
    m_inner = extents["M_inner"]
    n_inner = extents["N_inner"]

    N_BCAST = {"N_split", "N_inner"}  # A is broadcast over N modes
    M_BCAST = {"M_split", "M_inner"}  # B is broadcast over M modes
    aRegions = shape["regions"]["A"]
    bRegions = shape["regions"]["B"]
    span = shape["span"]

    return Ord(ord_, M_BCAST, N_BCAST, aRegions, bRegions, m_inner, n_inner, span,
               shape.get("factor_axes") or {})


def _read_movements(kernel):
    """The per-hop payload the target supplies: vector widths, the movement coverage, Phi."""
    # --- hops (per-hop delta = retime) ------------------------------------------------
    grvwA, grvwB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
    lrvw = kernel["LocalReadVectorWidth"]

    read_elements = dict(kernel.get("ReadVectorElems", {}) or {})

    read_coverage = dict(kernel.get("ReadQuantum", {}) or {})

    instances_per_instruction = dict(kernel.get("ReadPhi", {}) or {})
    agent_group_sizes = dict(kernel.get("ReadRho", {}) or {})

    return MovementInputs(grvwA, grvwB, lrvw, read_elements,
                          instances_per_instruction, read_coverage, agent_group_sizes)


def _build_agent_assignment(movements, nest):
    """Build agent-axis assignments and adapter movement inputs."""
    agent_inputs = {
        "A": "M_inner", "MXSA": "M_inner",
        "B": "N_inner", "MXSB": "N_inner",
    }
    _by_mode = {}

    def _assign(axis, level, extent, origin):
        prev = _by_mode.get(axis)
        if prev is not None:
            if (prev.level, prev.extent) != (level, int(extent)):
                raise RuntimeError(
                    "axis %r assigned twice with different answers -- %s/%d from %s vs "
                    "%s/%d from %s. An axis has ONE agent level."
                    % (axis, prev.level, prev.extent, prev.origin, level, extent, origin))
            return
        _by_mode[axis] = AgentAxisAssignment(
            axis=axis, level=level, extent=int(extent), origin=origin)

    for _name, _free in agent_inputs.items():
        _sub = int(movements.agent_group_sizes.get(_name, 0) or 0)
        if _sub > 1 and _free:  # sub-wave: the distributed carrier coverage
            _assign(_free, "subwave", _sub, (_name, DS_LOAD))
    assignment = AgentAssignment(
        axis_assignments=tuple(_by_mode[m] for m in sorted(_by_mode)))

    def _configure_read(fragment, name, default):
        merged = max(1, int(movements.instances_per_instruction.get(name, 1) or 1))
        _free = agent_inputs.get(name)
        agent_group_size = (
            assignment.group_size_over({_free} if _free else set(), level="subwave")
            if int(movements.agent_group_sizes.get(name, 0) or 0) > 1 else 0)
        fragment.vector_elements = int(movements.read_elements.get(name, default) or default)
        fragment.instances_per_instruction = merged
        fragment.agent_group_size = agent_group_size
        fragment.coverage = movements.read_coverage.get(name)
        if fragment.coverage is None:
            fragment.coverage = geometry.derive_transfer_coverage(fragment)
        return fragment

    return _configure_read, agent_inputs, assignment


def _build_fragments(kernel, matrix_instruction, nest, splits):
    """The operands: their paths, fragments, register widths and MX scale rings."""
    # --- register width: DERIVED, not a parameter
    def reg_fragment(bcast, grp_mode):
        return Fragment(broadcast_axes=set(bcast), parts=1, labels=group_labels(1),
                        grouping_mode=grp_mode, group_policy={"*": "pipeline"})

    fragA = reg_fragment(nest.N_BCAST, "M_inner")
    fragB = reg_fragment(nest.M_BCAST, "N_inner")

    fragElA = _frag_elems(kernel, matrix_instruction[0])
    fragElB = _frag_elems(kernel, matrix_instruction[1])
    splitA = splits.mSplit * splits.aDU
    splitB = splits.nSplit * splits.bDU

    numLdsBlk = kernel.get("NumLdsBlk", 2)
    ldsBufA = 0 if kernel["DirectToVgprA"] else _per_operand(kernel, "LDSBuffer", "A", numLdsBlk)
    ldsBufB = 0 if kernel["DirectToVgprB"] else _per_operand(kernel, "LDSBuffer", "B", numLdsBlk)
    ldsBufMXSA = _per_operand(kernel, "LDSBuffer", "MXSA", ldsBufA)
    ldsBufMXSB = _per_operand(kernel, "LDSBuffer", "MXSB", ldsBufB)
    for _name, _v in (("A", ldsBufA), ("B", ldsBufB),
                    ("MXSA", ldsBufMXSA), ("MXSB", ldsBufMXSB)):
        if _v and _v > numLdsBlk:
            raise RuntimeError(
                "LDSBuffer%s=%d exceeds the kernel's allocated NumLdsBlk=%d -- the LDS allocation "
                "is sized by Solution.py against MaxLDS, so a deeper ring is an out-of-bounds LDS "
                "write, not a tighter schedule.  Raise 1LDSBuffer/DepthU so the allocation grows, "
                "or keep LDSBuffer%s <= %d." % (_name, _v, numLdsBlk, _name, numLdsBlk))
    return Fragments(reg_fragment=reg_fragment, by_operand={
        "A":    OperandFragment(fragA, fragElA, splitA, ldsBufA),
        "B":    OperandFragment(fragB, fragElB, splitB, ldsBufB),
        "MXSA": OperandFragment(None, 0, splitA, ldsBufMXSA),
        "MXSB": OperandFragment(None, 0, splitB, ldsBufMXSB),
    })


def _build_operands(configure_read, elem_bytes, frags, movements, kernel, matrix_instruction,
                    nest, splits):
    """Build tensor identities and placement paths."""
    def region_layout(parent):
        is_a = parent == "A"
        return RegionLayout(
            axes=nest.aRegions if is_a else nest.bRegions,
            span=(nest.regionSpan or {}).get(parent, {}),
            free_split=max(1, splits.aMT if is_a else splits.bMT),
            wave_span=splits.waveRegionsA if is_a else splits.waveRegionsB,
            factor_axes=(nest.factorAxes or {}).get(parent, ()))

    def input_operand(name, free_mode, direct, copy_width, parent):
        spec = frags[name]
        fragment = spec.fragment
        fragment.fragment_elements = spec.frag_elems
        if direct:
            fragment.vector_elements = copy_width
            path = Trajectory(Global(), fragment)
        else:
            shared = Shared(vector_elements=copy_width, split=spec.regions,
                            ring_depth=spec.lds_buffers, regions=region_layout(parent))
            path = Trajectory(Global(), shared,
                              configure_read(fragment, name, movements.lrvw))
        return Operand(name, free_mode, trajectory=path, elem_bytes=elem_bytes)

    A = input_operand("A", "M_inner", kernel["DirectToVgprA"], movements.grvwA, "A")
    B = input_operand("B", "N_inner", kernel["DirectToVgprB"], movements.grvwB, "B")

    accumulator = Fragment(
        broadcast_axes={"K_split", "K_inner"},
        fragment_elements=_acc_frag_elems(kernel))
    C = Operand("C", "M_inner", trajectory=Trajectory(accumulator),
                elem_bytes=REG_BYTES, role="output")
    operands = [A, B, C]

    hasMXA, hasMXB = kernel["MXBlockA"] > 0, kernel["MXBlockB"] > 0

    def add_scale(name, free_mode, copy_width, broadcast, free_mi, block):
        fragment = frags.reg_fragment(broadcast, free_mode)
        fragment.fragment_elements = _mx_frag_elems(kernel, free_mi, block)
        # A SCALE IS NEVER SPLIT, so it owns no region axis.  Borrowing the parent's gave it
        # `region_axes` with `split=1`, and every region-count reader had to work around that.
        shared = Shared(vector_elements=copy_width, ring_depth=frags[name].lds_buffers)
        operands.append(Operand(
            name, free_mode,
            trajectory=Trajectory(Global(), shared,
                                  configure_read(fragment, name, movements.lrvw)),
            elem_bytes=1))

    if hasMXA:
        add_scale("MXSA", "M_inner", movements.grvwA, nest.N_BCAST,
                  matrix_instruction[0], kernel["MXBlockA"])
    if hasMXB:
        add_scale("MXSB", "N_inner", movements.grvwB, nest.M_BCAST,
                  matrix_instruction[1], kernel["MXBlockB"])

    return hasMXA, hasMXB, operands


def _fuse_view(hasMXA, hasMXB, kernel):
    """The solution-shaped mapping `Components.TDMFuse` reads, built from the theta params.

    A row that needs scales the problem does not have declines and falls back to the default,
    which is `tdmGrouping`'s own answer rather than a second rule stated here.
    """
    tdmInst = int(kernel.get("TDMInst", 0) or 0)
    return {"TDMFuse": int(kernel.get("TDMFuse", 0) or 0),
            "NumWaves": max(1, int(kernel["NumWaves"])),
            "TDMInst": tdmInst,
            "enableTDMA": bool(tdmInst & 0x01), "enableTDMB": bool(tdmInst & 0x02),
            # AB_first accepts on the split and the path, so the view owes both.
            "UseLoopModel": True,
            "TDMSplitA": int(kernel.get("TDMSplitA", 0) or 0),
            "TDMSplitB": int(kernel.get("TDMSplitB", 0) or 0),
            "ProblemType": {"MXBlockA": hasMXA, "MXBlockB": hasMXB, "Sparse": 0}}


def _build_fuse(hasMXA, hasMXB, kernel, operands):
    """The Phi fusion pattern: which operands ride each movement, in issue order.

    A tuple with one member is a solo movement, which only a tensor ALREADY seated needs stating
    -- it is a later step of its set's walk.  A lone newcomer is simply unfused and carries none.
    """
    view = _fuse_view(hasMXA, hasMXB, kernel)
    present = {operand.name for operand in operands}
    rows = liveGroups(view) if tdmWaveSeparated(view) else ()
    groups, seated = [], set()
    for row in rows:
        members = [t for t in row if t in present]
        if len(members) > 1 or (members and set(members) <= seated):
            groups.append(members)
        seated.update(members)
    return view, groups


def _agent_spread(kernel, tile01):
    """`(vector width, agents on this axis, vector groups each holds)` for one free axis.

    The read-address terms the region derivation needs. One function because the effect
    predicate and the region layout each need them and must not disagree.
    """
    mi = list(kernel["MatrixInstruction"])
    wave_group = list(kernel.get("MIWaveGroup")
                      or (mi[7:9] if len(mi) >= 9 else [1, 1])) + [1, 1]
    wave_tile = list(kernel.get("MIWaveTile") or mi[5:7]) + [1, 1]
    parent = "A" if tile01 == 0 else "B"
    width = max(1, int(kernel.get("VectorWidth" + parent, 1) or 1))
    return (width,
            max(1, int(wave_group[tile01])),
            max(1, int(wave_tile[tile01]) // width))


def _assign_effects(operands, kernel, nest, splits):
    """Compile target read grouping into logical Effects.

    Vector widths and wave-tile sizes are consumed here and disappear. LoopIR
    stores only the resulting tile expression and predicate.
    """

    mi = list(kernel["MatrixInstruction"])
    wave_group = list(
        kernel.get("MIWaveGroup")
        or (mi[7:9] if len(mi) >= 9 else [1, 1])
    )
    wave_group += [1, 1]
    axis_extent = {
        axis.name: int(axis.extent.constant_value)
        for axis in nest.ord_
    }
    total_waves = max(1, int(kernel.get("NumWaves", 1) or 1))

    for operand in operands:
        if operand.name not in ("A", "B", "MXSA", "MXSB"):
            continue
        parent = "A" if operand.name.endswith("A") else "B"
        tile01 = 0 if parent == "A" else 1
        free_role = "M" if parent == "A" else "N"
        free_split = (
            splits.aMT if parent == "A" else splits.bMT
        ) if operand.name in ("A", "B") else 1
        reduction_split = (
            splits.aDU if parent == "A" else splits.bDU
        ) if operand.name in ("A", "B") else 1
        width, wave_extent, groups = _agent_spread(kernel, tile01)
        wave_divisor = 1 if tile01 == 0 else max(1, int(wave_group[0]))
        wave_sensitive = (
            operand.trajectory.shared is not None and total_waves > 1
        )

        def make_effect(
            tile,
            wave,
            region=None,
            *,
            free_role=free_role,
            free_split=free_split,
            reduction_split=reduction_split,
            groups=groups,
            width=width,
            wave_extent=wave_extent,
            wave_divisor=wave_divisor,
            wave_sensitive=wave_sensitive,
            axis_extent=axis_extent,
        ):
            indices = tile.local_indices
            values = {
                axis.name: value
                for axis, value in zip(tile.operand.axes, indices)
            }
            semantic = (
                tile.operand[(wave, *indices)]
                if wave is not None and wave_sensitive
                else tile
            )
            split_name = free_role + "_split"

            flat = Expr.constant(0)
            for name in (split_name, free_role + "_inner"):
                if name in values:
                    flat = flat * axis_extent[name] + values[name]
            wave_value = (
                (wave // wave_divisor) % wave_extent
                if wave is not None and wave_extent > 1
                else Expr.constant(0)
            )
            # A read group is region-pinned exactly when the split divides
            # numVectorsPerTile. Otherwise wave and vector-group jointly
            # select the producer region (the odd-G middle group included).
            pinned = wave_extent <= 1 or groups % free_split == 0
            if free_split > 1 and pinned and split_name in values:
                free_region = (
                    free_split * values[split_name]
                ) // max(1, axis_extent.get(split_name, 1))
            else:
                free_region = (
                    free_split * (wave_extent * (flat // width) + wave_value)
                ) // max(1, wave_extent * groups)

            if region is None or free_split * reduction_split <= 1:
                # A read names the region it SOURCES. The agent term can push that past the
                # region its own coordinate names, and carrying it on the semantic tile is what
                # lets the consumer see the crossing without re-deriving it.
                if (region is None and free_split > 1 and not pinned
                        and split_name in values
                        and wave is not None and wave_sensitive):
                    sourced = list(indices)
                    sourced[[a.name for a in tile.operand.axes].index(split_name)] = free_region
                    semantic = tile.operand[(wave, *sourced)]
                return Effect(semantic)
            reduction_region = Expr.constant(0)
            if reduction_split > 1 and "K_split" in values:
                reduction_region = (
                    reduction_split * values["K_split"]
                ) // axis_extent["K_split"]
            return Effect(
                semantic,
                free_region * reduction_split + reduction_region == int(region),
            )

        operand.effect = make_effect


def _read_off_requests(kernel, operands, outer_level, copy_depth, read_depth):
    """PrefetchGlobalRead becomes off(copy, iter); PrefetchLocalRead is a per-operand request."""
    read_requests = {}
    for operand in operands:
        copies = _per_operand(kernel, "PrefetchGlobalRead", operand.name, copy_depth)
        # A scale reads the local-read parameters of the tensor it scales.
        reads = _per_operand(kernel, "PrefetchLocalRead",
                             _REGISTER_SIDE.get(operand.name, operand.name), read_depth)
        if operand.movements and operand.movements[0].dst is SHARED and copies > 0:
            movement = operand.movements[0]
            movement.destination.offsets[outer_level] = copies
        if (operand.movements and operand.movements[-1].dst is REGISTER
                and operand.movements[-1].src is SHARED and reads > 0):
            read_requests[operand.name] = reads
    return read_requests


def _copy_wave_shares(view, groups, operands, waves):
    """How many waves cooperate on each operand's global->LDS copy."""
    names = {operand.name for operand in operands}
    share = {}
    for group in groups:
        for name in group:
            share[name] = max(1, len(tdmWavePartition(view, name)[1]))
    for name in names:  # an operand in no group keeps every wave
        if not any(name in group for group in groups):
            share[name] = waves
    return {name: count for name, count in share.items() if name in names}


def _with_copy_shares(assignment, agent_inputs, shares):
    """Add copy-side wave assignments."""
    entries = tuple(AgentAxisAssignment(
                        axis=agent_inputs[name], level="wave", extent=count,
                        origin=(name, "copy"), role="copy")
                    for name, count in sorted(shares.items())
                    if name in agent_inputs and agent_inputs[name])
    return (AgentAssignment(
                axis_assignments=assignment.axis_assignments + entries)
            if entries else assignment)


def _build_theta(
    agent_inputs,
    copy_depth,
    read_depth,
    view,
    groups,
    kernel,
    nest,
    operands,
    assignment,
    validate_register_depths,
):
    """The retime matrix `off`, the copy-side wave partition, and the assembled Theta."""
    waves = max(1, int(kernel["NumWaves"]))
    read_requests = _read_off_requests(
        kernel, operands, nest.ord_[0].name, copy_depth, read_depth)
    if waves > 1:
        assignment = _with_copy_shares(
            assignment, agent_inputs, _copy_wave_shares(view, groups, operands, waves))
    theta = Theta(operands=operands, ord=tuple(nest.ord_), reg_bytes=REG_BYTES, lanes=WAVE32,
                  fused_copy_groups=groups, agent_assignment=assignment, wave_count=waves,
                  per_region_completion=bool(kernel["PerRegionCompletion"]),
                  suppress_no_load_loop=bool(kernel.get("SuppressNoLoadLoop", False)))
    for operand in theta.operands:
        if operand.movements and operand.fragment:
            operand.fragment.grouping_mode = geometry.grouping_mode_name(theta, operand)
    _set_register_depths(theta, kernel)
    _place_read_offsets(theta, read_requests)
    if validate_register_depths:
        _validate_register_depths(theta, kernel)
    return theta


#: an operand reads the PLR/CLR of the data tensor it belongs to
_REGISTER_SIDE = {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}


def _set_register_depths(theta, kernel):
    """Apply the register values calculated by LoopModel.Registers."""

    def value(base, side):
        result = kernel.get(base + side)
        if result is None or int(result) == -1:
            raise RuntimeError(
                "%s%s was not calculated before theta construction"
                % (base, side))
        return int(result)

    for operand in theta.operands:
        if not (operand.movements and operand.fragment):
            continue
        side = _REGISTER_SIDE.get(operand.name)
        if side is None:
            continue
        value("PrefetchLocalRead", side)
        clr = value("ClusterLocalRead", side)
        vg = value("VgprGroup", side)
        depth = value("VgprAlloc", side)
        if (clr and not bool(kernel.get("VgprAllocExplicit" + side, False))) \
                or int(getattr(operand.fragment, "agent_group_size", 0) or 0) > 0:
            depth = min(depth, geometry._ring_period(theta, operand))
        operand.fragment.vgpr_group = vg
        unit = geometry.prefetch_unit_tiles(theta, operand)
        if vg > 1 and unit % vg:
            raise RuntimeError(
                "VgprGroup%s=%d does not divide %s's %d-tile prefetch unit under this loop "
                "order." % (side, vg, operand.name, unit))
        operand.fragment.ring_depths = {
            label: depth for label in operand.fragment.groups()}

def _validate_register_depths(theta, kernel):
    def value(base, side):
        result = kernel.get(base + side)
        if result is None or int(result) == -1:
            raise RuntimeError(
                "%s%s was not calculated before theta construction"
                % (base, side))
        return int(result)

    from .schedule import BufferDepths
    depths = BufferDepths(theta, include_shared=False)
    for operand in theta.operands:
        if not (operand.movements and operand.fragment):
            continue
        if operand.name not in ("A", "B"):
            continue
        side = _REGISTER_SIDE.get(operand.name)
        if side is None:
            continue
        allocated = value("VgprAlloc", side)
        cluster = value("ClusterLocalRead", side)
        group = operand.fragment.groups()[0]
        buffers = geometry.group_ring_depth(
            theta,
            operand,
            group,
            depths,
        )
        required = 1
        if value("VgprGroup", side) > 1 \
                and geometry.reloads_whole_set(theta, operand) \
                    and geometry.requested_read_ahead(theta, operand, depths) > 0 \
                    and not geometry.vgpr_ring_carries(
                        theta, operand, group, buffers):
            required = max(
                required,
                geometry._ring_period(theta, operand),
                geometry.group_reuse_floor(
                    theta,
                    operand,
                    group,
                    charge_lead=False,
                    lead=0,
                ),
            )
        if cluster and geometry.reloads_whole_set(theta, operand):
            required = max(required, geometry._ring_period(theta, operand))
        elif cluster and not geometry.vgpr_ring_carries(
                theta, operand, group, buffers):
            required = max(
                required,
                geometry.group_reuse_floor(
                    theta,
                    operand,
                    group,
                    charge_lead=False,
                    lead=0,
                ),
            )
        if allocated < required:
            raise RuntimeError(
                "VgprAlloc%s=%d is smaller than %s's required "
                "rotation-unit depth %d"
                % (side, allocated, operand.name, required))


def params_to_theta(p: dict, *, validate_register_depths=True) -> Theta:
    """Translate a TensileLite parameter dictionary into Theta."""
    copy_depth, read_depth, elem_bytes, fanM, fanN, kernel, substeps, matrix_instruction = _read_params(p)
    splits = _read_splits(kernel)
    nest = _build_ord(fanM, fanN, kernel, substeps, splits)
    movements = _read_movements(kernel)
    configure_read, agent_inputs, assignment = _build_agent_assignment(movements, nest)
    frags = _build_fragments(kernel, matrix_instruction, nest, splits)
    hasMXA, hasMXB, operands = _build_operands(
        configure_read, elem_bytes, frags, movements, kernel, matrix_instruction, nest, splits)
    _assign_effects(operands, kernel, nest, splits)
    view, groups = _build_fuse(hasMXA, hasMXB, kernel, operands)
    theta = _build_theta(
        agent_inputs,
        copy_depth,
        read_depth,
        view,
        groups,
        kernel,
        nest,
        operands,
        assignment,
        validate_register_depths,
    )
    return theta


def _place_read_offsets(theta, requests):
    for operand in theta.operands:
        depth = requests.get(operand.name, 0)
        if depth <= 0:
            continue
        axis = readahead_level(theta)
        if axis is None:  # no intra-region axis this operand advances along
            continue
        movement = operand.trajectory.fragment_fill
        movement.destination.offsets[axis[0]] = depth


# --- solution --------------------------------------------------------------

def _shape_params(elem_bytes, kernel, mi, wt0, wt1):
    """The kernel's own shape: instruction, depth, fans, vector widths, prefetch depths."""
    return {
        "MatrixInstruction": mi,
        "DepthU": kernel["DepthU"],
        "MIWaveTile": [wt0, wt1],
        # A derived Solution spells the instruction [M,N,K,B], so `mi[7:9]` never fires.
        "MIWaveGroup": list(kernel.get("MIWaveGroup")
                            or (mi[7:9] if len(mi) >= 9 else [1, 1])),
        "ElemBytes": max(1, elem_bytes),
        "PrefetchGlobalRead": kernel.get("PrefetchGlobalRead", 2),
        # Both spellings: the solution keeps the pair only where the operands differ.
        "PrefetchLocalRead": kernel.get("PrefetchLocalRead", 1),
        "PrefetchLocalReadA": kernel.get("PrefetchLocalReadA", -1),
        "PrefetchLocalReadB": kernel.get("PrefetchLocalReadB", -1),
        "ClusterLocalRead": kernel.get("ClusterLocalRead", 0),
        "ClusterLocalReadA": kernel.get("ClusterLocalReadA", -1),
        "VgprGroupA": kernel.get("VgprGroupA", -1),
        "VgprGroupB": kernel.get("VgprGroupB", -1),
        "VgprAllocA": kernel.get("VgprAllocA", -1),
        "VgprAllocB": kernel.get("VgprAllocB", -1),
        "VgprAllocExplicitA": bool(kernel.get("VgprAllocExplicitA", False)),
        "VgprAllocExplicitB": bool(kernel.get("VgprAllocExplicitB", False)),
        "ClusterLocalReadB": kernel.get("ClusterLocalReadB", -1),
        # The BITMASK and its two bits: HalfPLR=1 is A alone, so the pair is what theta reads.
        "HalfPLR": kernel.get("HalfPLR", 0),
        "HalfPLRA": bool(kernel.get("HalfPLRA", int(kernel.get("HalfPLR", 0) or 0) & 0x1)),
        "HalfPLRB": bool(kernel.get("HalfPLRB", int(kernel.get("HalfPLR", 0) or 0) & 0x2)),
        # DERIVED by Solution, not asked for: HalfPLR and RAP both turn it on.
        "SuppressNoLoadLoop": bool(kernel.get("SuppressNoLoadLoop", False)),
        "GlobalReadVectorWidthA": kernel.get("GlobalReadVectorWidthA", 8) or 8,
        "GlobalReadVectorWidthB": kernel.get("GlobalReadVectorWidthB", 8) or 8,
        "VectorWidthA": kernel.get("VectorWidthA", 2) or 2,
        "VectorWidthB": kernel.get("VectorWidthB", 2) or 2,
        "LocalReadVectorWidth": kernel.get("LocalReadVectorWidth", 4) or 4,
        "DirectToVgprA": bool(kernel.get("DirectToVgprA", False)),
        "DirectToVgprB": bool(kernel.get("DirectToVgprB", False)),
        "WaveSeparateGlobalReadA": kernel.get("WaveSeparateGlobalReadA", 0),
        "WaveSeparateGlobalReadB": kernel.get("WaveSeparateGlobalReadB", 0),
        "InnerUnroll": kernel.get("InnerUnroll", 1) or 1,
        "NumLdsBlk": kernel.get("NumLdsBlk", 2),
        "1LDSBuffer": kernel.get("1LDSBuffer", 0),
    }


def _movement_params(kernel, target):
    """What the target and the layout supply: fusion, agents, splits, the coverage."""
    return {
        "TDMFuse": int(kernel.get("TDMFuse", 0) or 0),
        # A fuse is a wave selection between the movements the TDM makes, so theta needs to know
        # which tensors it moves -- `enableTDM{A,B}` is the resolved form of `TDMInst`.
        "TDMInst": ((0x01 if kernel.get("enableTDMA") else 0)
                    | (0x02 if kernel.get("enableTDMB") else 0)
                    or int(kernel.get("TDMInst", 0) or 0)),
        "NumWaves": kernel.get("NumWaves", 1),
        "TDMSplit": _tdm_split.split_factors(kernel) or [1, 1, 1, 1],
        "TDMSplitA": int(kernel.get("TDMSplitA", 0) or 0),
        "TDMSplitB": int(kernel.get("TDMSplitB", 0) or 0),
        "TDMSplitWaveRegions": [_tdm_split.wave_region_span(kernel, "A"),
                                _tdm_split.wave_region_span(kernel, "B")],
        # The extra WMMA axes, which TDMSplit pins where it splits and leaves free where it does not.
        "WmmaFactorM": int(kernel.get("WmmaFactorM", -1) or -1),
        "WmmaFactorN": int(kernel.get("WmmaFactorN", -1) or -1),
        "WmmaFactorK": int(kernel.get("WmmaFactorK", -1) or -1),
        "MXBlockA": int(kernel["ProblemType"].get("MXBlockA", 0) or 0),
        "MXBlockB": int(kernel["ProblemType"].get("MXBlockB", 0) or 0),
        "LoopOrder": kernel["LoopOrder"],
        "ReadVectorElems": _read_vector_elems(kernel, target),
        "ReadQuantum": _read_coverage(kernel, target),
        # The MX half-wave fold is DERIVED from the kernel, not only supplied by a target: one
        # ds_load carries blocks 2g and 2g+1 across the two half-waves (a Phi over a rho span).
        "ReadPhi": dict({tc: f[0] for tc in ("MXSA", "MXSB")
                         for f in (_mxs_fold(kernel, tc),) if f[0] > 1},
                        **((target or {}).get("ReadPhi", {}) or {})),
        "ReadRho": dict({tc: f[1] for tc in ("MXSA", "MXSB")
                         for f in (_mxs_fold(kernel, tc),) if f[1] > 1},
                        **((target or {}).get("ReadRho", {}) or {})),
    }


def kernel_to_params(kernel, target=None) -> dict:
    """Translate a TensileLite Solution dict into the decoder's TensileLite-param dict."""
    mi = list(kernel["MatrixInstruction"])
    wt0 = kernel.get("MIWaveTileA", kernel.get("MIWaveTile", [1, 1])[0])
    wt1 = kernel.get("MIWaveTileB", kernel.get("MIWaveTile", [1, 1])[1]
                     if isinstance(kernel.get("MIWaveTile"), (list, tuple)) else 1)

    try:
        elem_bytes = int(kernel["ProblemType"]["DataType"].numBytes())
    except Exception:
        elem_bytes = 2

    p = _shape_params(elem_bytes, kernel, mi, wt0, wt1)
    p.update(_movement_params(kernel, target))
    return p


def _mxs_fold(kernel, tc):
    """The MX scale read fold as `(phi, rho)` -- how many tiles one ds_read carries, and over how
    many sub-agents.

    TWO folds compose.  CONTIGUOUS: one wide ds_read covers `VectorWidth{tc}` adjacent tiles, whose
    scales are contiguous in LDS (`tilePerRead` in `leaves.buildMxScaleReadContext`).  DISTRIBUTED:
    the half-wave pair below, block `2g` low and `2g+1` high -- paper 2.2's scattered carrier group.
    """
    contiguous = max(1, int(kernel.get("VectorWidth%s" % tc, 1) or 1))
    distributed = _mxs_tile_span(kernel, tc)
    phi = contiguous * distributed
    return (phi, distributed if distributed > 1 else 0)


def _mxs_tile_span(kernel, tc) -> int:
    """The MX scale half-wave fold: `2` when one ds_load carries two scale blocks, else 1.

    theta's own form of `LocalRead.getMxsTileSpanInfo`.  One instruction holds block `2g` on the
    lower half-wave and `2g+1` on the upper, so it is a DISTRIBUTED carrier group -- Phi over a
    rho sub-agent span -- and modelling it here is what lets GIR carry one Move per instruction.
    """
    tile01 = 0 if tc.endswith("A") else 1
    pt = kernel.get("ProblemType", {}) or {}
    if "MXS" not in tc or not int(pt.get("MXBlock%s" % tc[-1], 0) or 0):
        return 1
    if tuple(kernel.get("ISA") or ()) != (12, 5, 0) \
            or kernel.get("MXScaleFormat") != "InMemorySwizzle":
        return 1
    vw = int(kernel.get("VectorWidth%s" % tc, 0) or 0)
    wave_tile = (kernel.get("MIWaveTile") or [0, 0])[tile01]
    if vw < 1 or not wave_tile:
        return 1
    vectors = int(wave_tile) // vw
    if vectors < 2 or vectors % 2:
        return 1
    inst_t = kernel.get("MatrixInstM" if tile01 == 0 else "MatrixInstN")
    if not inst_t or int(inst_t) != int(kernel.get("WavefrontSize", 0) or 0) // 2:
        return 1
    return 2


def _read_coverage(kernel, target) -> dict:
    return dict((target or {}).get("ReadQuantum", {}) or {})


def _read_vector_elems(kernel, target) -> dict:
    """{operand: elements one shared->register instruction moves per lane} --'s coverage domain."""
    out = dict((target or {}).get("ReadVectorElems", {}) or {})
    pt = kernel.get("ProblemType", {})
    for operand_letter, mx in (("MXSA", "MXBlockA"), ("MXSB", "MXBlockB")):
        if operand_letter in out or not int(pt.get(mx, 0) or 0):
            continue
        if kernel.get("UnrollMajorLDS%s" % operand_letter, kernel.get("UnrollMajorLDSA", 1)):
            out[operand_letter] = int(kernel.get("LocalReadVectorWidthMXS", 0) or 0) or 1
        else:
            out[operand_letter] = 1
    return out


