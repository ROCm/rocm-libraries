# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import COPY, FORWARD, HOP_COUNTER, READ, REVERSE, STORE, Space


def group_labels(count: int) -> tuple:
    """Register groups are indexed, not named: group i is `g{i}` for every count."""
    return tuple(f"g{i}" for i in range(count))

@dataclass(frozen=True)
class Axis:
    """One tiled traversal axis: a name and its extent."""
    name: str
    extent: int = 0  # static trip count; >0 = inner (unrolled), 0 = outer (runtime loop)

    @property
    def is_outer(self) -> bool:
        return self.extent == 0


@dataclass
class Fragment:
    """Register placement and the movement that fills it."""
    broadcast_axes: set = field(default_factory=set)
    parts: tuple = (1,)
    labels: tuple = None
    grouping_mode: str = None
    group_policy: dict = None
    fragment_elements: int = 4
    vector_elements: int = 0
    coverage: object = None
    instances_per_instruction: int = 1
    agent_group_size: int = 0
    kind: str = "load"
    offsets: dict = field(default_factory=dict)
    ring_depths: dict = None

    space = Space.REGISTER

    def __post_init__(self):
        if isinstance(self.parts, int):
            if self.parts < 1:
                raise RuntimeError(f"Fragment.parts={self.parts} must be >=1")
            self.parts = (1,) * self.parts

    def group_labels(self) -> tuple:
        if self.labels is not None:
            return tuple(self.labels)
        return tuple(f"g{i}" for i in range(len(self.parts)))

    def groups(self) -> tuple:
        return self.group_labels()

    def size_of(self, label) -> int:
        labs = self.group_labels()
        return self.parts[labs.index(label)] if label in labs else 1

    def total_slots(self) -> int:
        return sum(self.parts)

    def group_broadcast(self) -> set:
        return set(self.broadcast_axes)

    def policy_of(self, group):
        gp = self.group_policy or {}
        return gp.get(group, gp.get("*", "unroll"))


@dataclass(frozen=True)
class RegionLayout:
    """Storage-region geometry for one shared placement."""
    axes: tuple = ()
    span: dict = field(default_factory=dict)
    free_split: int = 1
    wave_span: int = 1

    @property
    def wave_relative(self) -> bool:
        return max(1, int(self.wave_span)) < max(1, int(self.free_split))

    def index(self, coord) -> tuple:
        at = dict(coord)
        return tuple(int(at.get(axis, 0)) // max(1, int(self.span.get(axis, 1)))
                     for axis in self.axes)


@dataclass
class Global:
    """Global-memory placement."""
    vector_elements: int = 0
    coverage: object = None
    kind: str = "load"
    offsets: dict = field(default_factory=dict)

    space = Space.GLOBAL


@dataclass
class Shared:
    """Shared-memory placement and its incoming copy."""
    vector_elements: int = 0
    split: int = 1
    ring_depth: int = 0
    regions: RegionLayout = field(default_factory=RegionLayout)
    coverage: object = None
    kind: str = "tdm"
    offsets: dict = field(default_factory=dict)

    space = Space.SHARED


@dataclass(frozen=True)
class Movement:
    """Movement between adjacent placements."""
    source: object
    destination: object

    @property
    def src(self) -> str:
        return self.source.space

    @property
    def dst(self) -> str:
        return self.destination.space

    @property
    def vector_elems(self) -> int:
        return int(getattr(self.destination, "vector_elements", 0) or 0)

    @property
    def coverage(self):
        return getattr(self.destination, "coverage", None)

    @property
    def instances_per_instruction(self) -> int:
        return max(1, int(getattr(self.destination, "instances_per_instruction", 1) or 1))

    @property
    def agent_group_size(self) -> int:
        return int(getattr(self.destination, "agent_group_size", 0) or 0)

    @property
    def kind(self) -> str:
        return getattr(self.destination, "kind", None) or (
            "tdm" if (self.src, self.dst) == (Space.GLOBAL, Space.SHARED) else "load")

    @property
    def split(self) -> int:
        return max(1, int(getattr(self.destination, "split", 1) or 1))

    @property
    def is_bulk(self) -> bool:
        return self.kind == "tdm"

    @property
    def role(self) -> str:
        try:
            return HOP_ROLE[(self.src, self.dst)]
        except KeyError:
            raise RuntimeError(f"movement {self.src}->{self.dst} has no operation role") from None

    @property
    def counter(self):
        return HOP_COUNTER[(self.src, self.dst)]

    @property
    def offsets(self) -> dict:
        return getattr(self.destination, "offsets", {})


@dataclass
class Trajectory:
    """Ordered placements for one operand."""
    placements: list

    def __init__(self, *placements):
        if len(placements) == 1 and isinstance(placements[0], (list, tuple)):
            placements = tuple(placements[0])
        self.placements = list(placements)
        if not self.placements:
            raise RuntimeError("a trajectory needs at least one placement")
        self._derive()

    def _derive(self):
        """Settle what the placement list implies.  Every mutator below ends by calling this.

        A hop is a PAIR of neighbours, so the movements and the four lookups are all restated
        whenever `placements` changes -- and only then.
        """
        self._movements = tuple(Movement(a, b)
                                for a, b in zip(self.placements, self.placements[1:]))
        self._shared = next((p for p in self.placements if isinstance(p, Shared)), None)
        self._fragment = next((p for p in reversed(self.placements)
                               if isinstance(p, Fragment)), None)
        self._shared_fill = next((m for m in self._movements
                                  if isinstance(m.destination, Shared)), None)
        self._fragment_fill = next((m for m in self._movements
                                    if isinstance(m.destination, Fragment)), None)
        fill = self._fragment_fill
        self._shared_read = fill if fill is not None and isinstance(fill.source, Shared) else None

    @property
    def movements(self) -> tuple:
        return self._movements

    @property
    def shared(self):
        return self._shared

    @property
    def fragment(self):
        return self._fragment

    @property
    def shared_fill(self):
        return self._shared_fill

    @property
    def fragment_fill(self):
        return self._fragment_fill

    @property
    def shared_read(self):
        return self._shared_read

    def add(self, placement):
        self.placements.append(placement)
        self._derive()

    def replace(self, old, new):
        self.placements[self.placements.index(old)] = new
        self._derive()


# --------------------------------------------------------------------------- The OP-CLASS role

HOP_ROLE = {
    (Space.GLOBAL,   Space.SHARED):   COPY,
    (Space.SHARED,   Space.REGISTER): READ,
    (Space.GLOBAL,   Space.REGISTER): READ,
    (Space.REGISTER, Space.SHARED):   STORE,
    (Space.SHARED,   Space.GLOBAL):   STORE,
    (Space.REGISTER, Space.GLOBAL):   STORE,
}


# --------------------------------------------------------------------------- TRAJECTORY SENSE --

_SPACE_RANK = {Space.GLOBAL: 0, Space.SHARED: 1, Space.REGISTER: 2}


def path_direction(op) -> str:
    """Return the trajectory's direction."""
    sense = REVERSE if op.is_output else FORWARD
    ranks = [_SPACE_RANK.get(placement.space) for placement in op.trajectory.placements]
    net = [b - a for a, b in zip(ranks, ranks[1:]) if a is not None and b is not None]
    if net and ((sense == FORWARD) != (sum(net) > 0)):
        raise RuntimeError(
            f"operand {op.name!r} is role={op.role!r}, but its trajectory "
            f"{[placement.space for placement in op.trajectory.placements]} runs the other way")
    return sense


@dataclass
class Operand:
    """Tensor identity and its placement trajectory."""
    name: str
    free_mode: str
    trajectory: Trajectory
    elem_bytes: int = 2
    role: str = "input"

    @property
    def is_output(self) -> bool:
        return self.role == "output"

    @property
    def is_input(self) -> bool:
        return self.role == "input"

    @property
    def movements(self) -> tuple:
        return self.trajectory.movements

    @property
    def fragment(self):
        return self.trajectory.fragment

    @fragment.setter
    def fragment(self, value):
        old = self.trajectory.fragment
        if old is None:
            self.trajectory.add(value)
        else:
            self.trajectory.replace(old, value)

    @property
    def frag_elems(self) -> int:
        return self.fragment.fragment_elements if self.fragment is not None else 0

    @property
    def lds_buffers(self) -> int:
        shared = self.trajectory.shared
        return int(shared.ring_depth) if shared is not None else 0

    @property
    def region_axes(self) -> tuple:
        shared = self.trajectory.shared
        return shared.regions.axes if shared is not None else ()

    @property
    def region_span(self) -> dict:
        shared = self.trajectory.shared
        return shared.regions.span if shared is not None else {}

    @property
    def free_split(self) -> int:
        shared = self.trajectory.shared
        return shared.regions.free_split if shared is not None else 1

    @property
    def wave_region_span(self) -> int:
        shared = self.trajectory.shared
        return shared.regions.wave_span if shared is not None else 1

    def region_of(self, coord) -> tuple:
        shared = self.trajectory.shared
        return shared.regions.index(coord) if shared is not None else ()

    @property
    def split(self) -> int:
        shared = self.trajectory.shared
        return max(1, shared.split) if shared is not None else 1

    def has_bulk(self) -> bool:
        return any(movement.is_bulk for movement in self.movements)


# ---  ------------------------------------------------------------------------------------------------

AGENT_LEVELS = ("block", "wave", "subwave")

ASSIGNMENT_ROLES = ("read", "copy")


@dataclass(frozen=True)
class AgentAxisAssignment:
    """Assign one loop axis to a hardware-agent level."""
    axis:   str
    level:  str
    extent: int
    origin: tuple = ()
    role:   str = "read"

    def __post_init__(self):
        if self.level not in AGENT_LEVELS:
            raise RuntimeError("agent level %r not in %r" % (self.level, AGENT_LEVELS))
        if int(self.extent) < 1:
            raise RuntimeError("agent extent must be >= 1, got %r" % (self.extent,))
        if self.role not in ASSIGNMENT_ROLES:
            raise RuntimeError("assignment role %r not in %r" % (self.role, ASSIGNMENT_ROLES))

    @property
    def opclass(self) -> str:
        return self.origin[0] if self.origin else ""

    @property
    def rank(self) -> int:
        return AGENT_LEVELS.index(self.level)


@dataclass(frozen=True)
class AgentAssignment:
    """Loop-axis assignment and producer/consumer partition."""
    axis_assignments: tuple = ()

    def __post_init__(self):
        limit = AGENT_LEVELS.index("wave")
        object.__setattr__(self, "_served",
                           frozenset(r.axis for r in self.axis_assignments
                                     if r.rank <= limit and r.role == "read"))

    def assignments_at(self, *levels, role: str = "read") -> tuple:
        return tuple(r for r in self.axis_assignments if r.level in levels and r.role == role)

    def coarser_than(self, level: str, role: str = "read") -> tuple:
        lim = AGENT_LEVELS.index(level)
        return tuple(r for r in self.axis_assignments if r.rank <= lim and r.role == role)

    def copy_agent_count(self, opname: str, default: int = 1) -> int:
        hits = [item for item in self.axis_assignments
                if item.role == "copy" and item.opclass == opname]
        if not hits:
            return default
        return max(int(item.extent) for item in hits)

    def served_axes(self) -> frozenset:
        return self._served

    def group_size_over(self, axes, level: str = "subwave") -> int:
        hits = [item for item in self.assignments_at(level) if item.axis in axes]
        if not hits:
            return 0
        if len(hits) > 1:
            raise RuntimeError(
                "%d axes assigned to %s within one presence set (%s)"
                % (len(hits), level, ", ".join(sorted(item.axis for item in hits))))
        return int(hits[0].extent)

def mode_role(name) -> str:
    """Logical K/M/N role of a factorized mode name, or ``None``."""
    role = str(name).split("_", 1)[0]
    return role if role in ("K", "M", "N") else None


def readahead_level_of(inner, region_names, broadcast_names=(), rotation_regions=()):
    """`(name, extent, span)` of the axis one PLR step advances: the OUTERMOST non-region axis.

 A step must REACH something, so a degenerate axis is never the level: it is in the nest whatever
 its extent, and choosing it would make the answer depend on that.  `span` counts every axis INSIDE
 the level, REGION AXES INCLUDED, because one step covers them; a split outside the level is not
 covered, so the read-ahead re-issues there.
    """
    bcast = set(broadcast_names or ())
    inner = [axis for axis in inner if axis.name not in bcast]
    regions = set(region_names or ())
    at = next((i for i, axis in enumerate(inner) if axis.name not in regions), None)
    if at is None:
        return None
    span = 1
    for axis in inner[at + 1:]:
        span *= max(1, int(axis.extent))
    return inner[at].name, max(1, int(inner[at].extent)), span


@dataclass(frozen=True)
class OperandGeometry:
    """What one operand's traversal is, given the nest -- derived once, at `Theta` construction.

    Every field below is a function of `ord`, the operand's own `region_axes`, and the axes its
    fragment declares it constant over.  None of the three moves after the theta is built: the
    register-depth search re-partitions `parts`/`labels`/`ring_depths`, which nothing here reads.
    """
    broadcast: frozenset            #: intra-iteration axes the operand does NOT depend on
    presence: tuple                 #: the walked axes it does vary over, outer -> inner
    varying: tuple                  #: `presence` as (name, extent), its physical tile modes
    role_modes: tuple               #: `varying` minus the region axes that select storage
    outer_role: str                 #: outermost logical role it varies on
    inner_role: str                 #: innermost -- the prefetch-unit axis
    reloads_whole_set: bool         #: invariant over the global outermost role
    rotation: tuple                 #: modes whose tile product is one rotation unit
    grouping: tuple                 #: the one contiguous mode register groups partition
    original_prefetch: tuple        #: every factor of the inner role, region factors included
    prefetch_unit: tuple            #: `original_prefetch` per region
    unit_enumerator: tuple          #: factors of the axis that enumerates successive units
    summation_modes: tuple          #: `varying` restricted to the reduction axes
    reduction_ring: tuple           #: the reduction ring when a free prefetch factor is packed


@dataclass
class Theta:
    operands: list
    ord: tuple  # the full permutation of looped axes, outer -> inner
    reg_bytes: int = 4
    lanes: int = 32
    fused_copy_groups: list = field(default_factory=list)
    agent_assignment: AgentAssignment = field(default_factory=AgentAssignment)
    wave_count: int = 1
    per_region_completion: bool = False

    def __post_init__(self):
        """The axis nest is fixed at construction, so split it here rather than on every read.

        Nothing rebinds `theta.ord`; use `dataclasses.replace` to get a theta with a different
        nest, which runs this again.
        """
        self.ord = tuple(self.ord)
        self.inner = tuple(axis for axis in self.ord if not axis.is_outer)
        self.outer = tuple(axis for axis in self.ord if axis.is_outer)
        extents = {axis.name: max(1, int(axis.extent)) for axis in self.inner}
        self._inner_axes = tuple(axis for axis in self.inner
                                 if extents[axis.name] > 1
                                 or extents.get(self._TILE_REGION.get(axis.name, ""), 1) > 1)
        self._readahead_level = readahead_level_of(self._inner_axes, self.region_axis_names())
        base = {operand.name: self._derive_presence(operand) for operand in self.operands}
        self._summation_names = self._derive_summation_names(
            {name: presence for name, (_b, presence, _v) in base.items()})
        #: the per-operand traversal, keyed by name because `replace(theta, ord=...)` reuses the
        #: SAME Operand objects -- a field on the operand would be the second theta's, not this one's
        self._geometry = {operand.name: self._derive_geometry(operand, base[operand.name])
                          for operand in self.operands}

    # --- convenience lookups ------------------------------------------------
    def op(self, name):
        """The operand of this name."""
        for operand in self.operands:
            if operand.name == name:
                return operand
        raise RuntimeError("no operand %r in theta (have %s)"
                         % (name, ", ".join(operand.name for operand in self.operands)))

    def agent_distributed(self, opname) -> bool:
        if self.op(opname).trajectory.shared is None:
            return False
        return self.wave_count > 1

    def movement_units(self):
        copy_ops = [operand for operand in self.operands if operand.trajectory.shared is not None]
        by_name = {operand.name: operand for operand in copy_ops}
        units, seen = [], set()
        for op in copy_ops:
            key = next((tuple(tile for tile in group if tile in by_name)
                        for group in self.fused_copy_groups
                        if op.name in group), (op.name,))
            if key in seen:
                continue
            seen.add(key)
            members = [by_name[name] for name in key]
            # THE WALK IS AS LONG AS THE LONGEST MEMBER.  A member with fewer regions than the
            # unit does not move on every step; `Operand.split` says how many it owns, and the
            # copy side skips it on the steps it does not (see `loopir_to_gir._absent_at_region`).
            units.append((key, members, max(max(1, operand.split) for operand in members)))
        return units

    def levels(self):
        return list(self.ord)

    def outer_axes(self):
        return self.outer

    def reduction_chunk_mode(self):
        outer = self.outer_axes()
        return outer[-1] if outer else None

    def summation_chunk_name(self, default=None):
        axis = self.reduction_chunk_mode()
        return axis.name if axis is not None else default

    #: tile axis -> the storage-region axis whose liveness keeps it in the walk
    _TILE_REGION = {"K_inner": "K_split", "M_inner": "M_split", "N_inner": "N_split"}

    def inner_axes(self) -> tuple:
        """The inner axes a traversal WALKS, fixed in `__post_init__` beside `inner`/`outer`.

        `ord` carries every tile axis whatever its extent, so presence there no longer says
        whether anything moves along one.  Something does when the axis itself steps, or when
        the region axis it pairs with does: one tile of a split operand is still visited once
        per region.  `levels()` is the full nest for anyone who needs the declaration.
        """
        return self._inner_axes

    def geometry(self, operand) -> OperandGeometry:
        """This operand's traversal under this nest, derived once in `__post_init__`."""
        return self._geometry[operand.name]

    def region_axis_names(self) -> frozenset:
        """Every axis some operand splits storage on."""
        return frozenset(axis for operand in self.operands
                         for axis in (operand.region_axes or ()))

    def global_outer_role(self) -> str:
        """Outermost logical role in `ord`: a TDMSplit region factor selects storage, not order."""
        regions = self.region_axis_names()
        names = [axis.name for axis in self._inner_axes if axis.name not in regions] \
            or [axis.name for axis in self._inner_axes]
        return next((role for role in map(mode_role, names) if role is not None), None)

    def readahead_level(self):
        """The read-ahead level: a property of `ord`, shared by every operand that walks it."""
        return self._readahead_level

    def _derive_presence(self, operand):
        """`(broadcast, presence, varying)` -- the first pass, which the reduction axes need."""
        walked = {axis.name for axis in self._inner_axes}
        fragment = operand.fragment
        declared = set(fragment.broadcast_axes) if fragment is not None else set()
        broadcast = frozenset({name for name in declared if name in walked}
                              | {name for name in self.wave_served_axes() if name in walked})
        presence = tuple(axis for axis in self._inner_axes if axis.name not in broadcast)
        return broadcast, presence, tuple((axis.name, max(1, int(axis.extent)))
                                          for axis in presence)

    def _derive_summation_names(self, presence) -> frozenset:
        """The axes the output is NOT present on: the reduction axes."""
        on_output = {axis.name for operand in self.operands if operand.is_output
                     for axis in presence[operand.name]}
        served = self.wave_served_axes()
        return frozenset(axis.name for axis in self._inner_axes
                         if axis.name not in on_output and axis.name not in served)

    def _derive_geometry(self, operand, base) -> OperandGeometry:
        """Build one `OperandGeometry`.  See its docstring for why this is settled here."""
        broadcast, presence, varying = base
        regions = set(operand.region_axes or ())
        role_modes = tuple((name, extent) for name, extent in varying
                           if name not in regions) or varying
        roles = [mode_role(name) for name, _extent in role_modes if mode_role(name) is not None]
        outer_role = roles[0] if roles else None
        inner_role = roles[-1] if roles else None

        reloads = outer_role is not None and outer_role != self.global_outer_role()
        rotation = varying if reloads else tuple(
            (name, extent) for name, extent in varying if mode_role(name) != outer_role)
        grouping = tuple((name, extent) for name, extent in rotation
                         if name not in regions)[:1]

        original = tuple((name, extent) for name, extent in varying
                         if mode_role(name) == inner_role)
        prefetch = tuple((name, extent) for name, extent in original if name not in regions)
        enumerator = tuple((name, extent) for name, extent in varying
                           if mode_role(name) == outer_role)
        summation = tuple((name, extent) for name, extent in varying
                          if name in self._summation_names)
        # The reduction ring only stands in when a free prefetch factor is packed into each slot.
        rotation_names = {name for name, _extent in rotation}
        unpacked = 1
        for name, extent in prefetch:
            if name not in rotation_names and name not in regions:
                unpacked *= max(1, int(extent))
        level = self._readahead_level
        reduction = tuple((name, extent) for name, extent in rotation
                          if name in self._summation_names) \
            if (level and level[0] in self._summation_names and unpacked > 1) else ()

        return OperandGeometry(broadcast=broadcast, presence=presence, varying=varying,
                               role_modes=role_modes, outer_role=outer_role,
                               inner_role=inner_role, reloads_whole_set=reloads,
                               rotation=rotation, grouping=grouping,
                               original_prefetch=original, prefetch_unit=prefetch,
                               unit_enumerator=enumerator, summation_modes=summation,
                               reduction_ring=reduction)

    def free_extent(self, mode_name):
        for axis in self.ord:
            if axis.name == mode_name:
                return axis.extent
        return 1

    def off_at(self, opname, role, level_name) -> int:
        operand = self.op(opname) if isinstance(opname, str) else opname
        movement = next((item for item in operand.movements if item.role == role), None)
        return int(movement.offsets.get(level_name, 0)) if movement is not None else 0

    def off_of(self, operand, hop, level_name) -> int:
        return self.off_at(operand, hop.role, level_name)

    def movement_offsets(self) -> dict:
        return {
            (operand.name, movement.role, level): int(depth)
            for operand in self.operands
            for movement in operand.movements
            for level, depth in movement.offsets.items()
        }

    def output_operands(self):
        return [operand for operand in self.operands if operand.is_output]

    def wave_served_axes(self) -> set:
        return self.agent_assignment.served_axes()

    def tensor_instrs_per_kiter(self) -> int:
        """distinct global->shared instructions per kiter (one shared counter tracks all)."""
        byname = {operand.name: operand for operand in self.operands
                  if operand.trajectory.shared is not None}
        fused = {tile for group in self.fused_copy_groups for tile in group}
        n = 0
        for group in self.fused_copy_groups:
            members = [byname[tile] for tile in group if tile in byname]
            n += max((max(1, operand.split) for operand in members), default=1)
        for name, operand in byname.items():  # unfused copies
            if name not in fused:
                n += max(1, operand.split)
        return max(1, n)

