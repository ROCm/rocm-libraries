# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The coordinate graph: named input coordinates -> named output coordinates.

A ``TransformGraph`` says "these loop coordinates become these tensor coordinates". It is a
frozen value built from declarations written in read -> write order::

    g = make_transform_graph(
        inputs(m=N * Ho * Wo, k=Y * X * C),
        unmerge("m", into=("n", "ho", "wo"), lengths=(N, Ho, Wo)),
        unmerge("k", into=("y", "x", "c"), lengths=(Y, X, C)),
        embed(("ho", "y"), into="h_dil", coeffs=(sH, dH)),
        offset("h_dil", into="hi", shift=-pH),
        embed(("wo", "x"), into="w_dil", coeffs=(sW, dW)),
        offset("w_dil", into="wi", shift=-pW),
        outputs("n", "hi", "wi", "c"),
    )
    g.evaluate((m, k))   # -> (n, hi, wi, c), in outputs(...) order

Why the graph holds no tensor and no bound: a tensor's lengths have one home, its
``TensorDesc``. A graph that restated a length (as a CK ``pad`` does) would be a second home that
can drift from the first. So the graph declares only the domains its own kinds need (an input's
extent, the pieces of an ``unmerge``, the product a ``merge`` writes, the power-of-two extent an
``xor`` needs), and an output coordinate is checked against the tensor it is bound to, later.
There is no PAD kind: a halo is ``offset(..., shift=-left)`` and the tensor's own length masks
both sides.

Rules the builders and ``make_transform_graph`` enforce:

- Positional calls. Coordinate names appear only while BUILDING a graph and when reading it back
  (``resolve``'s keys, ``edge``). Every call that supplies coordinates takes a tuple in
  ``inputs(...)`` order: graph input ``i`` is position ``i``. ``inputs(m=.., n=..)`` and
  ``inputs(n=.., m=..)`` are different graphs, as two functions with swapped parameters are.
- Single writer. Each coordinate is written once; a step never rewrites its own read (write
  ``offset("h", into="h_halo", ...)``, never ``offset("h", into="h", ...)``).
- Canonical order. Nodes are stored in topological order with ties broken by
  ``(kind, read, write)``, so two graphs that list the same steps in a different order are equal.
  The tie key is unique per node: written names are unique, and a repeated step (the only way two
  ``broadcast`` steps, which write nothing, could tie) is rejected.
- Flow-correct names: ``merge`` is N -> 1 (combine), ``unmerge`` is 1 -> N (split). CK and the
  V4 transform graph use the opposite names (their MERGE is this ``unmerge``).

``evaluate`` and ``resolve`` compute in pure Python what every lane computes, valid or not: an
``unmerge`` read past its domain drives the outer piece past its length (it is not wrapped), and
a ``merge`` read past its length is not reduced. Which of those values need a run-time check is
decided elsewhere, from value ranges against these domains.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ranges import RuntimeValue

__all__ = [
    "BroadcastParams",
    "Edge",
    "EmbedParams",
    "FreezeParams",
    "InputsDecl",
    "MergeParams",
    "Node",
    "OffsetParams",
    "OutputsDecl",
    "SliceParams",
    "TransformGraph",
    "TransformType",
    "UnmergeParams",
    "XorParams",
    "broadcast",
    "embed",
    "freeze",
    "identity_graph",
    "inputs",
    "make_transform_graph",
    "merge",
    "offset",
    "outputs",
    "slice_window",
    "unmerge",
    "xor",
]


class TransformType(Enum):
    """The eight node kinds. The declaration order is the kind part of the canonical tie key."""

    EMBED = 1
    MERGE = 2
    UNMERGE = 3
    OFFSET = 4
    SLICE = 5
    FREEZE = 6
    XOR = 7
    BROADCAST = 8


# The builder that makes each kind: the name an author wrote, used in every message.
_BUILDER_NAME = {
    TransformType.EMBED: "embed",
    TransformType.MERGE: "merge",
    TransformType.UNMERGE: "unmerge",
    TransformType.OFFSET: "offset",
    TransformType.SLICE: "slice_window",
    TransformType.FREEZE: "freeze",
    TransformType.XOR: "xor",
    TransformType.BROADCAST: "broadcast",
}


@dataclass(frozen=True)
class EmbedParams:
    """``into = sum(read_i * coeffs[i])``; coefficients are >= 0."""

    coeffs: tuple[int, ...]


@dataclass(frozen=True)
class MergeParams:
    """Mixed-radix combine; read ``i`` must lie in ``[0, lengths[i])``."""

    lengths: tuple[int, ...]


@dataclass(frozen=True)
class UnmergeParams:
    """Split by divide / remainder; piece ``i`` has domain ``[0, lengths[i])``."""

    lengths: tuple[int, ...]


@dataclass(frozen=True)
class OffsetParams:
    """``into = read + shift``; a negative shift is a halo."""

    shift: int


@dataclass(frozen=True)
class SliceParams:
    """``into = read + begin``; the read must lie in ``[0, end - begin)``."""

    begin: int
    end: int


@dataclass(frozen=True)
class FreezeParams:
    """``into = value`` (no read)."""

    value: int


@dataclass(frozen=True)
class XorParams:
    """The XOR swizzle has no parameters: its mask comes from ``reads[1]``'s domain."""


@dataclass(frozen=True)
class BroadcastParams:
    """BROADCAST has no parameters: it reads coordinates that address nothing."""


_Params = (
    EmbedParams
    | MergeParams
    | UnmergeParams
    | OffsetParams
    | SliceParams
    | FreezeParams
    | XorParams
    | BroadcastParams
)


@dataclass(frozen=True)
class Node:
    """One step of a graph: a kind, the coordinates it reads and writes, and its parameters.

    Built by the kind builders (``embed``, ``merge``, ...), never by hand.
    """

    kind: TransformType
    read: tuple[str, ...]
    write: tuple[str, ...]
    params: _Params

    def __str__(self) -> str:
        reads = ", ".join(f"'{r}'" for r in self.read) or "()"
        writes = ", ".join(f"'{w}'" for w in self.write) or "()"
        return f"{_BUILDER_NAME[self.kind]}({reads} -> {writes})"

    def _tie_key(self) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
        return (self.kind.value, self.read, self.write)


@dataclass(frozen=True)
class Edge:
    """One named coordinate of a graph.

    ``domain`` is the declared extent ``[0, domain)``, or ``None`` when the coordinate has none:
    an input over a runtime-length axis, or the output of ``embed`` / ``offset`` /
    ``slice_window`` / ``freeze`` (those have a range, not a domain), or the first output of
    ``xor``. ``source`` says in plain words what writes the coordinate: ``"inputs(...)"`` for an
    input, otherwise the writing step (for example ``"unmerge('k' -> 'y', 'x', 'c')"``).
    """

    name: str
    domain: int | None
    source: str


@dataclass(frozen=True)
class InputsDecl:
    """The ordered input declaration ``inputs(...)`` returns; the graph's call signature."""

    domains: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True)
class OutputsDecl:
    """The ordered output declaration ``outputs(...)`` returns (tensor-axis order)."""

    names: tuple[str, ...]


def _require_lengths_at_least_one(kind: str, lengths: tuple[int, ...]) -> None:
    if any(length < 1 for length in lengths):
        raise ValueError(f"{kind} lengths must be >= 1, got {lengths}")


def _require_new_names(
    kind: str, reads: tuple[str, ...], writes: tuple[str, ...]
) -> None:
    for name in writes:
        if name in reads:
            raise ValueError(
                f"{kind} into='{name}' rewrites its own read -- give the step a new output "
                "name (each coordinate is written once)"
            )


def inputs(**name_to_length: int | None) -> InputsDecl:
    """Declare the graph's input coordinates, in order: input ``i`` is call position ``i``.

    A length ``n`` gives the input the domain ``[0, n)``; ``None`` declares an input over a
    runtime-length axis, which has no domain and is bounded only by its tensor.
    """
    lengths = tuple(v for v in name_to_length.values() if v is not None)
    _require_lengths_at_least_one("inputs", lengths)
    return InputsDecl(tuple(name_to_length.items()))


def outputs(*names: str) -> OutputsDecl:
    """Declare the graph's output coordinates, in tensor-axis order."""
    return OutputsDecl(tuple(names))


def embed(reads: tuple[str, ...], into: str, coeffs: tuple[int, ...]) -> Node:
    """``into = sum(reads[i] * coeffs[i])``: arbitrary coefficients (for example stride, dilation).

    Reach for ``embed`` when the weights are not the radix weights of the reads' lengths (that is
    ``merge``). Coefficients must be >= 0: a negative one would make the per-lane address part
    negative; express a reversed axis through the tensor's strides instead.
    """
    reads, coeffs = tuple(reads), tuple(coeffs)
    if len(reads) != len(coeffs):
        raise ValueError(
            f"embed reads {len(reads)} coordinates but has {len(coeffs)} coeffs -- give one "
            "coefficient per read"
        )
    for read, coeff in zip(reads, coeffs):
        if coeff < 0:
            raise ValueError(
                f"embed coefficient {coeff} on read '{read}' is negative -- reversed axes are "
                "not supported; reorder the tensor's strides instead"
            )
    _require_new_names("embed", reads, (into,))
    return Node(TransformType.EMBED, reads, (into,), EmbedParams(coeffs))


def merge(reads: tuple[str, ...], into: str, lengths: tuple[int, ...]) -> Node:
    """Combine ``reads`` mixed-radix into one coordinate, the last read innermost.

    Read ``i``'s weight is the product of the later ``lengths``; read ``i`` must lie in
    ``[0, lengths[i])``. ``into`` gets the domain ``[0, prod(lengths))``. CK and V4 call this
    UNMERGE.
    """
    reads, lengths = tuple(reads), tuple(lengths)
    if len(reads) != len(lengths):
        raise ValueError(
            f"merge reads {len(reads)} coordinates but has {len(lengths)} lengths -- give one "
            "length per read"
        )
    _require_lengths_at_least_one("merge", lengths)
    _require_new_names("merge", reads, (into,))
    return Node(TransformType.MERGE, reads, (into,), MergeParams(lengths))


def unmerge(read: str, into: tuple[str, ...], lengths: tuple[int, ...]) -> Node:
    """Split one coordinate into ``len(into)`` pieces by divide / remainder, the last innermost.

    Piece ``i`` gets the domain ``[0, lengths[i])``; ``read`` must lie in ``[0, prod(lengths))``.
    A read past that drives the outer piece past its length (it is not wrapped). CK and V4 call
    this MERGE.
    """
    into, lengths = tuple(into), tuple(lengths)
    if len(into) != len(lengths):
        raise ValueError(
            f"unmerge writes {len(into)} coordinates but has {len(lengths)} lengths -- give one "
            "length per written coordinate"
        )
    _require_lengths_at_least_one("unmerge", lengths)
    _require_new_names("unmerge", (read,), into)
    return Node(TransformType.UNMERGE, (read,), into, UnmergeParams(lengths))


def offset(read: str, into: str, shift: int) -> Node:
    """``into = read + shift``. A negative shift is a halo (the CK ``pad`` left side)."""
    _require_new_names("offset", (read,), (into,))
    return Node(TransformType.OFFSET, (read,), (into,), OffsetParams(shift))


def slice_window(read: str, into: str, begin: int, end: int) -> Node:
    """``into = read + begin``: select the window ``[begin, end)``; ``read`` must lie in
    ``[0, end - begin)``. (Named ``slice_window`` because ``slice`` is a Python builtin.)
    """
    if not begin < end:
        raise ValueError(
            f"slice_window needs begin < end, got begin={begin}, end={end}"
        )
    _require_new_names("slice_window", (read,), (into,))
    return Node(TransformType.SLICE, (read,), (into,), SliceParams(begin, end))


def freeze(into: str, value: int) -> Node:
    """``into = value``: a coordinate pinned to one value (it reads nothing)."""
    return Node(TransformType.FREEZE, (), (into,), FreezeParams(value))


def xor(reads: tuple[str, str], into: tuple[str, str]) -> Node:
    """The XOR swizzle: ``into[0] = reads[0]``; ``into[1] = reads[1] xor (reads[0] & (len - 1))``.

    ``len`` is ``reads[1]``'s declared domain and must be a power of two: that is what makes the
    swizzle a bijection on ``[0, len)``. ``into[1]`` carries that domain.
    """
    reads, into = tuple(reads), tuple(into)
    if len(reads) != 2 or len(into) != 2:
        raise ValueError(
            f"xor needs exactly two reads and two outputs, got reads={reads}, into={into}"
        )
    _require_new_names("xor", reads, into)
    return Node(TransformType.XOR, reads, into, XorParams())


def broadcast(reads: tuple[str, ...]) -> Node:
    """Read coordinates that address nothing (for example a bias vector broadcast over rows)."""
    return Node(TransformType.BROADCAST, tuple(reads), (), BroadcastParams())


def _canonical_order(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    """Topological order, ties broken by ``(kind, read, write)``; raises on a cycle."""
    writer = {name: node for node in nodes for name in node.write}
    deps = {node: {writer[r] for r in node.read if r in writer} for node in nodes}
    users: dict[Node, list[Node]] = {node: [] for node in nodes}
    for node, needs in deps.items():
        for need in needs:
            users[need].append(node)
    waiting = {node: len(needs) for node, needs in deps.items()}
    ready = [(node._tie_key(), node) for node in nodes if waiting[node] == 0]
    heapq.heapify(ready)
    order: list[Node] = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for user in users[node]:
            waiting[user] -= 1
            if waiting[user] == 0:
                heapq.heappush(ready, (user._tie_key(), user))
    if len(order) != len(nodes):
        raise ValueError(
            f"the steps form a cycle through {_cycle_names(deps, set(order))} -- a coordinate "
            "cannot depend on itself"
        )
    return tuple(order)


def _cycle_names(deps: dict[Node, set[Node]], placed: set[Node]) -> str:
    """Name the coordinates of one cycle among the steps the topological order could not place."""
    start = next(node for node in sorted(deps, key=Node._tie_key) if node not in placed)
    path: list[Node] = []
    node = start
    while node not in path:
        path.append(node)
        node = min((d for d in deps[node] if d not in placed), key=Node._tie_key)
    cycle = path[path.index(node) :]
    return ", ".join(f"'{name}'" for step in cycle for name in step.write)


def _edge_table(
    input_domains: tuple[tuple[str, int | None], ...], order: tuple[Node, ...]
) -> dict[str, Edge]:
    """Every coordinate's ``Edge``, checking each restated or required domain as it is reached."""
    edges = {name: Edge(name, domain, "inputs(...)") for name, domain in input_domains}
    for node in order:
        source = str(node)
        kind = node.kind
        if kind is TransformType.UNMERGE:
            read = node.read[0]
            lengths = node.params.lengths
            domain = edges[read].domain
            if domain is not None and prod(lengths) != domain:
                raise ValueError(
                    f"unmerge of '{read}' declares lengths {lengths} (product {prod(lengths)}) "
                    f"but '{read}' has domain {domain} -- the pieces must multiply to the "
                    "read's domain"
                )
            for name, length in zip(node.write, lengths):
                edges[name] = Edge(name, length, source)
        elif kind is TransformType.MERGE:
            lengths = node.params.lengths
            domains = tuple(edges[r].domain for r in node.read)
            if any(
                d is not None and d != length for d, length in zip(domains, lengths)
            ):
                raise ValueError(
                    f"merge into '{node.write[0]}' declares lengths {lengths} but its reads have "
                    f"domains {domains} -- restated lengths must match"
                )
            edges[node.write[0]] = Edge(node.write[0], prod(lengths), source)
        elif kind is TransformType.XOR:
            r1 = node.read[1]
            domain = edges[r1].domain
            if domain is None:
                raise ValueError(
                    f"xor reads '{r1}', which has no declared domain -- xor needs a "
                    "power-of-two domain; split it with unmerge first"
                )
            if domain & (domain - 1):
                raise ValueError(
                    f"xor reads '{r1}' whose domain is {domain}; xor needs a power-of-two "
                    "domain to be a bijection"
                )
            edges[node.write[0]] = Edge(node.write[0], None, source)
            edges[node.write[1]] = Edge(node.write[1], domain, source)
        else:
            for name in node.write:
                edges[name] = Edge(name, None, source)
    return edges


def _assemble(
    input_domains: tuple[tuple[str, int | None], ...],
    nodes: tuple[Node, ...],
    output_names: tuple[str, ...],
) -> tuple[tuple[Node, ...], dict[str, Edge]]:
    """Check a graph's declarations and return its canonical nodes and its edge table."""
    input_names = tuple(name for name, _ in input_domains)
    seen: set[Node] = set()
    for node in nodes:
        if node in seen:
            raise ValueError(f"duplicate step {node} -- remove the repeat")
        seen.add(node)

    written_by: dict[str, str] = {name: "inputs(...)" for name in input_names}
    for node in nodes:
        for name in node.write:
            if name in written_by:
                raise ValueError(
                    f"coordinate '{name}' is written by two steps ({written_by[name]}, {node}) "
                    "-- give one of them a new output name"
                )
            written_by[name] = str(node)

    for node in nodes:
        for name in node.read:
            if name not in written_by:
                raise ValueError(
                    f"coordinate '{name}' is read by {node} but never declared or written -- "
                    "declare it in inputs(...) or write it in an earlier step"
                )

    order = _canonical_order(nodes)
    edges = _edge_table(input_domains, order)

    for name in output_names:
        if name not in written_by:
            raise ValueError(f"output '{name}' is never written or declared")

    read_somewhere = {name for node in nodes for name in node.read} | set(output_names)
    for name in input_names:
        if name not in read_somewhere:
            raise ValueError(
                f"input '{name}' is never read -- remove it or use broadcast to consume it"
            )
    return order, edges


def _evaluate_node(node: Node, env: dict[str, int], edges: Mapping[str, Edge]) -> None:
    """Write ``node``'s outputs into ``env``: the value every lane computes, valid or not."""
    vals = tuple(env[r] for r in node.read)
    kind, params = node.kind, node.params
    if kind is TransformType.EMBED:
        env[node.write[0]] = sum(v * c for v, c in zip(vals, params.coeffs))
    elif kind is TransformType.MERGE:
        total = 0
        for v, length in zip(vals, params.lengths):
            total = total * length + v
        env[node.write[0]] = total
    elif kind is TransformType.UNMERGE:
        rest = vals[0]
        # Innermost piece first; the outer piece keeps the quotient unwrapped.
        for name, length in zip(node.write[:0:-1], params.lengths[:0:-1]):
            rest, env[name] = divmod(rest, length)
        env[node.write[0]] = rest
    elif kind is TransformType.OFFSET:
        env[node.write[0]] = vals[0] + params.shift
    elif kind is TransformType.SLICE:
        env[node.write[0]] = vals[0] + params.begin
    elif kind is TransformType.FREEZE:
        env[node.write[0]] = params.value
    elif kind is TransformType.XOR:
        mask = edges[node.write[1]].domain - 1
        env[node.write[0]] = vals[0]
        env[node.write[1]] = vals[1] ^ (vals[0] & mask)
    # BROADCAST writes nothing.


@dataclass(frozen=True, eq=False)
class TransformGraph:
    """A frozen map from named input coordinates to named output coordinates.

    Built by ``make_transform_graph``. ``inputs`` keeps its declaration order (input ``i`` is
    call position ``i``); ``outputs`` is in tensor-axis order; ``nodes`` is in canonical order.
    Two graphs are equal when their ordered inputs, canonical nodes and ordered outputs are equal.
    The graph holds no tensor and no bound.
    """

    inputs: Mapping[str, int | None]
    outputs: tuple[str, ...]
    nodes: tuple[Node, ...]
    _edges: Mapping[str, Edge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        input_domains = tuple(self.inputs.items())
        order, edges = _assemble(input_domains, tuple(self.nodes), tuple(self.outputs))
        object.__setattr__(self, "inputs", MappingProxyType(dict(input_domains)))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "nodes", order)
        object.__setattr__(self, "_edges", MappingProxyType(edges))

    def _signature(self) -> tuple:
        return (tuple(self.inputs.items()), self.nodes, self.outputs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransformGraph):
            return NotImplemented
        return self._signature() == other._signature()

    def __hash__(self) -> int:
        return hash(self._signature())

    def edge(self, name: str) -> Edge:
        """The ``Edge`` of a coordinate the graph was built with (its domain and its writer)."""
        if name not in self._edges:
            raise ValueError(
                f"'{name}' is not a coordinate of this graph -- its coordinates are "
                f"{tuple(self._edges)}"
            )
        return self._edges[name]

    def evaluate(self, coords: tuple[int, ...]) -> tuple[int, ...]:
        """The output coordinates for ``coords`` (one int per input, in ``inputs(...)`` order).

        Pure Python over ints; returns a tuple in ``outputs(...)`` order.
        """
        env = self._run("evaluate", coords)
        return tuple(env[name] for name in self.outputs)

    def resolve(self, coords: tuple[int, ...]) -> dict[str, int]:
        """Every named coordinate, intermediates included, for ``coords`` (in ``inputs(...)``
        order), keyed by the names the graph was built with."""
        return self._run("resolve", coords)

    def _run(self, caller: str, coords: tuple[int, ...]) -> dict[str, int]:
        _require_input_coordinates(self, caller, coords)
        env = dict(zip(self.inputs, coords))
        for node in self.nodes:
            _evaluate_node(node, env, self._edges)
        return env


def _require_input_coordinates(
    graph: TransformGraph, caller: str, coords: object
) -> None:
    """Refuse anything but a tuple of one coordinate per graph input (the positional rule).

    ``caller`` is the public function the author called, so the message names it.
    """
    if not isinstance(coords, tuple):
        raise TypeError(
            f"{caller} takes the coordinates as a tuple in inputs(...) order, got "
            f"{type(coords).__name__} -- coordinates are positional"
        )
    names = tuple(graph.inputs)
    if len(coords) != len(names):
        raise ValueError(
            f"{caller} got {len(coords)} coordinates, the graph has {len(names)} inputs "
            f"{names} -- give one int per input, in inputs(...) order"
        )


def _require_outputs_match_rank(graph: TransformGraph, rank: int) -> None:
    """Refuse binding ``graph`` to a tensor whose rank is not its number of outputs.

    Output ``i`` is tensor axis ``i``, so the two counts must agree.
    """
    if len(graph.outputs) != rank:
        raise ValueError(
            f"graph has {len(graph.outputs)} outputs {graph.outputs} but the tensor has rank "
            f"{rank}"
        )


def make_transform_graph(
    inputs: InputsDecl, *steps_then_outputs: Node | OutputsDecl
) -> TransformGraph:
    """Assemble and check a ``TransformGraph``: ``inputs(...)``, the steps, then ``outputs(...)``.

    Checked here (the builders check only their own arguments): each coordinate written once, no
    repeated step, every read declared or written, no cycle, every restated domain agreeing
    (``unmerge`` pieces multiply to the read's domain, ``merge`` lengths match its reads'
    domains), ``xor``'s second read has a power-of-two domain, every output written, every input
    read.
    """
    last = steps_then_outputs[-1] if steps_then_outputs else None
    if not isinstance(inputs, InputsDecl) or not isinstance(last, OutputsDecl):
        # A misplaced declaration, not a wrong type: the message says where each one goes.
        raise ValueError(  # noqa: TRY004
            "make_transform_graph needs inputs(...) first and outputs(...) last, got "
            f"{_describe_arg(inputs)} ... {_describe_arg(last)}"
        )
    steps = steps_then_outputs[:-1]
    for step in steps:
        if not isinstance(step, Node):
            raise TypeError(
                f"make_transform_graph got {_describe_arg(step)} between inputs(...) and "
                "outputs(...) -- every step there is what a kind builder returns (embed, "
                "merge, unmerge, ...)"
            )
    return TransformGraph(dict(inputs.domains), last.names, tuple(steps))


def _describe_arg(arg: object) -> str:
    if isinstance(arg, InputsDecl):
        return "inputs(...)"
    if isinstance(arg, OutputsDecl):
        return "outputs(...)"
    if isinstance(arg, Node):
        return str(arg)
    if arg is None:
        return "nothing"
    return type(arg).__name__


def identity_graph(lengths: tuple[int | RuntimeValue | None, ...]) -> TransformGraph:
    """The graph of a plain window over a tensor with these lengths: outputs = inputs.

    Inputs and outputs are named ``axis0``, ``axis1``, ... in tensor-axis order. An int length
    gives its input the domain ``[0, length)``; a runtime length (or ``None``) gives an input with
    no domain, bounded only by the tensor. A plain window and a graph window then have their
    checks decided by the same rule.
    """
    names = tuple(f"axis{i}" for i in range(len(lengths)))
    domains = {
        name: length if isinstance(length, int) else None
        for name, length in zip(names, lengths)
    }
    return TransformGraph(domains, names, ())
