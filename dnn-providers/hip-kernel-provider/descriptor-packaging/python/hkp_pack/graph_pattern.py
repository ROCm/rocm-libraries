"""Pack-time validation of a UED's declarative `graph_match.nodes` pattern.

Applies the checks compileGraphPattern (CompiledGraphPattern.hpp) runs when
the runtime loads a pattern (RFC 0020 §4.3). A pattern the runtime would refuse
then fails the pack instead of dropping the engine at load. There are two
stages, in the runtime's order:

- Well-formedness, checked from the block alone.
- Resolution against the op-schema registry, which is generated from the
  annotated FlatBuffers schemas.

The runtime is authoritative. The limit and reserved names below copy its
constants, and every error carries the runtime's wording.
"""

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import HkpPackError, OpSchemaRegistryError
from .json_rules import (
    DuplicateKeyError,
    NonStandardLiteralError,
    UnrepresentableValueError,
    loads_strict,
    require_known_keys,
)

MAX_PATTERN_NODES = 32

# Keys a pattern node may carry, and the keys it must carry.
NODE_KEYS = ("kind", "id", "op", "operands", "results")
NODE_REQUIRED_KEYS = ("kind", "id", "op")

# Symbol roots the runtime publishes. Node ids and pattern variables cannot use
# them (RFC 0020 §6.1).
RESERVED_ROOTS = ("graph", "kernel", "device")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Environment override for the registry path. CMake sets it for tests and the
# configure probe. The pack step passes the path as an argument.
OP_SCHEMA_REGISTRY_ENV = "HKP_OP_SCHEMA_REGISTRY"

# Set by CMake, with a reason, when the build has no registry to offer. A pattern
# that needs the registry then fails with that reason instead of resolving against
# the repository copy, which belongs to a different hipDNN than the one the build
# links.
OP_SCHEMA_REGISTRY_UNAVAILABLE_ENV = "HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE"

# The committed registry. This package sits at
# <repo>/dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack.
DEFAULT_OP_SCHEMA_REGISTRY = (
    Path(__file__).resolve().parents[5]
    / "projects"
    / "hipdnn"
    / "flatbuffers_sdk"
    / "op_schema_registry.json"
)


@dataclass(frozen=True)
class OpSchema:
    """One registry entry. Each edge name maps to True if the edge is optional."""

    opcode: str
    operands: dict
    results: dict


@dataclass(frozen=True)
class _Edge:
    name: str
    variable: str
    optional: bool


@dataclass(frozen=True)
class _Node:
    id: str
    opcodes: tuple
    operands: tuple
    results: tuple


def resolve_op_schema_registry(path=None):
    """Return the registry path from the argument, the environment, or the repo copy.

    An empty argument or environment value counts as absent, so
    `hkp_pack.py --op-schema-registry ""` gets the default. When neither names a
    path and OP_SCHEMA_REGISTRY_UNAVAILABLE_ENV carries a reason, raise
    OpSchemaRegistryError with it.
    """
    if path:
        return Path(path)
    env = os.environ.get(OP_SCHEMA_REGISTRY_ENV)
    if env:
        return Path(env)
    unavailable = os.environ.get(OP_SCHEMA_REGISTRY_UNAVAILABLE_ENV)
    if unavailable:
        raise OpSchemaRegistryError(unavailable)
    return DEFAULT_OP_SCHEMA_REGISTRY


def _registry_edges(entries, path, opcode, direction):
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("optional"), bool)
        for entry in entries
    ):
        raise OpSchemaRegistryError(
            f"malformed op-schema registry {path}: op '{opcode}' {direction} must "
            "be an array of objects with a string 'name' and a bool 'optional'"
        )
    return {entry["name"]: entry["optional"] for entry in entries}


@functools.lru_cache(maxsize=None)
def _load_op_schema_registry(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OpSchemaRegistryError(
            f"cannot read op-schema registry {path}: {exc}"
        ) from exc
    try:
        doc = loads_strict(text)
    except (
        DuplicateKeyError,
        NonStandardLiteralError,
        UnrepresentableValueError,
        json.JSONDecodeError,
    ) as exc:
        raise OpSchemaRegistryError(
            f"malformed op-schema registry {path}: {exc}"
        ) from exc
    ops = doc.get("ops") if isinstance(doc, dict) else None
    if not isinstance(ops, dict):
        raise OpSchemaRegistryError(
            f"malformed op-schema registry {path}: 'ops' must be an object"
        )
    registry = {}
    for opcode, entry in ops.items():
        if not isinstance(entry, dict):
            raise OpSchemaRegistryError(
                f"malformed op-schema registry {path}: op '{opcode}' must be an object"
            )
        registry[opcode] = OpSchema(
            opcode=opcode,
            operands=_registry_edges(entry.get("operands"), path, opcode, "operands"),
            results=_registry_edges(entry.get("results"), path, opcode, "results"),
        )
    return registry


def load_op_schema_registry(path):
    """Return {opcode: OpSchema} from the registry at path. Each path is read once."""
    return _load_op_schema_registry(str(path))


def _dump(value):
    """Format a value as nlohmann::json::dump() does, for error messages."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _node_locator(node_id, where):
    return f"node '{node_id}' in {where}"


def _parse_opcodes(op, locator):
    if isinstance(op, str):
        if not op:
            raise HkpPackError(f"key 'op' of {locator} must not be empty")
        return (op,)
    if not isinstance(op, dict):
        raise HkpPackError(
            f"key 'op' of {locator} must be an opcode string or an object "
            '{"one_of": [opcodes]}'
        )
    set_locator = f"key 'op' of {locator}"
    require_known_keys(op, ("one_of",), set_locator)
    if "one_of" not in op:
        raise HkpPackError(f"missing required key 'one_of' in {set_locator}")
    members = op["one_of"]
    if not isinstance(members, list):
        raise HkpPackError(
            f"key 'one_of' in {set_locator} must be an array of opcode strings"
        )
    if len(members) < 2:
        raise HkpPackError(
            f"key 'one_of' in {set_locator} lists {len(members)} opcode(s); an "
            "opcode set needs at least 2"
        )
    opcodes = []
    seen = set()
    for member in members:
        if not isinstance(member, str) or not member:
            raise HkpPackError(
                f"key 'one_of' in {set_locator} must list non-empty opcode strings"
            )
        if member in seen:
            raise HkpPackError(
                f"opcode '{member}' is listed twice in key 'one_of' in {set_locator}"
            )
        seen.add(member)
        opcodes.append(member)
    return tuple(opcodes)


def _parse_binding(name, value, operand, edge_locator):
    """Parse `"$name"`, or `"$name?"` on an operand."""
    if not isinstance(value, str):
        raise HkpPackError(f"{edge_locator} must be a string binding '$<identifier>'")
    body = value
    optional = body.endswith("?")
    if optional:
        if not operand:
            raise HkpPackError(
                f"{edge_locator} has binding '{value}'; '?' is only allowed on an "
                "operand binding"
            )
        body = body[:-1]
    if not body.startswith("$") or not _IDENT_RE.fullmatch(body[1:]):
        expected = (
            "'$<identifier>' or '$<identifier>?'" if operand else "'$<identifier>'"
        )
        raise HkpPackError(
            f"{edge_locator} has malformed binding '{value}'; expected {expected}"
        )
    return _Edge(name=name, variable=body[1:], optional=optional)


def _parse_edges(node, key, locator):
    """Parse the `operands` or `results` map.

    Edges are checked and kept in name order, the order the runtime reads them, so
    a map with several faults reports the same one on both sides.
    """
    edges_doc = node.get(key, {})
    if not isinstance(edges_doc, dict):
        raise HkpPackError(
            f"key '{key}' of {locator} must be an object mapping edge names to bindings"
        )
    operand = key == "operands"
    direction = "operand" if operand else "result"
    return tuple(
        _parse_binding(name, value, operand, f"{direction} '{name}' of {locator}")
        for name, value in sorted(edges_doc.items())
    )


def _parse_node(index, node, where):
    """Parse one node on its own. Checks that span nodes wait for _check_variables."""
    position = f"graph_match.nodes[{index}] in {where}"
    if not isinstance(node, dict):
        raise HkpPackError(f"{position} must be a JSON object")
    require_known_keys(node, NODE_KEYS, position)
    for key in NODE_REQUIRED_KEYS:
        if key not in node:
            raise HkpPackError(f"missing required key '{key}' in {position}")
    node_id = node["id"]
    if not isinstance(node_id, str) or not _IDENT_RE.fullmatch(node_id):
        raise HkpPackError(
            f"key 'id' in {position} must be an identifier "
            f"([A-Za-z_][A-Za-z0-9_]*), got {_dump(node_id)}"
        )
    locator = _node_locator(node_id, where)
    if node_id in RESERVED_ROOTS:
        raise HkpPackError(f"{locator}: node id '{node_id}' is a reserved root")
    if node["kind"] != "op":
        raise HkpPackError(
            f"key 'kind' of {locator} must be \"op\", got {_dump(node['kind'])}"
        )
    return _Node(
        id=node_id,
        opcodes=_parse_opcodes(node["op"], locator),
        operands=_parse_edges(node, "operands", locator),
        results=_parse_edges(node, "results", locator),
    )


def _check_variable_name(node, edge, node_ids, where):
    if edge.variable in RESERVED_ROOTS:
        raise HkpPackError(
            f"{_node_locator(node.id, where)}: variable '${edge.variable}' is a "
            "reserved root"
        )
    if edge.variable in node_ids:
        raise HkpPackError(
            f"{_node_locator(node.id, where)}: variable '${edge.variable}' collides "
            f"with node id '{edge.variable}'"
        )


def _check_variables(nodes, where):
    """Check node ids and variable bindings across the whole pattern.

    Node ids are unique. A variable is neither a reserved root nor a node id,
    and at most one result binds it. A variable no result binds is a graph
    input, so exactly one operand binds it. An operand that binds a produced
    variable cannot be optional. The checks run in the runtime's order: node
    ids, then every node's results, then every node's operands.
    """
    node_ids = set()
    for node in nodes:
        if node.id in node_ids:
            raise HkpPackError(f"node id '{node.id}' is used by two nodes in {where}")
        node_ids.add(node.id)
    producer = {}
    for node in nodes:
        for edge in node.results:
            _check_variable_name(node, edge, node_ids, where)
            site = f"node '{node.id}' result '{edge.name}'"
            if edge.variable in producer:
                raise HkpPackError(
                    f"variable '${edge.variable}' is bound by two results in {where}: "
                    f"{producer[edge.variable]} and {site}"
                )
            producer[edge.variable] = site
    graph_input = {}
    for node in nodes:
        for edge in node.operands:
            _check_variable_name(node, edge, node_ids, where)
            if edge.variable in producer:
                if edge.optional:
                    raise HkpPackError(
                        f"{_node_locator(node.id, where)}: optional binding "
                        f"'${edge.variable}?' on operand '{edge.name}' names a "
                        f"variable {producer[edge.variable]} produces; only a graph "
                        "input may be optional"
                    )
                continue
            site = f"node '{node.id}' operand '{edge.name}'"
            if edge.variable in graph_input:
                raise HkpPackError(
                    f"graph input variable '${edge.variable}' is bound by two "
                    f"operands in {where}: {graph_input[edge.variable]} and {site}; "
                    "a variable no node produces is bound by exactly one operand"
                )
            graph_input[edge.variable] = site


def _check_acyclic(nodes, where):
    """Reject nodes that depend on each other in a cycle.

    A node depends on the producer of each variable it reads. hipDNN graphs are
    acyclic, so a cyclic pattern matches no graph. A node that reads its own
    result is a cycle of one. Runs after _check_variables, which guarantees each
    variable has at most one producer.
    """
    producer_of = {}
    for index, node in enumerate(nodes):
        for edge in node.results:
            producer_of.setdefault(edge.variable, index)
    producers = [
        [
            producer_of[edge.variable]
            for edge in node.operands
            if edge.variable in producer_of
        ]
        for node in nodes
    ]

    # Place every node whose producers are all placed. What remains is the
    # cycles and the nodes downstream of them.
    placed = [False] * len(nodes)
    progress = True
    while progress:
        progress = False
        for index, node_producers in enumerate(producers):
            if not placed[index] and all(placed[p] for p in node_producers):
                placed[index] = True
                progress = True
    if all(placed):
        return

    # Every unplaced node has an unplaced producer, so following producers from
    # any one of them must revisit a node. The walk from that node on is a cycle.
    walk = []
    visited = set()
    current = placed.index(False)
    while current not in visited:
        visited.add(current)
        walk.append(current)
        current = next(p for p in producers[current] if not placed[p])

    # The walk runs from reader to producer. Reverse it so each node reads a
    # result of the one before it, and close the loop on the first node.
    cycle = list(reversed(walk[walk.index(current) :]))
    path = " -> ".join(f"'{nodes[i].id}'" for i in cycle + [cycle[0]])
    raise HkpPackError(
        f"pattern nodes {path} in {where} depend on each other in a cycle; no "
        "hipDNN graph is cyclic, so the pattern could never match"
    )


def _optionality(optional):
    return "optional" if optional else "required"


def _resolve_edge(node, edge, direction, registry, locator):
    """Check an edge against each opcode in turn, as the runtime does.

    Each opcode must declare the edge in this direction, must declare an operand
    bound with `?` optional, and must agree with the first opcode on its
    optionality.
    """
    if direction == "operand":
        other_kind, other_section = "a result", "results"
    else:
        other_kind, other_section = "an operand", "operands"
    first = None
    for opcode in node.opcodes:
        schema = registry[opcode]
        declared, other = (
            (schema.operands, schema.results)
            if direction == "operand"
            else (schema.results, schema.operands)
        )
        if edge.name not in declared:
            message = (
                f"{locator}: opcode '{opcode}' declares no {direction} '{edge.name}'"
            )
            if edge.name in other:
                message += (
                    f"; it is {other_kind}, so it belongs under '{other_section}'"
                )
            raise HkpPackError(message)
        optional = declared[edge.name]
        if edge.optional and not optional:
            raise HkpPackError(
                f"{locator}: operand '{edge.name}' of opcode '{opcode}' is required, "
                f"so binding '${edge.variable}?' cannot be optional"
            )
        if first is None:
            first = (opcode, optional)
        elif optional != first[1]:
            raise HkpPackError(
                f"{locator}: one_of members disagree on {direction} '{edge.name}': "
                f"opcode '{first[0]}' declares it {_optionality(first[1])}, opcode "
                f"'{opcode}' declares it {_optionality(optional)}"
            )


def _resolve(nodes, registry, where):
    for node in nodes:
        locator = _node_locator(node.id, where)
        for opcode in node.opcodes:
            if opcode not in registry:
                raise HkpPackError(
                    f"{locator}: opcode '{opcode}' is not in the op-schema registry"
                )
        for edge in node.operands:
            _resolve_edge(node, edge, "operand", registry, locator)
        for edge in node.results:
            _resolve_edge(node, edge, "result", registry, locator)


def validate_graph_pattern(nodes, where, registry_path=None):
    """Validate a `graph_match.nodes` block as the runtime loads it.

    `where` names the UED in error messages. `registry_path` overrides the
    registry location (see resolve_op_schema_registry). The registry is read
    only after the well-formedness checks pass. An unreadable registry raises an
    error that names its path.
    """
    block = f"graph_match.nodes in {where}"
    if not isinstance(nodes, list):
        raise HkpPackError(f"{block} must be an array of node objects")
    if not nodes:
        raise HkpPackError(f"{block} must not be empty")
    if len(nodes) > MAX_PATTERN_NODES:
        raise HkpPackError(
            f"{block} has {len(nodes)} nodes; the limit is {MAX_PATTERN_NODES}"
        )
    parsed = [_parse_node(index, node, where) for index, node in enumerate(nodes)]
    _check_variables(parsed, where)
    _check_acyclic(parsed, where)
    registry = load_op_schema_registry(resolve_op_schema_registry(registry_path))
    _resolve(parsed, registry, where)
