"""Pack-time validation of a UED's declarative `graph_match.nodes` pattern.

Applies the checks the runtime runs when it loads a pattern (RFC 0020 §4.3).
A pattern the runtime would refuse then fails the pack instead of dropping the
engine at load. There are two stages:

- Well-formedness, checked from the block alone. Mirrors parseGraphPattern in
  GraphPattern.hpp.
- Resolution against the op-schema registry, which is generated from the
  annotated FlatBuffers schemas. Mirrors compileGraphPattern in
  CompiledGraphPattern.hpp.

The runtime is authoritative. The limits and reserved names below copy its
constants.
"""

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import HkpPackError
from .json_rules import DuplicateKeyError, loads_strict, require_known_keys

MAX_PATTERN_NODES = 32
MAX_PATTERN_EDGES_PER_NODE = 64
MAX_PATTERN_OPCODE_SET = 32

# Keys a pattern node may carry, and the keys it must carry.
NODE_KEYS = ("kind", "id", "op", "operands", "results")
NODE_REQUIRED_KEYS = ("kind", "id", "op")

# Symbol roots the runtime publishes. Node ids and pattern variables cannot use
# them (RFC 0020 §6.1).
RESERVED_ROOTS = ("graph", "kernel", "device")

_IDENT_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_IDENT_RE = re.compile(rf"^{_IDENT_PATTERN}$")
_BINDING_RE = re.compile(rf"^\$({_IDENT_PATTERN})(\?)?$")

# Environment override for the registry path. CMake sets it for tests and the
# configure probe. The pack step passes the path as an argument.
OP_SCHEMA_REGISTRY_ENV = "HKP_OP_SCHEMA_REGISTRY"

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
    """Return the registry path from the argument, the environment, or the repo copy."""
    if path is not None:
        return Path(path)
    env = os.environ.get(OP_SCHEMA_REGISTRY_ENV)
    if env:
        return Path(env)
    return DEFAULT_OP_SCHEMA_REGISTRY


def _registry_edges(entries, path, opcode, direction):
    if not isinstance(entries, list):
        raise HkpPackError(
            f"malformed op-schema registry {path}: op '{opcode}' {direction} "
            "must be an array"
        )
    edges = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("optional"), bool)
        ):
            raise HkpPackError(
                f"malformed op-schema registry {path}: op '{opcode}' {direction} "
                "entries must be objects with a string 'name' and a bool 'optional'"
            )
        edges[entry["name"]] = entry["optional"]
    return edges


def _check_registry_edge_lists(entries, path, opcode, key):
    """Validate an op's edge lists, which are tensor-uid vectors.

    A pattern never binds a list, so no names are kept. A pattern edge that
    names a list field resolves as undeclared, like any other unknown edge.
    """
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("name"), str)
        for entry in entries
    ):
        raise HkpPackError(
            f"malformed op-schema registry {path}: op '{opcode}' {key} must be an "
            "array of objects with a string 'name'"
        )


@functools.lru_cache(maxsize=None)
def _load_op_schema_registry(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HkpPackError(f"cannot read op-schema registry {path}: {exc}") from exc
    try:
        doc = loads_strict(text)
    except (DuplicateKeyError, json.JSONDecodeError) as exc:
        raise HkpPackError(f"malformed op-schema registry {path}: {exc}") from exc
    ops = doc.get("ops") if isinstance(doc, dict) else None
    if not isinstance(ops, dict):
        raise HkpPackError(
            f"malformed op-schema registry {path}: 'ops' must be an object"
        )
    registry = {}
    for opcode, entry in ops.items():
        if not isinstance(entry, dict):
            raise HkpPackError(
                f"malformed op-schema registry {path}: op '{opcode}' must be an object"
            )
        registry[opcode] = OpSchema(
            opcode=opcode,
            operands=_registry_edges(entry.get("operands"), path, opcode, "operands"),
            results=_registry_edges(entry.get("results"), path, opcode, "results"),
        )
        for key in ("operand_lists", "result_lists"):
            _check_registry_edge_lists(entry.get(key), path, opcode, key)
    return registry


def load_op_schema_registry(path):
    """Return {opcode: OpSchema} from the registry at path. Each path is read once."""
    return _load_op_schema_registry(str(path))


def _quote(value):
    """Format a value as JSON for error messages."""
    return json.dumps(value)


def _parse_op(node, label):
    op = node["op"]
    if isinstance(op, str):
        if not op:
            raise HkpPackError(f"{label} field 'op' must be a nonempty string")
        return (op,)
    if not isinstance(op, dict):
        raise HkpPackError(
            f"{label} field 'op' must be an opcode string or an object "
            '{"one_of": [...]}'
        )
    require_known_keys(op, ("one_of",), f"{label} op")
    if "one_of" not in op:
        raise HkpPackError(f"{label} op object missing required field 'one_of'")
    members = op["one_of"]
    if not isinstance(members, list) or any(
        not isinstance(member, str) or not member for member in members
    ):
        raise HkpPackError(
            f"{label} op 'one_of' must be an array of nonempty opcode strings"
        )
    if len(members) < 2:
        raise HkpPackError(f"{label} op 'one_of' needs at least two opcodes")
    if len(set(members)) != len(members):
        raise HkpPackError(f"{label} op 'one_of' contains duplicates")
    if len(members) > MAX_PATTERN_OPCODE_SET:
        raise HkpPackError(
            f"{label} op 'one_of' lists {len(members)} opcodes, more than the "
            f"limit of {MAX_PATTERN_OPCODE_SET}"
        )
    return tuple(members)


def _parse_edges(node, key, label):
    """Parse the `operands` or `results` map. Only an operand may bind with `?`."""
    direction = key[:-1]
    edges_doc = node.get(key, {})
    if not isinstance(edges_doc, dict):
        raise HkpPackError(f"{label} field '{key}' must be an object")
    edges = []
    for name, binding in edges_doc.items():
        match = _BINDING_RE.fullmatch(binding) if isinstance(binding, str) else None
        if match is None:
            form = "'$name' or '$name?'" if key == "operands" else "'$name'"
            raise HkpPackError(
                f"{label} {direction} '{name}' binding {_quote(binding)} must be "
                f"of the form {form}"
            )
        variable, optional = match.group(1), match.group(2) is not None
        if optional and key == "results":
            raise HkpPackError(
                f"{label} result '{name}' binding {_quote(binding)} takes no '?': "
                "an op's declared result is always produced"
            )
        if variable in RESERVED_ROOTS:
            raise HkpPackError(
                f"{label} {direction} '{name}' binds reserved root '${variable}'"
            )
        edges.append(_Edge(name=name, variable=variable, optional=optional))
    return tuple(edges)


def _parse_node(index, node, seen_ids, where):
    if not isinstance(node, dict):
        raise HkpPackError(f"{where} graph_match node {index} must be an object")
    for key in NODE_REQUIRED_KEYS:
        if key not in node:
            raise HkpPackError(
                f"{where} graph_match node {index} missing required field '{key}'"
            )
    node_id = node["id"]
    if not isinstance(node_id, str) or not _IDENT_RE.fullmatch(node_id):
        raise HkpPackError(
            f"{where} graph_match node {index} id {_quote(node_id)} must be an "
            f"identifier matching ^{_IDENT_PATTERN}$"
        )
    label = f"{where} graph_match node '{node_id}'"
    if node_id in RESERVED_ROOTS:
        raise HkpPackError(f"{label} id is a reserved root")
    if node_id in seen_ids:
        raise HkpPackError(f"{label} id is not unique within 'nodes'")
    seen_ids.add(node_id)
    require_known_keys(node, NODE_KEYS, label)
    if node["kind"] != "op":
        raise HkpPackError(
            f"{label} kind {_quote(node['kind'])} is not 'op', the only node kind"
        )
    opcodes = _parse_op(node, label)
    operands = _parse_edges(node, "operands", label)
    results = _parse_edges(node, "results", label)
    edge_count = len(operands) + len(results)
    if edge_count > MAX_PATTERN_EDGES_PER_NODE:
        raise HkpPackError(
            f"{label} binds {edge_count} edges, more than the limit of "
            f"{MAX_PATTERN_EDGES_PER_NODE}"
        )
    return _Node(id=node_id, opcodes=opcodes, operands=operands, results=results)


def _check_variables(nodes, where):
    """Check variable bindings across the whole pattern.

    A variable cannot share a name with a node id, and at most one result binds
    it. A variable no result binds is a graph input, so exactly one operand binds
    it. An operand that binds a produced variable cannot be optional.
    """
    node_ids = {node.id for node in nodes}
    producer = {}
    for node in nodes:
        for edge in node.operands + node.results:
            if edge.variable in node_ids:
                raise HkpPackError(
                    f"{where} graph_match node '{node.id}' variable "
                    f"'${edge.variable}' collides with node id '{edge.variable}'"
                )
        for edge in node.results:
            if edge.variable in producer:
                first_node, first_edge = producer[edge.variable]
                raise HkpPackError(
                    f"{where} graph_match node '{node.id}' result '{edge.name}' "
                    f"binds '${edge.variable}', already bound by node "
                    f"'{first_node}' result '{first_edge}'"
                )
            producer[edge.variable] = (node.id, edge.name)
    graph_input = {}
    for node in nodes:
        for edge in node.operands:
            if edge.variable in producer:
                if edge.optional:
                    raise HkpPackError(
                        f"{where} graph_match node '{node.id}' operand "
                        f"'{edge.name}' marks '${edge.variable}' optional, but "
                        f"node '{producer[edge.variable][0]}' produces it"
                    )
                continue
            if edge.variable in graph_input:
                first_node, first_edge = graph_input[edge.variable]
                raise HkpPackError(
                    f"{where} graph_match node '{node.id}' operand '{edge.name}' "
                    f"binds '${edge.variable}', already bound by node "
                    f"'{first_node}' operand '{first_edge}'; no node produces it, "
                    "so exactly one operand binds it"
                )
            graph_input[edge.variable] = (node.id, edge.name)


def _resolve_edge(node, edge, direction, registry, label):
    """Return the registry optionality of an edge. Every opcode must agree on it."""
    optional_in = {}
    for opcode in node.opcodes:
        schema = registry[opcode]
        if direction == "operand":
            declared, other, misplaced = schema.operands, schema.results, "a result"
        else:
            declared, other, misplaced = schema.results, schema.operands, "an operand"
        if edge.name not in declared:
            if edge.name in other:
                raise HkpPackError(
                    f"{label} {direction} '{edge.name}' is declared by op "
                    f"'{opcode}' as {misplaced}"
                )
            raise HkpPackError(
                f"{label} {direction} '{edge.name}' is not declared by op '{opcode}'"
            )
        optional_in[opcode] = declared[edge.name]
    if len(set(optional_in.values())) > 1:
        required = [op for op, optional in optional_in.items() if not optional]
        optional = [op for op, opt in optional_in.items() if opt]
        raise HkpPackError(
            f"{label} one_of members disagree on {direction} '{edge.name}': "
            f"required in {', '.join(required)}; optional in {', '.join(optional)}"
        )
    return next(iter(optional_in.values()))


def _resolve(nodes, registry, where):
    for node in nodes:
        label = f"{where} graph_match node '{node.id}'"
        for opcode in node.opcodes:
            if opcode not in registry:
                raise HkpPackError(
                    f"{label} op '{opcode}' is not in the op-schema registry"
                )
        for edge in node.operands:
            optional = _resolve_edge(node, edge, "operand", registry, label)
            if edge.optional and not optional:
                raise HkpPackError(
                    f"{label} operand '{edge.name}' is bound with '?', but the "
                    "registry declares it required"
                )
        for edge in node.results:
            _resolve_edge(node, edge, "result", registry, label)


def validate_graph_pattern(nodes, where, registry_path=None):
    """Validate a `graph_match.nodes` block as the runtime loads it.

    `where` names the UED in error messages. `registry_path` overrides the
    registry location (see resolve_op_schema_registry). The registry is read
    only after the well-formedness checks pass. An unreadable registry raises an
    error that names its path.
    """
    if not isinstance(nodes, list) or not nodes:
        raise HkpPackError(f"{where} graph_match 'nodes' must be a nonempty array")
    if len(nodes) > MAX_PATTERN_NODES:
        raise HkpPackError(
            f"{where} graph_match 'nodes' has {len(nodes)} nodes, more than the "
            f"limit of {MAX_PATTERN_NODES}"
        )
    seen_ids = set()
    parsed = [
        _parse_node(index, node, seen_ids, where) for index, node in enumerate(nodes)
    ]
    _check_variables(parsed, where)
    registry = load_op_schema_registry(resolve_op_schema_registry(registry_path))
    _resolve(parsed, registry, where)
