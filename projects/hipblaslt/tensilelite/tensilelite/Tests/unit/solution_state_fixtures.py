# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The solution-state cases of test_data/solution_records.

states/<base>.yaml holds the base states, and fixtures.yaml lists each case as
a base and its edits:

    <case>:
      base: <base>
      set: {<dotted key>: <value>}
      delete: [<dotted key>]

A set value replaces the old one wholesale; set is applied before delete.

test_solution_state_record.py reads the cases through state(). The
tensilelite-tests build runs this module as

    python solution_state_fixtures.py <directory>

and writes, for each case, <case>.msgpack, the state as solutionFromState
reads it. For a case with its golden files it also writes
<case>.predicates.msgpack, the trees of <case>.predicates.yaml, and a copy of
<case>.record.txt. It imports only PyYAML and msgpack, so the build needs no
TensileLite.
"""

import copy
import functools
import shutil
import sys
from pathlib import Path

import msgpack
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


def records_dir():
    return Path(__file__).resolve().parent / "test_data" / "solution_records"


def _load(path):
    with path.open() as handle:
        return yaml.load(handle, Loader=SafeLoader)


@functools.lru_cache(maxsize=None)
def fixtures():
    """Each case of fixtures.yaml by name."""
    return _load(records_dir() / "fixtures.yaml")


@functools.lru_cache(maxsize=None)
def _base(name):
    return _load(records_dir() / "states" / f"{name}.yaml")


def _parent(value, path):
    *parents, key = path.split(".")
    for parent in parents:
        value = value[parent]
    return value, key


def ordered(value):
    """Every map in key order and every sequence a list."""
    if isinstance(value, dict):
        return {key: ordered(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [ordered(item) for item in value]
    return value


def state(name):
    """The state of one case: its base with the edits applied, in key order.

    The record builders walk the state in its order, so both read this one.
    """
    case = fixtures()[name]
    rv = copy.deepcopy(_base(case["base"]))
    for path, value in (case.get("set") or {}).items():
        parent, key = _parent(rv, path)
        parent[key] = copy.deepcopy(value)
    for path in case.get("delete") or []:
        parent, key = _parent(rv, path)
        del parent[key]
    return ordered(rv)


def _pack(value):
    return msgpack.packb(value, use_bin_type=True)


def write(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(fixtures()):
        (directory / f"{name}.msgpack").write_bytes(_pack(state(name)))
        record = records_dir() / f"{name}.record.txt"
        predicates = records_dir() / f"{name}.predicates.yaml"
        if record.is_file() and predicates.is_file():
            shutil.copyfile(record, directory / record.name)
            (directory / f"{name}.predicates.msgpack").write_bytes(_pack(_load(predicates)))


if __name__ == "__main__":
    write(sys.argv[1])
