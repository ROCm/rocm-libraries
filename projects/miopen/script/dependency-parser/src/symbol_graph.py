#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Symbol bridge (Option B) for the dependency parser.

Additive attribution pass that runs after the ninja-deps (include-graph) mapping.
It closes the common-`.cpp`-body gap precisely, using the linker's own view.

A change to a ``.cpp`` body can only affect another translation unit through an
out-of-line symbol that unit references, so we attribute each source to the test
sources whose objects reference a symbol that source defines -- exactly what the
linker pulls in. This is semantically exact for compiled bodies and composes with
the include graph, which covers header-only / template / inline / macro code (none
of which has an out-of-line symbol for ``nm`` to see).

Not a replacement for the include graph -- only an additive layer. Additive and
idempotent; never removes edges. Costs one ``nm`` per project-compiled object
(parallelized); heavier than the stem bridge but does not over- or under-attribute
the way the sibling-header heuristic can.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# nm type codes. Undefined = a reference the linker must resolve. Defined-global =
# a symbol other objects can link against (uppercase = external/global; 'u'/'i' are
# unique/indirect globals). Local (lowercase t/d/b/r/s) symbols are not cross-TU
# linkable and are intentionally excluded as providers.
_UNDEFINED_TYPES = frozenset("Uw")
_DEFINED_GLOBAL_TYPES = frozenset("TDBRWVGSui")


def _norm(parser, path):
    """Project-relative path via whatever normalizer the base parser exposes."""
    fn = getattr(parser, "_to_project_relative", None) or getattr(
        parser, "_project_relative"
    )
    return fn(path)


def _nm_symbols(obj_path):
    """Return (defined_global, undefined) symbol sets for an object file via nm."""
    defined, undefined = set(), set()
    try:
        result = subprocess.run(
            ["nm", "--no-demangle", obj_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return defined, undefined
    if result.returncode != 0:
        return defined, undefined
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:  # "U symbol" (undefined: no address)
            type_code, symbol = parts
        elif len(parts) == 3:  # "address T symbol"
            _, type_code, symbol = parts
        else:
            continue
        if type_code in _UNDEFINED_TYPES:
            undefined.add(symbol)
        elif type_code in _DEFINED_GLOBAL_TYPES:
            defined.add(symbol)
    return defined, undefined


def apply(parser):
    """Attribute each defining source to the test sources referencing its symbols."""
    objects = list(parser.object_to_source.keys())
    if not objects:
        print("[bridge:symbol] no objects to scan")
        return

    def abs_obj(obj):
        return obj if os.path.isabs(obj) else os.path.join(parser.build_dir, obj)

    with ThreadPoolExecutor(max_workers=min(16, len(objects))) as executor:
        symbols = dict(
            zip(objects, executor.map(lambda o: _nm_symbols(abs_obj(o)), objects))
        )

    # symbol -> objects that define it (global)
    definers = {}
    for obj, (defined, _undef) in symbols.items():
        for sym in defined:
            definers.setdefault(sym, set()).add(obj)

    # For each test object, route its undefined symbols back to the defining source
    # and attribute that source to this test's synthetic bin/test_<stem> key.
    f2e = parser.file_to_executables
    added = 0
    for test_obj, (_defined, undefined) in symbols.items():
        test_src = parser.object_to_source.get(test_obj)
        if not test_src or not parser._is_gtest_source(test_src):
            continue
        test_key = f"bin/test_{Path(test_src).stem}"
        for sym in undefined:
            for def_obj in definers.get(sym, ()):
                if def_obj == test_obj:
                    continue
                def_src = parser.object_to_source.get(def_obj)
                if not def_src:
                    continue
                rel = _norm(parser, def_src)
                if rel.startswith(".."):
                    continue  # outside the project source tree
                if test_key not in f2e[rel]:
                    f2e[rel].add(test_key)
                    added += 1
    print(f"[bridge:symbol] symbol-graph attribution: {added} edges added")
