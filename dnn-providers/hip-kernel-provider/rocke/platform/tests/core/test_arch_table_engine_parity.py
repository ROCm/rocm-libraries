# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The two engines' arch tables must agree on the scalar facts.

The arch table is duplicated by design: the Python engine reads
``core/arch/data/arch_specs.json`` and the C++ engine compiles
``cpp/core/arch/data.cpp``. Keeping two copies in step is the repository's #1
invariant, and for everything that reaches the emitted IR the byte-identity gate
enforces it.

These fields do not reach the emitted IR. ``lds_capacity_bytes`` is consulted by
tile-size validation and by the LDS planner -- it decides which kernels are
*allowed*, not what any allowed kernel emits -- so a gfx1250 kernel that passes
validation on one engine and is rejected by the other produces identical bytes
for every kernel that survives, and the gate stays GREEN. Raising gfx1250 from
160 KiB to 320 KiB in one engine and not the other is exactly that shape of bug:
no wrong answer, no failing gate, just tiles that the C++ engine refuses and the
Python engine accepts.

So this test reads the C++ table as text rather than through the built engine.
That is deliberate: ``rocke_engine`` is an optional build artifact, and a parity
test that skips whenever the engine is unbuilt would be silent on precisely the
machines where someone edits one engine and not the other. Parsing the source
costs a regex and runs everywhere.

CPU-only: no GPU, compile, or built C++ engine required.
"""

from __future__ import annotations

import json
import re
import unittest

from rocke.assets import platform_root

# The leading scalars of rocke_arch_target_t, which is initialized positionally.
# Anchoring on the three string fields before them keeps the match from drifting
# onto some other aggregate if a field is inserted: a new field would break the
# match loudly here rather than silently shift which number is read as the LDS
# capacity.
_TARGET_BLOCK = re.compile(
    r"static const rocke_arch_target_t k_target_\w+ = \{\s*"
    r'"(?P<name>[\w-]+)",\s*'
    r'"(?P<family>\w+)",\s*'
    r'"(?P<target_family>\w+)",\s*'
    r"(?P<wave_size>\d+),\s*"
    r"(?P<lds_capacity_bytes>\d+),\s*"
    r"(?P<vmcnt_bits>\d+),",
)

_INT_FIELDS = ("wave_size", "lds_capacity_bytes", "vmcnt_bits")
_STR_FIELDS = ("family", "target_family")


def _cpp_targets() -> dict[str, dict[str, str]]:
    src = (platform_root() / "cpp" / "core" / "arch" / "data.cpp").read_text()
    return {m.group("name"): m.groupdict() for m in _TARGET_BLOCK.finditer(src)}


def _python_targets() -> dict[str, dict]:
    path = (
        platform_root()
        / "python"
        / "rocke"
        / "core"
        / "arch"
        / "data"
        / "arch_specs.json"
    )
    return json.loads(path.read_text())["arches"]


class TestArchTableEngineParity(unittest.TestCase):
    def setUp(self):
        self.cpp = _cpp_targets()
        self.py = _python_targets()

    def test_both_engines_define_the_same_arches(self):
        # An arch present in one engine only is the same defect as a mismatched
        # field, and it would make every per-arch assertion below vacuous for
        # that arch.
        self.assertEqual(set(self.cpp), set(self.py))

    def test_the_parse_found_every_arch(self):
        # Guards the regex, not the data: a formatting change in data.cpp that
        # stopped it matching would otherwise turn this whole file into a test
        # that compares an empty dict to itself.
        self.assertGreaterEqual(len(self.cpp), 7)
        for arch in ("gfx942", "gfx950", "gfx1250"):
            self.assertIn(arch, self.cpp)

    def test_scalar_facts_agree(self):
        for arch, py in self.py.items():
            cpp = self.cpp[arch]
            for field in _INT_FIELDS:
                with self.subTest(arch=arch, field=field):
                    self.assertEqual(
                        int(cpp[field]),
                        py[field],
                        f"{arch}.{field}: data.cpp has {cpp[field]}, "
                        f"arch_specs.json has {py[field]}",
                    )
            for field in _STR_FIELDS:
                with self.subTest(arch=arch, field=field):
                    self.assertEqual(cpp[field], py[field])

    def test_gfx1250_carries_the_raised_lds_capacity(self):
        # Named rather than left to the loop above: this is the value Phase 0
        # changed, and a bare "they match" failure would not say which engine
        # was left behind or what the number was supposed to be.
        self.assertEqual(int(self.cpp["gfx1250"]["lds_capacity_bytes"]), 327680)
        self.assertEqual(self.py["gfx1250"]["lds_capacity_bytes"], 327680)


if __name__ == "__main__":
    unittest.main()
