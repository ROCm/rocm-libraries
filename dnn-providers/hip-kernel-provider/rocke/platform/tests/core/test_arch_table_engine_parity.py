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
costs a regex and runs everywhere a source tree exists.

CPU-only: no GPU, compile, or built C++ engine required. Source-only: it reads
``cpp/core/arch/data.cpp``, which an installed test artifact does not carry, so
``CMakeLists.txt`` keeps it out of the installed pytest cohort and
``tests/cmake/CoreTests.cmake`` runs it from the source tree as the
``rocke_arch_table_engine_parity`` CTest.
"""

from __future__ import annotations

import json
import re
import unittest

from rocke.assets import platform_root

# One rocke_arch_target_t initializer. data.cpp spells every field with a
# designated initializer (``.gfx = "gfx1250"``), so each scalar is read by name
# from inside its target's block rather than by position.
_TARGET_BLOCK = re.compile(
    r"static (?:const|constexpr) rocke_arch_target_t k_target_(?P<sym>\w+) = "
    r"\{(?P<body>.*?)\n\};",
    re.S,
)
# A row that takes its facts from another (arch_specs.json "facts_of"):
# ``k_target_gfx1100 = facts_of(k_target_gfx11_generic, "gfx1100");``.
_FACTS_OF = re.compile(
    r"static constexpr rocke_arch_target_t k_target_\w+ = "
    r'facts_of\(k_target_(?P<src>\w+), "(?P<gfx>[\w-]+)"\);'
)

_INT_FIELDS = ("wave_size", "lds_capacity_bytes", "vmcnt_bits")
_STR_FIELDS = ("family", "target_family")

# The universal GEMM's capabilities (arch_specs.json "gemm"; in data.cpp a
# rocke_ati_gemm_caps_t that the arch's registry row points at).
_GEMM_ATOMS = re.compile(
    r"static const int k_gemm_atoms_(?P<sym>\w+)\[\]\[3\] = \{(?P<body>.*?)\};", re.S
)
_GEMM_CAPS = re.compile(
    r"static const rocke_ati_gemm_caps_t k_gemm_(?P<sym>\w+) = \{(?P<body>.*?)\n\};",
    re.S,
)
_REGISTRY_ROW = re.compile(
    r'\{"(?P<gfx>[\w-]+)", &k_target_\w+'
    r'(?:, (?:"(?P<facts_of>[\w-]+)"|NULL))?(?:, &k_gemm_(?P<gemm>\w+))?\}'
)
_GEMM_FLAGS = ("fp8_ab", "wmma_cshuffle", "cshuffle_lds_aliased")


def _field(body: str, name: str) -> str:
    # Top-level fields only: a nested aggregate (.mma, .memory, .limits) never
    # reuses these names, and the anchor on a leading newline keeps a longer
    # name such as target_family from matching "family".
    m = re.search(r"\n\s*\." + name + r'\s*=\s*"?([\w-]+)"?\s*,', body)
    if m is None:
        raise AssertionError(f"data.cpp target block has no .{name} field")
    return m.group(1)


def _cpp_targets() -> dict[str, dict[str, str]]:
    """Each arch's scalar facts; a ``facts_of`` row's are its source's, with
    ``facts_of`` naming the source arch."""
    src = (platform_root() / "cpp" / "core" / "arch" / "data.cpp").read_text()
    targets, by_symbol = {}, {}
    for m in _TARGET_BLOCK.finditer(src):
        body = "\n" + m.group("body")
        fields = {f: _field(body, f) for f in _INT_FIELDS + _STR_FIELDS}
        gfx = _field(body, "gfx")
        targets[gfx] = fields
        by_symbol[m.group("sym")] = gfx
    for m in _FACTS_OF.finditer(src):
        source = by_symbol[m.group("src")]
        targets[m.group("gfx")] = {**targets[source], "facts_of": source}
    return targets


def _cpp_gemm_caps() -> dict[str, dict]:
    """Each arch's GEMM capabilities as data.cpp's registry resolves them: a
    ``facts_of`` row takes its source's, and a row without a block gets none."""
    src = (platform_root() / "cpp" / "core" / "arch" / "data.cpp").read_text()
    atoms = {
        m.group("sym"): [
            [int(v) for v in t]
            for t in re.findall(r"\{(\d+), (\d+), (\d+)\}", m.group("body"))
        ]
        for m in _GEMM_ATOMS.finditer(src)
    }
    blocks = {}
    for m in _GEMM_CAPS.finditer(src):
        body = m.group("body")
        ref = re.search(r"\.wmma_atoms = k_gemm_atoms_(\w+),", body)
        caps = {"wmma_atoms": atoms[ref.group(1)] if ref else []}
        for flag in _GEMM_FLAGS:
            caps[flag] = re.search(r"\." + flag + r" = true,", body) is not None
        blocks[m.group("sym")] = caps
    registry = src[src.index("const rocke_ati_arch_row_t rocke_ati_arch_registry[]") :]
    rows = {m.group("gfx"): m for m in _REGISTRY_ROW.finditer(registry)}
    none = {"wmma_atoms": [], **{flag: False for flag in _GEMM_FLAGS}}
    resolved = {}
    for gfx, m in rows.items():
        own = rows[m.group("facts_of")] if m.group("facts_of") else m
        resolved[gfx] = blocks[own.group("gemm")] if own.group("gemm") else none
    return resolved


def _python_gemm_caps(row: dict) -> dict:
    gemm = row.get("gemm", {})
    return {
        "wmma_atoms": [list(a) for a in gemm.get("wmma_atoms", [])],
        **{flag: gemm.get(flag, False) for flag in _GEMM_FLAGS},
    }


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
    arches = json.loads(path.read_text())["arches"]
    return {
        gfx: (
            {**arches[row["facts_of"]], "facts_of": row["facts_of"]}
            if "facts_of" in row
            else row
        )
        for gfx, row in arches.items()
    }


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
            with self.subTest(arch=arch, field="facts_of"):
                self.assertEqual(cpp.get("facts_of"), py.get("facts_of"))

    def test_gemm_capabilities_agree(self):
        cpp_caps = _cpp_gemm_caps()
        self.assertEqual(set(cpp_caps), set(self.py))
        for arch, row in self.py.items():
            with self.subTest(arch=arch):
                self.assertEqual(cpp_caps[arch], _python_gemm_caps(row))

    def test_gemm_capabilities_parse_found_the_blocks(self):
        # Guards the regexes: gfx1250 sets every flag and the RDNA targets
        # inherit their generic's atom, so an empty parse cannot pass.
        cpp_caps = _cpp_gemm_caps()
        self.assertEqual(
            cpp_caps["gfx1250"],
            {
                "wmma_atoms": [[16, 16, 32], [16, 16, 64], [16, 16, 128]],
                "fp8_ab": True,
                "wmma_cshuffle": True,
                "cshuffle_lds_aliased": True,
            },
        )
        self.assertEqual(cpp_caps["gfx1201"]["wmma_atoms"], [[16, 16, 16]])
        self.assertEqual(cpp_caps["gfx942"]["wmma_atoms"], [])

    def test_gemm_wmma_atoms_are_sorted(self):
        # The C++ validator lists them in table order where Python sorts them,
        # so the two reject messages agree only while the table is sorted.
        for arch, caps in _cpp_gemm_caps().items():
            with self.subTest(arch=arch):
                self.assertEqual(caps["wmma_atoms"], sorted(caps["wmma_atoms"]))

    def test_gfx1250_carries_the_raised_lds_capacity(self):
        # Named rather than left to the loop above: this is the value Phase 0
        # changed, and a bare "they match" failure would not say which engine
        # was left behind or what the number was supposed to be.
        self.assertEqual(int(self.cpp["gfx1250"]["lds_capacity_bytes"]), 327680)
        self.assertEqual(self.py["gfx1250"]["lds_capacity_bytes"], 327680)


if __name__ == "__main__":
    unittest.main()
