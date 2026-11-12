# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The GEMM spec dicts and the binding that reads them must name the same keys.

``backend="cpp"`` hands the C++ engine a spec as a flat dict built from the
dataclasses (``universal_gemm_spec_to_dict`` and the batched, grouped and
FlatMM variants in ``rocke.core.backend``). The binding reads that dict key by
key (``fill_universal_spec`` and the family readers in
``cpp/bindings/rocke_engine.cpp``), and any key it does not read keeps the C
default without a word: the C engine then validates and plans a different
kernel from the one the Python spec describes. A key the binding reads that the
dict never writes is the same bug the other way round, a field one engine can
set and the other cannot.

So this test compares the two key sets in both directions, per reader. It reads
the binding as text rather than through the built engine, as
``test_arch_table_engine_parity.py`` does: ``rocke_engine`` is an optional build
artifact and CI does not run the binding tests, so a check that needs it would
be silent exactly where someone adds a field to one side only.

CPU-only: no GPU, compile, or built C++ engine required. Source-only: it reads
``cpp/bindings/rocke_engine.cpp``, which an installed test artifact does not
carry, so ``CMakeLists.txt`` keeps it out of the installed pytest cohort and
``tests/cmake/CoreTests.cmake`` runs it from the source tree as the
``rocke_gemm_spec_dict_binding_parity`` CTest.
"""

from __future__ import annotations

import dataclasses
import re
import unittest

from rocke.assets import platform_root
from rocke.core.backend import (
    batched_gemm_spec_to_dict,
    flatmm_spec_to_dict,
    grouped_gemm_spec_to_dict,
    universal_gemm_spec_to_dict,
)
from rocke.instances.common.batched_gemm import BatchedGemmSpec
from rocke.instances.common.flatmm import FlatMMSpec
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec, UniversalGemmSpec
from rocke.instances.common.grouped_gemm import GroupedGemmSpec

_BINDING = platform_root() / "cpp" / "bindings" / "rocke_engine.cpp"

# A key read by name: dict_int / dict_bool / dict_str(<dict>, "key", ...), or a
# presence test <dict>.contains("key") for the Optional fields.
_KEY_READ = re.compile(
    r'\bdict_(?:int|bool|str)\(\s*\w+\s*,\s*"(?P<a>\w+)"'
    r'|\.contains\(\s*"(?P<b>\w+)"\s*\)'
)


# A C/C++ comment, or a string literal kept as it is (a "//" inside a string is
# not a comment).
_COMMENT_OR_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"|/\*.*?\*/|//[^\n]*', re.S)


def _strip_comments(src: str) -> str:
    """``src`` without its comments, so a read that is commented out is not
    counted as a read. Each comment becomes one space; strings stay."""
    return _COMMENT_OR_STRING.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else " ", src
    )


def _function_body(src: str, signature: str) -> str:
    """The brace-balanced body of the one function whose definition matches
    ``signature`` (a regex for the text before its opening brace)."""
    matches = list(re.finditer(signature + r"[^;{]*\{", src))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one definition matching {signature!r}, found {len(matches)}"
        )
    start = matches[0].end()
    depth = 1
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _keys_read(body: str) -> set[str]:
    return {m.group("a") or m.group("b") for m in _KEY_READ.finditer(body)}


def _tile() -> TileSpec:
    return TileSpec(tile_m=64, tile_n=64, tile_k=32, warp_m=2, warp_n=2)


# reader function -> (signature regex, a spec, the dict its Python builder
# writes for it). The family readers fill the shared tile/trait pair through
# fill_universal_spec and then read their own keys.
def _specs() -> dict[str, tuple[str, object, object]]:
    return {
        "fill_universal_spec": (
            r"\bvoid fill_universal_spec\(",
            UniversalGemmSpec(name="parity", tile=_tile()),
            universal_gemm_spec_to_dict,
        ),
        "bg_build_spec": (
            r"\brocke_batched_gemm_spec_t bg_build_spec\(",
            BatchedGemmSpec(name="parity", tile=_tile()),
            batched_gemm_spec_to_dict,
        ),
        "gg_build_spec": (
            r"\brocke_grouped_gemm_spec_t gg_build_spec\(",
            GroupedGemmSpec(name="parity", tile=_tile(), trait=TraitSpec()),
            grouped_gemm_spec_to_dict,
        ),
        "fm_build_spec": (
            r"\brocke_flatmm_spec_t fm_build_spec\(",
            FlatMMSpec(tile=_tile()),
            flatmm_spec_to_dict,
        ),
    }


def _cases() -> dict[str, tuple[str, dict]]:
    return {r: (sig, to_dict(spec)) for r, (sig, spec, to_dict) in _specs().items()}


def _spec_fields(spec: object) -> set[str]:
    """Every field a spec carries: the fields of its nested specs (tile, trait,
    data) and its own top-level fields."""
    out = set()
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        if dataclasses.is_dataclass(value):
            out |= {g.name for g in dataclasses.fields(value)}
        else:
            out.add(f.name)
    return out


def _reader_keys(src: str, reader: str, signature: str) -> set[str]:
    body = _function_body(src, signature)
    keys = _keys_read(body)
    if reader != "fill_universal_spec":
        # A family reader fills a throwaway universal spec and copies only its
        # tile and trait (s.tile = u.tile; s.trait = u.trait), so of
        # fill_universal_spec's reads only the tile and trait keys reach it.
        for copy in ("s.tile = u.tile;", "s.trait = u.trait;"):
            if "fill_universal_spec(" not in body or copy not in body:
                raise AssertionError(
                    f"{reader} no longer fills tile/trait via fill_universal_spec"
                )
        shared = {f.name for f in dataclasses.fields(TileSpec)} | {
            f.name for f in dataclasses.fields(TraitSpec)
        }
        keys |= (
            _keys_read(_function_body(src, r"\bvoid fill_universal_spec\(")) & shared
        )
    return keys


def _mismatch(src: str) -> dict[str, tuple[list[str], list[str]]]:
    """reader -> (keys the dict writes that the reader skips,
    keys the reader reads that the dict never writes); empty when in step."""
    out = {}
    for reader, (signature, written) in _cases().items():
        read = _reader_keys(src, reader, signature)
        unread = sorted(set(written) - read)
        unwritten = sorted(read - set(written))
        if unread or unwritten:
            out[reader] = (unread, unwritten)
    return out


class TestGemmSpecDictBindingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _strip_comments(_BINDING.read_text(encoding="utf-8"))

    def test_each_dict_carries_every_field_of_its_spec(self):
        # The top-level keys (name, wave_size, block_size, dtype, batch_size ...)
        # are written by hand, so a new top-level spec field would be missing
        # from the dict, and then from the comparison below, without a word.
        for reader, (_, spec, to_dict) in _specs().items():
            with self.subTest(reader=reader):
                missing = sorted(_spec_fields(spec) - set(to_dict(spec)))
                self.assertEqual(missing, [], f"the {reader} dict omits {missing}")

    def test_each_reader_reads_every_key_its_dict_writes(self):
        for reader, (unread, _) in _mismatch(self.src).items():
            with self.subTest(reader=reader):
                self.assertEqual(
                    unread,
                    [],
                    f"{reader} in cpp/bindings/rocke_engine.cpp does not read "
                    f"{unread}: backend='cpp' would keep the C default",
                )

    def test_each_reader_reads_only_keys_its_dict_writes(self):
        for reader, (_, unwritten) in _mismatch(self.src).items():
            with self.subTest(reader=reader):
                self.assertEqual(
                    unwritten,
                    [],
                    f"{reader} reads {unwritten}, which its Python dict never writes",
                )

    def test_the_parser_sees_every_read(self):
        # The comparison is only as good as the regex: it must see the plain
        # reads, the multi-line ones and the Optional presence tests, and a read
        # taken out of the source must be reported.
        read = _keys_read(_function_body(self.src, r"\bvoid fill_universal_spec\("))
        for key in (
            "tile_m",
            "pipeline",
            "chiplet_wgm",
            "waves_per_eu",
            "emit_sched_hints",
            "cshuffle_no_alias",
            "dtype_a",
            "batched",
        ):
            with self.subTest(key=key):
                self.assertIn(key, read)
        for key in ("emit_sched_hints", "cshuffle_no_alias"):
            with self.subTest(withheld=key):
                mutated = self.src.replace(f'"{key}"', '"withheld_key"')
                unread, _ = _mismatch(mutated)["fill_universal_spec"]
                self.assertIn(key, unread)

    def test_a_commented_out_read_is_not_a_read(self):
        raw = _BINDING.read_text(encoding="utf-8")
        line = next(
            ln for ln in raw.splitlines() if 'dict_bool(tr, "cshuffle_no_alias"' in ln
        )
        for style, commented in (
            ("line", "// " + line),
            ("block", "/* " + line + " */"),
        ):
            with self.subTest(style=style):
                src = _strip_comments(raw.replace(line, commented))
                unread, _ = _mismatch(src)["fill_universal_spec"]
                self.assertIn("cshuffle_no_alias", unread)


if __name__ == "__main__":
    unittest.main()
