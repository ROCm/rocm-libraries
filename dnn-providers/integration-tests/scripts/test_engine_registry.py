#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for engine_registry.py.

The parse exists so no script has to hand-maintain a copy of the engine list,
which means the thing worth testing is that it agrees with the real header --
and that it fails *closed* (returns None, so callers skip the check) rather
than returning a plausible-looking wrong answer when the header moves or the
macro is renamed.  A registry that quietly comes back empty would condemn every
sidecar in the tree as retired.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from engine_registry import ENGINE_NAMES_HEADER, known_engines, parse_registrations


class TestParseRegistrations:
    def test_reads_a_one_argument_registration(self) -> None:
        assert parse_registrations("HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE)\n") == {
            "MIOPEN_ENGINE"
        }

    def test_a_quoted_second_argument_names_the_engine(self) -> None:
        """The string literal is what reaches a sidecar, so it wins."""
        source = 'HIPDNN_REGISTER_ENGINE(MyEngine, "ON_DISK_NAME")\n'
        assert parse_registrations(source) == {"ON_DISK_NAME"}

    def test_ignores_indented_occurrences(self) -> None:
        """The macro definition and its doc-comment examples are not engines.

        ``EngineNames.hpp`` documents the macro with ``HIPDNN_REGISTER_ENGINE
        (MyEngine)`` inside a comment. Matching those would invent an engine
        named MyEngine and warn about every real sidecar that lacks it.
        """
        source = (
            "/// - `HIPDNN_REGISTER_ENGINE(MyEngine)`\n"
            "#define HIPDNN_REGISTER_ENGINE(...) /* ... */\n"
            "    HIPDNN_REGISTER_ENGINE(IndentedEngine)\n"
            "HIPDNN_REGISTER_ENGINE(REAL_ENGINE)\n"
        )
        assert parse_registrations(source) == {"REAL_ENGINE"}

    def test_tolerates_whitespace_inside_the_parentheses(self) -> None:
        source = 'HIPDNN_REGISTER_ENGINE( Spaced ,  "SPACED_NAME" )\n'
        assert parse_registrations(source) == {"SPACED_NAME"}

    def test_falls_back_to_the_identifier_when_the_second_argument_is_bare(
        self,
    ) -> None:
        """Not a form the macro accepts; guessing beats inventing a name."""
        assert parse_registrations("HIPDNN_REGISTER_ENGINE(Name, Other)\n") == {"Name"}


class TestKnownEngines:
    def test_missing_header_returns_none(self, tmp_path: Path) -> None:
        assert known_engines(tmp_path / "nope.hpp") is None

    def test_header_without_registrations_returns_none(self, tmp_path: Path) -> None:
        """Empty is indistinguishable from "wrong file", so report it as such."""
        header = tmp_path / "EngineNames.hpp"
        header.write_text("// no registrations here\n", encoding="utf-8")
        assert known_engines(header) is None

    def test_reads_a_synthetic_header(self, tmp_path: Path) -> None:
        header = tmp_path / "EngineNames.hpp"
        header.write_text(
            "HIPDNN_REGISTER_ENGINE(A_ENGINE)\nHIPDNN_REGISTER_ENGINE(B_ENGINE)\n",
            encoding="utf-8",
        )
        assert known_engines(header) == {"A_ENGINE", "B_ENGINE"}


@pytest.mark.skipif(
    not ENGINE_NAMES_HEADER.is_file(),
    reason="EngineNames.hpp not present in this checkout",
)
class TestRealHeader:
    def test_default_path_resolves(self) -> None:
        """The relative walk from scripts/ to the header still lands."""
        assert known_engines() is not None

    def test_finds_the_engines_the_harness_loads(self) -> None:
        """A spot check, not the full list.

        Asserting the exact set would make this test fail every time an engine
        is added -- which is precisely the maintenance burden the parse exists
        to remove. These three are load-bearing: the bundle tree's committed
        sidecars claim them, so if the parse stops seeing them the renderer
        starts warning that the whole tree is retired.
        """
        engines = known_engines()
        assert engines is not None
        assert {"MIOPEN_ENGINE", "HIP_MLOPS_ENGINE", "HIPBLASLT_ENGINE"} <= engines

    def test_no_name_looks_like_a_doc_comment_artefact(self) -> None:
        engines = known_engines()
        assert engines is not None
        assert "MyEngine" not in engines
        assert all(name.isidentifier() or '"' not in name for name in engines)
