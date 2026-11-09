# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ engine's keyword table must hold exactly the Python engine's list.

The build generates the C++ copy from ``python/rocke/core/cxx_keywords.py``
(``tools/gen_cxx_keywords_cpp.py``), and both the engine and
``tests/core/test_value_names.cpp`` include that copy: the C++ test checks the
engine refuses every word in it. These tests hold the generated text to the
Python list, so the two engines refuse the same words.
"""

from __future__ import annotations

import importlib.util
import re

import pytest
from rocke.assets import platform_tools_dir
from rocke.core.cxx_keywords import CXX_KEYWORDS

GENERATOR = platform_tools_dir() / "gen_cxx_keywords_cpp.py"

# The installed test layout ships no tools/ tree.
pytestmark = pytest.mark.skipif(
    not GENERATOR.exists(),
    reason="runs the platform's generator, which an installed test layout omits",
)


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "rocke_gen_cxx_keywords_cpp", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literals(text: str) -> list[str]:
    """The string literals of a generated table, in order."""
    return re.findall(r'^"([^"]*)",$', text, flags=re.MULTILINE)


def test_the_generated_table_holds_the_python_list_sorted(generator, tmp_path):
    out = tmp_path / "rocke" / "cxx_keywords.inc"
    assert generator.main(["--out", str(out)]) == 0
    words = _literals(out.read_text())
    assert words == sorted(CXX_KEYWORDS)


def test_the_generator_reads_the_list_the_python_engine_imports(generator):
    """The generator loads the module by path, not through the rocke package;
    both must see the same list."""
    assert generator.load_keywords().CXX_KEYWORDS == CXX_KEYWORDS


def test_a_word_that_is_not_an_identifier_is_refused(generator):
    with pytest.raises(ValueError) as refused:
        generator.render(["for", 'say"hi'])
    assert str(refused.value) == (
        "C++ keyword 'say\"hi' is not an identifier; fix "
        "python/rocke/core/cxx_keywords.py"
    )
