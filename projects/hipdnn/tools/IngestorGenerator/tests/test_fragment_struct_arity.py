# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The splice fragments' structural contracts against the REAL provider source.

``templates/fragments/ingestor_packs_cpp.j2`` once emitted a two-field
``s_packs`` row (``{"name", &registerFn}``) against a THREE-field
``IngestorPack`` struct (the third field, ``resetModuleCache``, was added by
PR review); the emitted fragment did not compile as spliced and was fixed by
hand during a real integration run. The template is 3-field today (verified:
``git log -p`` on the template shows the 2-field -> 3-field commit), but
nothing asserts it STAYS in sync with the struct -- a hardcoded ``3`` in a
test would just re-freeze today's coincidental agreement.

This test parses ``IngestorPack``'s real field count out of
``IngestorPacks.hpp`` (regex over member declarations between the struct's
opening and closing braces, skipping comments) and asserts the fragment's
emitted aggregate-initializer arity matches it -- so a future field added to
the struct without updating the template fails HERE, not as a build error an
agent has to hand-debug days later.
"""
import re
from pathlib import Path

import pytest

from codegen.generator import mint_ids

_STRUCT_HEADER = (
    Path(__file__).resolve().parents[5]
    / "dnn-providers"
    / "hip-kernel-provider"
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "IngestorPacks.hpp"
)


def _real_ingestor_pack_field_count() -> int:
    """Parse the real field count of ``struct IngestorPack`` from its header.

    Isolates the struct body (opening `{` to matching `};`), strips `//` and
    `/* */` comments, and counts semicolon-terminated member declarations --
    the same shape RUNBOOK.md tells a human to eyeball with
    ``sed -n '/struct IngestorPack/,/};/p'``, just parsed instead of read.
    """
    text = _STRUCT_HEADER.read_text()
    match = re.search(r"struct IngestorPack\s*\{(.*?)\n\};", text, re.S)
    assert match, f"could not find 'struct IngestorPack { {...} };' in {_STRUCT_HEADER}"
    body = match.group(1)
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    members = [line.strip() for line in body.split(";") if line.strip()]
    return len(members)


@pytest.fixture(scope="module")
def real_struct_field_count():
    if not _STRUCT_HEADER.is_file():
        pytest.skip(f"provider source not found beside this checkout: {_STRUCT_HEADER}")
    count = _real_ingestor_pack_field_count()
    assert count > 0, "parsed zero fields -- the struct-body regex is broken, not empty"
    return count


class TestIngestorPacksCppFragmentArity:
    def test_emitted_row_arity_matches_the_real_struct(
        self, generator, scale_add_config, real_struct_field_count
    ):
        rendered = generator._render_template(
            "fragments/ingestor_packs_cpp.j2",
            scale_add_config,
            ids=mint_ids(scale_add_config),
        )
        row_match = re.search(r"\{[^{}]*\},", rendered)
        assert (
            row_match
        ), f"no aggregate-init row found in rendered fragment:\n{rendered}"
        row = row_match.group(0)
        # Count top-level comma-separated fields inside the braces (the row
        # has no nested braces/commas today; this is not a general C++
        # parser, just enough to count this specific shape).
        inner = row[row.index("{") + 1 : row.rindex("}")]
        arity = len([part for part in inner.split(",") if part.strip()])
        assert arity == real_struct_field_count, (
            f"emitted s_packs row has {arity} fields {row!r}, but the real "
            f"IngestorPack struct in {_STRUCT_HEADER} declares "
            f"{real_struct_field_count} -- the fragment has drifted from the "
            "struct it splices into and would not compile as emitted"
        )

    def test_a_stale_two_field_template_would_be_caught(
        self, generator, scale_add_config, real_struct_field_count
    ):
        """Sanity check on the check: the ORIGINAL two-field row (the real
        regression this fragment shipped, per git history) must fail this
        arity check against today's three-field struct, or the test above is
        vacuous."""
        stale_row = '{"{{ config.engine.name }}", &{{ config.register_symbols_fn }}},'
        inner = stale_row[stale_row.index("{") + 1 : stale_row.rindex("}")]
        stale_arity = len([p for p in inner.split(",") if p.strip()])
        assert stale_arity != real_struct_field_count, (
            "the historical two-field row now happens to match the struct's "
            "field count -- either the struct shrank back to two fields, or "
            "this sanity check needs updating"
        )


class TestOtherFragmentsStructuralContracts:
    """The other fragments with a similar 'must match a real declared shape'
    contract, checked the same way: parse the real thing, don't hardcode."""

    def test_ingestor_packs_hpp_declares_the_same_register_fn_the_cpp_row_uses(
        self, generator, scale_add_config
    ):
        """The .hpp declaration and the .cpp table row must name the exact
        same register-function symbol -- a mismatch would not compile."""
        ids = mint_ids(scale_add_config)
        hpp = generator._render_template(
            "fragments/ingestor_packs_hpp.j2", scale_add_config, ids=ids
        )
        cpp = generator._render_template(
            "fragments/ingestor_packs_cpp.j2", scale_add_config, ids=ids
        )
        hpp_fn = re.search(r"void (\w+)\(", hpp)
        cpp_fn = re.search(r"&(\w+),", cpp)
        assert (
            hpp_fn and cpp_fn
        ), "could not find the register-fn name in one of the two fragments"
        assert (
            hpp_fn.group(1) == cpp_fn.group(1) == scale_add_config.register_symbols_fn
        )

    def test_cmake_test_sources_names_files_this_generator_actually_writes(
        self, generator, scale_add_config, tmp_path
    ):
        """cmake_test_sources.txt names two files by their fragments'-output
        naming convention (Test<Name>Packs.cpp / Test<Name>Matchers.cpp,
        under packs/ per the fragment's own splice-target convention) -- both
        must correspond to files this run actually wrote (under tests/, per
        this tool's own output layout; the fragment's header comment says the
        provider moves them to packs/ when splicing)."""
        written = generator.render(scale_add_config, tmp_path)
        fragment = (tmp_path / "fragments" / "cmake_test_sources.txt").read_text()
        basenames = re.findall(r"packs/(Test\w+\.cpp)", fragment)
        assert basenames, f"no Test*.cpp basenames found in fragment:\n{fragment}"
        written_basenames = {Path(w).name for w in written if w.startswith("tests/")}
        for name in basenames:
            assert name in written_basenames, (
                f"fragment names '{name}', which this run never wrote under tests/ "
                f"(wrote: {sorted(written_basenames)})"
            )
