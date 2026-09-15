# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The splice fragments' structural contracts against each other.

``templates/fragments/ingestor_packs_cpp.j2`` once emitted a two-field
``s_packs`` row against a THREE-field ``IngestorPack`` struct; the emitted
fragment did not compile as spliced and was fixed by hand during a real
integration run.

That drift is caught by the compiler, at the splice, with the struct in
hand -- which is the only place it can be judged. Parsing the provider's
``IngestorPacks.hpp`` from here to re-check it coupled this tool's test
suite to a header it does not own: a field added there reddened this suite,
for a mismatch the build already reports. So the arity check is gone.

What remains are contracts between fragments this generator itself emits,
which it does own and can check without reading anyone else's source.
"""
import re
from pathlib import Path

from codegen.generator import PLACEHOLDER_MARKER, mint_ids


class TestFragmentsAgreeWithEachOther:
    """Fragments that must agree on a shape, checked by parsing what this
    generator emits -- never by reading source it does not own."""

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

    def test_a_packaged_engine_gets_a_real_reset_pointer_not_the_default(
        self, generator, gfx950_attention_dense_config
    ):
        """A kpack engine owns a module cache and MUST drop it.

        The row used to be hardcoded `false, nullptr` for every dialect, with a
        comment telling the reader to hand-correct it. `TestIngestorPacksModuleCacheOwnership`
        asserts only `ownsModuleCache == (resetModuleCache != nullptr)`, which
        `false, nullptr` satisfies -- so a packaged engine left at the default
        shipped green and never dropped its cache. The dialect is known here, so
        the pair is emitted rather than remembered.
        """
        config = gfx950_attention_dense_config
        assert config.is_packaged, "fixture is no longer the packaged-dialect one"
        ids = mint_ids(config)
        cpp = generator._render_template(
            "fragments/ingestor_packs_cpp.j2", config, ids=ids
        )
        hpp = generator._render_template(
            "fragments/ingestor_packs_hpp.j2", config, ids=ids
        )
        row = next(line for line in cpp.splitlines() if line.strip().startswith('{"'))
        assert "true" in row and "nullptr" not in row, (
            f"packaged engine emitted a non-owning row: {row!r} -- a kpack engine "
            "that never drops its module cache passes every existing test"
        )
        reset_fn = re.search(r"&(reset\w+ModuleCache)", row)
        assert reset_fn, f"no reset symbol in the packaged row: {row!r}"
        assert f"void {reset_fn.group(1)}();" in hpp, (
            f"the row names {reset_fn.group(1)} but the .hpp fragment does not "
            "declare it -- the splice would not link"
        )

    def test_the_packaged_reset_symbol_is_actually_defined_somewhere(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        """Declared and referenced is not the same as defined.

        The first cut of the dialect fix emitted the row and the .hpp
        declaration but nothing that DEFINED `reset<Name>ModuleCache`, so every
        packaged engine spliced to an undefined-reference link error -- caught by
        review, not by the test above, which only checked the declaration.
        """
        config = gfx950_attention_dense_config
        written = generator.render(config, tmp_path)
        cpp_row = (tmp_path / "fragments/ingestor_packs.cpp.txt").read_text()
        reset_fn = re.search(r"&(reset\w+ModuleCache)", cpp_row)
        assert reset_fn, "packaged row names no reset symbol"
        symbol = reset_fn.group(1)

        defining = [
            rel
            for rel in written
            if rel.endswith(".cpp")
            and f"void {symbol}()" in (tmp_path / rel).read_text()
        ]
        assert defining, (
            f"{symbol} is referenced by the table row and declared in the .hpp, but "
            f"no generated .cpp defines it -- the splice would fail to link. "
            f"Generated sources: {[r for r in written if r.endswith('.cpp')]}"
        )

        # ...and outside the pack's anonymous namespace, or it has internal
        # linkage and the row still cannot see it.
        body = (tmp_path / defining[0]).read_text()
        anon_close = body.rindex("} // namespace\n")
        assert body.index(f"void {symbol}()") > anon_close, (
            f"{symbol} is defined INSIDE the anonymous namespace -- internal "
            "linkage, so IngestorPacks.cpp cannot reference it"
        )

    def test_a_direct_load_engine_defines_no_reset(
        self, generator, scale_add_config, tmp_path
    ):
        """Control: an embedded_source pack holds no archive, so emitting a
        module-cache reset for it would be dead code contradicting its own row."""
        written = generator.render(scale_add_config, tmp_path)
        for rel in written:
            if rel.endswith(".cpp"):
                assert "ModuleCache" not in (tmp_path / rel).read_text(), rel

    def test_a_direct_load_engine_stays_non_owning(self, generator, scale_add_config):
        """The companion control: an embedded_source engine holds no archive, so
        the owning shape would be wrong for it. Without this, the test above
        passes trivially if the template started emitting `true` unconditionally."""
        assert not scale_add_config.is_packaged
        cpp = generator._render_template(
            "fragments/ingestor_packs_cpp.j2",
            scale_add_config,
            ids=mint_ids(scale_add_config),
        )
        row = next(line for line in cpp.splitlines() if line.strip().startswith('{"'))
        assert "false" in row and "nullptr" in row, row

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


class TestPlaceholderScanSeesEveryEmittedFile:
    """The scan must cover what the generator WROTE, not one hand-picked glob.

    The runbook carried `grep -c "FILL THIS OUT" .../packs/*Native.cpp`, which
    silently omitted the generated matcher test -- so a run could report the
    step-6 gate green with placeholder bodies still shipping.
    """

    def test_the_generated_matcher_test_is_included(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        unfilled = generator.unfilled_placeholders(tmp_path, written)
        assert unfilled, "a freshly generated engine must carry unfilled stubs"
        assert any(rel.startswith("packs/") for rel in unfilled), unfilled
        assert any(rel.startswith("tests/") for rel in unfilled), (
            "the tests/ stub carries placeholders but the scan missed it -- "
            f"this is the exact gap the packs/-only glob had: {unfilled}"
        )

    def test_filling_every_placeholder_empties_the_scan(
        self, generator, scale_add_config, tmp_path
    ):
        """The control: the scan must be able to reach zero, or the gate it
        backs is unsatisfiable and would be worked around."""
        written = generator.render(scale_add_config, tmp_path)
        for rel in generator.unfilled_placeholders(tmp_path, written):
            path = tmp_path / rel
            path.write_text(path.read_text().replace(PLACEHOLDER_MARKER, "done"))
        assert generator.unfilled_placeholders(tmp_path, written) == {}

    def test_the_scan_finds_files_the_provider_splits_across_two_trees(
        self, generator, scale_add_config, tmp_path
    ):
        """The emitted layout is flat; the SPLICED layout is not.

        Packs go to the engine directory, test stubs to
        `src/tests/engines/.../packs/` -- the generator's own cmake_test_sources
        fragment says so. Resolving `root / rel` therefore found the packs and
        silently skipped every test stub, so the gate printed green on a tree
        whose matcher stubs were untouched: the exact packs/-only blind spot it
        was written to close.
        """
        written = generator.render(scale_add_config, tmp_path / "src")
        engine = tmp_path / "spliced/engine/packs"
        tests = tmp_path / "spliced/tests/engines/kernel_ingestor_engine/packs"
        engine.mkdir(parents=True)
        tests.mkdir(parents=True)
        for rel in written:
            if rel.startswith("packs/"):
                (engine / Path(rel).name).write_text(
                    (tmp_path / "src" / rel).read_text()
                )
            elif rel.startswith("tests/"):
                (tests / Path(rel).name).write_text(
                    (tmp_path / "src" / rel).read_text()
                )

        root = tmp_path / "spliced"
        located, missing, _amb = generator.locate_emitted(root, written)
        assert not [m for m in missing if m.startswith(("packs/", "tests/"))], missing

        # Fill only the pack, as a reader who trusted a packs/-only glob would.
        pack = engine / "ScaleAddNative.cpp"
        pack.write_text(pack.read_text().replace(PLACEHOLDER_MARKER, "done"))
        unfilled = generator.unfilled_placeholders(root, written)
        assert any(rel.startswith("tests/") for rel in unfilled), (
            "the matcher stub is still unfilled in the spliced tests tree but the "
            f"scan did not report it: {unfilled}"
        )

    def test_an_unlocatable_shippable_file_is_reported_missing(
        self, generator, scale_add_config, tmp_path
    ):
        """A file the engine ships that is nowhere under the root is an
        unfinished splice. Silence here is how the gate passed on a half-spliced
        tree; `fragments/` are splice instructions and correctly excluded."""
        written = generator.render(scale_add_config, tmp_path / "src")
        empty = tmp_path / "nothing"
        empty.mkdir()
        located, missing, _amb = generator.locate_emitted(empty, written)
        assert located == {}
        assert missing, "every shippable file is absent but none were reported"
        assert not any(m.startswith("fragments/") for m in missing), missing

    def test_two_files_with_one_basename_are_ambiguous_not_a_coin_flip(
        self, generator, scale_add_config, tmp_path
    ):
        """A stale copy must not be able to answer for the real file.

        The scan kept the first `rglob` hit, so the binding followed filesystem
        order: a FILLED stale copy sorting first reported the gate green while
        the real file still carried its markers. Basenames are not unique here
        (1809 collide repo-wide; `build/` duplicates shipped descriptor names),
        so this is luck, not a property. The decoy is named to sort BEFORE the
        real directory, which is the case that used to pass.
        """
        generator.render(scale_add_config, tmp_path / "gen")
        real = tmp_path / "root/real/packs"
        decoy = tmp_path / "root/aa_stale/packs"
        real.mkdir(parents=True)
        decoy.mkdir(parents=True)
        src = (tmp_path / "gen/packs/ScaleAddNative.cpp").read_text()
        (real / "ScaleAddNative.cpp").write_text(src)
        (decoy / "ScaleAddNative.cpp").write_text(src.replace(PLACEHOLDER_MARKER, "x"))

        written = generator.preview_files(scale_add_config)
        located, _missing, ambiguous = generator.locate_emitted(
            tmp_path / "root", written
        )
        assert "packs/ScaleAddNative.cpp" in ambiguous, (
            "a basename matching in two places was silently bound to one of them "
            f"-- located={located}"
        )
        assert len(ambiguous["packs/ScaleAddNative.cpp"]) == 2
        assert "packs/ScaleAddNative.cpp" not in located, (
            "an ambiguous file must not also be reported as located, or a caller "
            "that only checks `located` still gets the coin flip"
        )
