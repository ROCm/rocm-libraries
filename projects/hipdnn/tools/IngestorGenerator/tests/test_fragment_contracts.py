# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The splice fragments' structural contracts against each other. A fragment's
agreement with the provider's own headers is caught by the compiler at the splice;
parsing those headers here would couple this suite to source it does not own."""
import re
from pathlib import Path

import pytest

from codegen.generator import PLACEHOLDER_MARKER, mint_ids
from tests.helpers import make_engine, make_kernel, make_minimal_config, make_pack

# Imported rather than re-written: a second extractor would be a second opinion about
# what counts as a case, and the pin checked below is a claim about that set.
from tests.test_generator import emitted_cases


class TestFragmentsAgreeWithEachOther:

    def test_ingestor_packs_hpp_declares_the_same_register_fn_the_cpp_row_uses(
        self, generator, scale_add_config
    ):
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
        """A kpack engine owns a module cache and MUST drop it: the reset sweep branches
        on `resetModuleCache != nullptr` alone, so a `nullptr` row reads as a pack with
        no archive to drop."""
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
        assert "nullptr" not in row, (
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
        """Declared and referenced is not defined: a row naming a
        `reset<Name>ModuleCache` that nothing defines splices to an undefined-reference
        link error."""
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
        """Control: an embedded_source pack holds no archive, so a module-cache reset
        would be dead code contradicting its own row."""
        written = generator.render(scale_add_config, tmp_path)
        for rel in written:
            if rel.endswith(".cpp"):
                assert "ModuleCache" not in (tmp_path / rel).read_text(), rel

    def test_a_direct_load_engine_keeps_a_null_reset_pointer(
        self, generator, scale_add_config
    ):
        """Without this, the test above passes against a template that emits a reset
        pointer unconditionally."""
        assert not scale_add_config.is_packaged
        cpp = generator._render_template(
            "fragments/ingestor_packs_cpp.j2",
            scale_add_config,
            ids=mint_ids(scale_add_config),
        )
        row = next(line for line in cpp.splitlines() if line.strip().startswith('{"'))
        assert "nullptr" in row, row

    def test_cmake_test_sources_names_files_this_generator_actually_writes(
        self, generator, scale_add_config, tmp_path
    ):
        """The fragment names its files under packs/ but this run writes them under
        tests/: the provider moves them when splicing."""
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


@pytest.fixture
def packaged_opposite_shape_config():
    """A packaged engine taking the OTHER arm of all three suite conditionals;
    ``configs/gfx950_attention_dense.yaml`` renders one fixed arm of each, against which
    a transcribed case list is indistinguishable from a derived one."""
    return make_minimal_config(
        dialect="packaged",
        kernel_source_kind="rocke",
        engine=make_engine(heuristic="none", behavior_notes=["runtime_compilation"]),
        packs=[
            make_pack(
                name="left",
                discriminator="left",
                arch=["gfx950"],
                kernels=[make_kernel(name="left.f32_block64")],
            ),
            make_pack(
                name="right",
                discriminator="right",
                arch=["gfx950"],
                kernels=[
                    make_kernel(
                        name="right.f32_block128",
                        metadata={"block_size": 128, "dtype": "FLOAT"},
                    )
                ],
            ),
        ],
    )


class TestCensusCasePinIsDerivedFromTheSuite:
    """``EXPECTED_CASES`` is READ OUT OF the suite template, never restated: the pin is
    what lets the census see a suite that SHRINKS, since the execution guard builds its
    obligations from the cases that registered."""

    @staticmethod
    def _census_call(fragment: str) -> list[str]:
        """The emitted ``hkp_register_census_tests(...)`` call, line by line. Sliced out
        rather than matched across the fragment, which is mostly prose using the same
        keywords."""
        lines = fragment.splitlines()
        opens = [
            index
            for index, line in enumerate(lines)
            if line.startswith("hkp_register_census_tests(")
        ]
        assert len(opens) == 1, f"expected exactly one census call:\n{fragment}"
        closes = [
            index
            for index, line in enumerate(lines[opens[0] :], opens[0])
            if line.strip() == ")"
        ]
        assert closes, f"the census call is never closed:\n{fragment}"
        return lines[opens[0] : closes[0] + 1]

    @classmethod
    def _pinned_case_lines(cls, fragment: str) -> list[str]:
        """The raw argument lines the ``EXPECTED_CASES`` keyword carries."""
        lines = cls._census_call(fragment)
        keywords = [
            index
            for index, line in enumerate(lines)
            if line.strip() == "EXPECTED_CASES"
        ]
        assert len(keywords) == 1, f"expected one EXPECTED_CASES keyword:\n{lines}"
        return lines[keywords[0] + 1 : -1]

    @classmethod
    def _pin_and_suite(cls, generator, config) -> tuple[list[str], dict]:
        """Both from ONE render context, so the pair compared is what a single run
        emits."""
        ids = mint_ids(config)
        fragment = generator._render_template(
            "fragments/cmake_test_sources.j2", config, ids=ids
        )
        suite = generator._render_template("test_packs.cpp.j2", config, ids=ids)
        pin = [line.strip() for line in cls._pinned_case_lines(fragment)]
        cases = emitted_cases(suite, "TEST_F", f"Test{config.engine.pascal_name}Packs")
        return pin, cases

    def test_the_pin_is_exactly_the_case_set_the_suite_renders(
        self, generator, gfx950_attention_dense_config
    ):
        """A list written here would be a third authority on the suite's shape."""
        config = gfx950_attention_dense_config
        assert config.is_packaged, "fixture is no longer the packaged-dialect one"
        pin, cases = self._pin_and_suite(generator, config)
        assert cases, "the suite template rendered no cases at all"
        assert set(pin) == set(cases), (
            f"the pin and the suite disagree: pinned-but-unregistered "
            f"{sorted(set(pin) - set(cases))}, registered-but-unpinned "
            f"{sorted(set(cases) - set(pin))}. Either way the census fails for a "
            "reason that is about this fragment rather than about the bundle"
        )
        assert len(pin) == len(set(pin)), f"the pin names a case twice: {pin}"

    def test_the_pin_follows_the_suites_conditional_arms(
        self, generator, packaged_opposite_shape_config
    ):
        """This config takes the opposite arm of all three gates in
        ``test_packs.cpp.j2``, so a pin agreeing with it cannot also be the shipped
        fixture's list. The gated names are stated because set equality alone is
        satisfied by a pin from an empty scrape."""
        config = packaged_opposite_shape_config
        assert not config.engine.has_heuristic
        assert config.is_multi_pack
        assert config.engine.behavior_notes
        pin, cases = self._pin_and_suite(generator, config)
        assert set(pin) == set(cases), (
            f"pinned-but-unregistered {sorted(set(pin) - set(cases))}, "
            f"registered-but-unpinned {sorted(set(cases) - set(pin))}"
        )
        assert "ShipsNoHeuristicAndRegistersNoScoreSymbol" in pin, pin
        assert "RanksThroughItsRegisteredScoreSymbol" not in pin, pin
        assert "CarriesOneGraphScopedMatcherPerPack" in pin, pin
        assert "CarriesNoGraphScopedMatcher" not in pin, pin
        assert "DeclaresItsConfiguredBehaviorNotes" in pin, pin

    def test_every_pinned_case_survives_the_wire_to_the_binary(
        self, generator, gfx950_attention_dense_config
    ):
        """The pin crosses three separators and may contain none of them: it leaves the
        fragment as CMake arguments, is joined into ONE comma-separated value, and
        travels inside the ENVIRONMENT property, itself a semicolon-separated VAR=VALUE
        list."""
        config = gfx950_attention_dense_config
        fragment = generator._render_template("fragments/cmake_test_sources.j2", config)
        lines = self._pinned_case_lines(fragment)
        assert lines, (
            "the call supplies the EXPECTED_CASES keyword with no names, which "
            "hkp_register_census_tests rejects: an empty pin admits every case "
            "set while reading as a pinned suite"
        )
        for raw in lines:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw.strip()), (
                f"pinned case {raw!r} is not the bare identifier a GTest case name "
                "is -- a separator or a space in it is a name no suite can register"
            )

    def test_a_direct_load_bundle_pins_nothing_because_it_registers_nothing(
        self, generator, scale_add_config
    ):
        """The call is not emitted unconditionally: a direct-load engine's suite is
        censused by a hand-added call at the pack target, so emitting EXPECTED_CASES
        here would hand the author a pin for a call they are not making."""
        assert not scale_add_config.is_packaged
        fragment = generator._render_template(
            "fragments/cmake_test_sources.j2", scale_add_config
        )
        payload = [
            line
            for line in fragment.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert payload, "the fragment splices nothing at all"
        assert not [
            line for line in payload if "hkp_register_census_tests" in line
        ], payload
        assert not [line for line in payload if "EXPECTED_CASES" in line], payload


class TestPlaceholderScanSeesEveryEmittedFile:
    """A packs/-only glob omits the generated matcher test, so RUNBOOK §3's placeholder
    gate reports green with placeholder bodies still shipping."""

    def test_the_generated_matcher_test_is_included(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        unfilled = generator.unfilled_placeholders([tmp_path], written)
        assert unfilled, "a freshly generated engine must carry unfilled stubs"
        assert any(rel.startswith("packs/") for rel in unfilled), unfilled
        assert any(rel.startswith("tests/") for rel in unfilled), (
            "the tests/ stub carries placeholders but the scan missed it -- "
            f"this is the exact gap the packs/-only glob had: {unfilled}"
        )

    def test_filling_every_placeholder_empties_the_scan(
        self, generator, scale_add_config, tmp_path
    ):
        """The scan must be able to reach zero, or the gate it backs is
        unsatisfiable."""
        written = generator.render(scale_add_config, tmp_path)
        for rel in generator.unfilled_placeholders([tmp_path], written):
            path = tmp_path / rel
            path.write_text(path.read_text().replace(PLACEHOLDER_MARKER, "done"))
        assert generator.unfilled_placeholders([tmp_path], written) == {}

    def test_the_scan_finds_files_the_provider_splits_across_two_trees(
        self, generator, scale_add_config, tmp_path
    ):
        """The emitted layout is flat; the SPLICED layout is not. Packs go to the engine
        directory and test stubs to `src/tests/engines/.../packs/`, so resolving `root /
        rel` finds the packs and skips every test stub."""
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
        located, missing, _amb = generator.locate_emitted([root], written)
        assert not [m for m in missing if m.startswith(("packs/", "tests/"))], missing

        # Fill only the pack, as a reader who trusted a packs/-only glob would.
        pack = engine / "ScaleAddNative.cpp"
        pack.write_text(pack.read_text().replace(PLACEHOLDER_MARKER, "done"))
        unfilled = generator.unfilled_placeholders([root], written)
        assert any(rel.startswith("tests/") for rel in unfilled), (
            "the matcher stub is still unfilled in the spliced tests tree but the "
            f"scan did not report it: {unfilled}"
        )

    def test_an_unlocatable_shippable_file_is_reported_missing(
        self, generator, scale_add_config, tmp_path
    ):
        """A shipped file nowhere under the root is an unfinished splice; `fragments/`
        are splice instructions and correctly excluded."""
        written = generator.render(scale_add_config, tmp_path / "src")
        empty = tmp_path / "nothing"
        empty.mkdir()
        located, missing, _amb = generator.locate_emitted([empty], written)
        assert located == {}
        assert missing, "every shippable file is absent but none were reported"
        assert not any(m.startswith("fragments/") for m in missing), missing

    def test_two_files_at_one_spliced_path_are_ambiguous_not_a_coin_flip(
        self, generator, scale_add_config, tmp_path
    ):
        """Two trees can legitimately hold the same spliced relative path, so keeping
        the first hit binds by filesystem order -- hence a decoy named to sort BEFORE
        the real directory."""
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
            [tmp_path / "root"], written
        )
        assert "packs/ScaleAddNative.cpp" in ambiguous, (
            "one spliced relative path matching in two places was silently bound "
            f"to one of them -- located={located}"
        )
        assert len(ambiguous["packs/ScaleAddNative.cpp"]) == 2
        assert "packs/ScaleAddNative.cpp" not in located, (
            "an ambiguous file must not also be reported as located, or a caller "
            "that only checks `located` still gets the coin flip"
        )

    def test_a_same_named_file_elsewhere_neither_satisfies_nor_confuses(
        self, generator, scale_add_config, tmp_path
    ):
        """A basename is not evidence: basenames collide repo-wide and `build/`
        duplicates shipped descriptor names, so matching on one would answer for a
        missing target."""
        generator.render(scale_add_config, tmp_path / "gen")
        decoy = tmp_path / "root/unrelated/subsystem"
        decoy.mkdir(parents=True)
        (decoy / "ScaleAddNative.cpp").write_text("// somebody else's file\n")

        written = generator.preview_files(scale_add_config)
        located, missing, ambiguous = generator.locate_emitted(
            [tmp_path / "root"], written
        )
        assert (
            "packs/ScaleAddNative.cpp" not in located
        ), "a file at an unrelated path answered for this engine's pack source"
        assert "packs/ScaleAddNative.cpp" not in ambiguous, (
            "an unrelated file must not manufacture ambiguity either -- that turns "
            "a correct tree red for a file that is not ours"
        )
        assert "packs/ScaleAddNative.cpp" in missing

    def test_the_real_spliced_path_is_located_across_separate_roots(
        self, generator, scale_add_config, tmp_path
    ):
        """The roots are a LIST because the engine tree and the provider's test tree may
        share no ancestor worth scanning, and widening one root drags in build trees."""
        written = generator.render(scale_add_config, tmp_path / "gen")
        engine = tmp_path / "provider/engines/kernel_ingestor_engine/packs"
        tests = tmp_path / "elsewhere/tests/engines/kernel_ingestor_engine/packs"
        engine.mkdir(parents=True)
        tests.mkdir(parents=True)
        for rel in written:
            if rel.startswith("packs/"):
                (engine / Path(rel).name).write_text(
                    (tmp_path / "gen" / rel).read_text()
                )
            elif rel.startswith("tests/"):
                (tests / Path(rel).name).write_text(
                    (tmp_path / "gen" / rel).read_text()
                )

        located, missing, ambiguous = generator.locate_emitted(
            [tmp_path / "provider", tmp_path / "elsewhere"], written
        )
        assert not ambiguous, ambiguous
        assert "packs/ScaleAddNative.cpp" in located
        assert "tests/TestScaleAddMatchers.cpp" in located, (
            "the test stub is spliced under packs/ in the provider's test tree -- "
            f"the cmake_test_sources fragment says so: {sorted(located)}"
        )
        assert all(m.startswith("test_descriptors/") for m in missing), missing

    def test_a_root_that_does_not_exist_is_an_error_not_an_empty_search(
        self, generator, scale_add_config, tmp_path
    ):
        """Zero hits with no complaint is a gate that passes because it looked
        nowhere."""
        written = generator.render(scale_add_config, tmp_path / "gen")
        with pytest.raises(ValueError, match="do not exist"):
            generator.locate_emitted([tmp_path / "gen", tmp_path / "typo"], written)
