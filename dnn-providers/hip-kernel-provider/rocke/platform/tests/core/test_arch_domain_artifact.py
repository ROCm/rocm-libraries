# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate the generated intrinsic arch-domain artifacts against drift.

``tools/gen_arch_domain.py`` measures, per LLVM flavor, which intrinsic
declarations actually link for which gfx target, and commits the answer as
``python/rocke/core/arch/data/intrinsic_arch_domain.<flavor>.json``. A generated
artifact that is not gated drifts from its generator, and a stale availability
table is worse than none because it is trusted.

Two gates, because they fail for different reasons and are available in
different places:

* **Structure** runs everywhere — no LLVM, no GPU, no built C++ engine. It reads
  the committed JSON and checks it against the decl table it claims to describe.
  This is the gate that catches the realistic drift: someone adds, renames or
  removes an intrinsic key and the artifact silently stops covering it. The
  artifact would still be internally consistent, still parse, and still look
  authoritative, while saying nothing about the new key.

* **Regeneration** re-probes the toolchain and asserts the committed column for
  *this host's* flavor comes back byte-identical. It is the only check that can
  catch a wrong measurement rather than a missing one. A full sweep is ~150 keys
  x 7 targets and finishes in seconds, so it is a test rather than a nightly.

A host can only ever measure its own LLVM, so a flavor with no matching
toolchain is skipped, never failed — "we did not get an answer" must not be
recorded as "the answer is no". Same reason the artifact distinguishes
``arch_absent`` (the target genuinely cannot lower it) from ``target_unsupported``,
``toolchain_crash`` and ``toolchain_timeout`` (no data).

This file is a **source-tree** gate and is excluded from the installed test
tree by `CMakeLists.txt`. It has to be: it imports the generator out of
`tools/`, which is not installed, and an uninstallable import in an installed
test is not one skipped test but a collection error that takes the whole
pytest session down with it. The exclusion is the contract; keep the two in
step if this file is ever renamed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import files as resource_files
from pathlib import Path

from rocke.core.isa.backend import wired_arches
from rocke.core.lower_llvm import _resolve_llvm_flavor

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parents[1]  # tests/core -> platform
_TOOL = _ROCKE / "tools" / "gen_arch_domain.py"

# Through the package, not through the checkout layout. The artifacts are
# package data (`pyproject.toml` ships `rocke/**/*.json`, CMake installs the
# whole `python/rocke` tree), so resolving them by walking up from this file
# would check a directory that exists only in a source tree while the copy
# everything else imports sits elsewhere.
_DATA = Path(str(resource_files("rocke.core.arch") / "data"))

# Import the generator for its constants and its decl-table merge rule, so the
# gate cannot disagree with the tool about what "the decl table for a flavor"
# means. Re-deriving the merge here would just create a second place to forget
# a rung when flavor resolution gains one.
#
# `_hostcaps` rides along on the same path entry. It is shared with
# `tools/check_byte_identity.py`, so the pool-sizing test below is covering
# both callers, not just this one.
if str(_ROCKE / "tools") not in sys.path:
    sys.path.insert(0, str(_ROCKE / "tools"))
import gen_arch_domain as G
from _hostcaps import available_cpus

_STATUSES = {
    G.STATUS_OK,
    G.STATUS_NAME_ABSENT,
    G.STATUS_ARCH_ABSENT,
    G.STATUS_TARGET_UNSUPPORTED,
    G.STATUS_TOOLCHAIN_CRASH,
    G.STATUS_TOOLCHAIN_TIMEOUT,
    G.STATUS_PROBE_ERROR,
}

# Every status except `ok` is a claim about something that went wrong, and a
# claim with no diagnostic behind it cannot be acted on by whoever hits it.
_NEEDS_EVIDENCE = _STATUSES - {G.STATUS_OK}


def _columns() -> list[tuple[str, Path]]:
    """Committed artifacts as ``(flavor, path)``, flavor taken from the name."""
    out = []
    for path in sorted(_DATA.glob("intrinsic_arch_domain.*.json")):
        out.append((path.name.split(".")[1], path))
    return out


class ArchDomainArtifactStructureTest(unittest.TestCase):
    """Checks that need no toolchain, so they run on every host."""

    def test_at_least_one_column_is_committed(self):
        self.assertTrue(_columns(), f"no arch-domain artifact under {_DATA}")

    def test_columns_are_well_formed(self):
        for flavor, path in _columns():
            with self.subTest(flavor=flavor):
                doc = json.loads(path.read_text())
                self.assertEqual(doc.get("schema"), G.SCHEMA)

                # Provenance must agree with the filename. An artifact filed
                # under the wrong LLVM vintage is worse than a missing one: the
                # measurement is real, just attributed to a toolchain that never
                # produced it, and nothing downstream can tell.
                tc = doc.get("toolchain", {})
                self.assertEqual(tc.get("flavor"), flavor)
                clang_flavor = G.flavor_of_clang(tc.get("clang", ""))
                if clang_flavor is not None:
                    self.assertEqual(clang_flavor, flavor, tc.get("clang"))

                self.assertEqual(sorted(tc.get("arches", [])), sorted(wired_arches()))

    def test_every_decl_key_is_covered_on_every_wired_arch(self):
        """The drift this file exists to catch.

        The artifact answers a question *about the decl table*. Add a key and
        the artifact does not grow a row; it just quietly has no opinion on the
        one intrinsic nobody has measured yet. Nothing else in the tree notices,
        because the JSON is still valid and every row it does have is still
        right.

        Enforced asymmetrically, for the same reason the artifact separates
        ``arch_absent`` from ``target_unsupported``:

        * a **stale** key -- in the artifact, gone from the decl table -- is
          always a defect, and is caught on every column from any host. Nothing
          about it needs a toolchain to fix.
        * an **unmeasured** key can only be answered on a host running that
          flavor's toolchain. Failing every column here would mean one new decl
          key reds the build on every machine until someone has run all three
          toolchains -- "we did not get an answer" recorded as "the answer is
          no", which is exactly what the design forbids. So full coverage is
          required of this host's flavor, and reported as a named skip for the
          rest.
        """
        host = _resolve_llvm_flavor()
        arches = sorted(wired_arches())
        for flavor, path in _columns():
            with self.subTest(flavor=flavor):
                doc = json.loads(path.read_text())
                expected = set(G._decl_table(flavor))
                got = set(doc["keys"])
                for key, row in doc["keys"].items():
                    self.assertEqual(sorted(row), arches, key)
                self.assertEqual(
                    sorted(got - expected),
                    [],
                    f"{path.name} measures keys the decl table no longer has "
                    "-- re-run tools/gen_arch_domain.py and commit the result",
                )
                missing = sorted(expected - got)
                if not missing:
                    continue
                why = (
                    f"{path.name} has no answer for {len(missing)} decl "
                    f"key(s): {missing} -- re-run tools/gen_arch_domain.py on "
                    f"a {flavor} toolchain and commit the result"
                )
                # Both of these are raised *inside* `subTest`, whose executor
                # records the failure or skip against this column and does not
                # re-raise. Every later column is still checked -- which is the
                # point, since a stale key is a defect on any column and must
                # not be masked by an earlier column's missing measurement.
                if flavor == host:
                    self.fail(why)
                self.skipTest(why)

    def test_every_cell_carries_an_actionable_answer(self):
        for flavor, path in _columns():
            doc = json.loads(path.read_text())
            for key, row in doc["keys"].items():
                for arch, cell in row.items():
                    with self.subTest(flavor=flavor, key=key, arch=arch):
                        status = cell.get("status")
                        self.assertIn(status, _STATUSES)
                        # `probe_error` means our own probe module was
                        # malformed -- the one status that is a defect in the
                        # generator rather than a fact about the toolchain, and
                        # so must never survive into a committed column.
                        self.assertNotEqual(status, G.STATUS_PROBE_ERROR)
                        self.assertEqual(cell.get("verified_on"), flavor)
                        if status in _NEEDS_EVIDENCE:
                            self.assertTrue(cell.get("evidence"), "no diagnostic")

    def test_every_column_records_how_it_was_measured(self):
        """`clang` says which compiler answered, not which question it was asked.

        Those are different facts, and the difference is not academic: the same
        toolchain build reports ``ok`` or fails to terminate for the same key
        depending on the optimisation level the probe used, because -O3 can
        delete the call being probed. A column that carries only the compiler
        identity cannot be told apart from one measured the wrong way, which is
        how a false ``ok`` survived in the llvm20 column long enough to be
        committed.

        Asserted on the committed artifact rather than on a fresh run, because
        this is exactly the check a host with no toolchain can still make.
        """
        for flavor, path in _columns():
            with self.subTest(flavor=flavor):
                tc = json.loads(path.read_text())["toolchain"]
                self.assertIsInstance(tc.get("generator"), int)
                self.assertGreaterEqual(tc["generator"], G.GENERATOR)
                self.assertIn(
                    "-O0",
                    tc.get("probe_cflags", []),
                    "column measured without -O0: the IR pipeline can delete "
                    "the probed call, so its `ok` cells prove nothing",
                )
                self.assertGreater(tc.get("probe_timeout_s", 0), 0)


class AvailableCpusTest(unittest.TestCase):
    """The worker cap must reflect this process's share, not the host's size.

    `os.cpu_count()` reports the machine. On a scheduler-pinned host or in a
    container it overstates by orders of magnitude -- observed at 384 against
    an affinity of 2 -- and the sweep sized itself accordingly. Oversubscribing
    is not just slow here: the probes already run near the resource limits the
    cap exists to stay under, and a `probe_error` that depends on load makes
    the artifact nondeterministic and `--check` flaky.
    """

    def _root(self, files: dict[str, str]) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root

    def test_a_cpu_quota_caps_the_worker_count(self):
        # Asserted as "<= 1", not "== 1": affinity also feeds the minimum, so
        # the guarantee is that a quota can only ever lower the answer.
        for label, files in (
            ("v2", {"cpu.max": "100000 100000"}),
            (
                "v1",
                {"cpu/cpu.cfs_quota_us": "100000", "cpu/cpu.cfs_period_us": "100000"},
            ),
        ):
            with self.subTest(cgroup=label):
                self.assertLessEqual(available_cpus(self._root(files)), 1)

    def test_a_fractional_quota_still_leaves_one_worker(self):
        """Half a CPU is a *rate*, not half a process. Rounding down gives a
        zero-worker pool, which is a hang rather than a slow sweep."""
        root = self._root({"cpu.max": "50000 100000"})
        self.assertEqual(available_cpus(root), 1)

    def test_no_quota_does_not_lower_the_answer(self):
        """`max` and a negative v1 quota both mean unlimited -- neither is a
        cap of zero, which is what parsing them as numbers would produce."""
        for label, files in (
            ("v2-max", {"cpu.max": "max 100000"}),
            (
                "v1-negative",
                {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"},
            ),
            ("absent", {}),
        ):
            with self.subTest(cgroup=label):
                root = self._root(files)
                self.assertEqual(available_cpus(root), available_cpus(root / "nope"))

    def test_the_answer_is_always_a_usable_worker_count(self):
        for files in ({}, {"cpu.max": "garbage"}, {"cpu.max": "0 0"}):
            with self.subTest(files=sorted(files)):
                self.assertGreaterEqual(available_cpus(self._root(files)), 1)


class DriftTest(unittest.TestCase):
    """What `--check` forgives, and what it must not.

    The column embeds the exact ROCm build that measured it, so a byte
    comparison fails on any host whose patch level differs from the one that
    blessed the artifact -- which is most of them, and none of them for a
    reason anyone can act on. But the fields saying *how* the probe was posed
    are ours, not the host's, and forgiving those would forgive the class of
    defect they were added to expose.
    """

    def _doc(self, **toolchain) -> str:
        tc = {
            "flavor": "llvm22",
            "clang": "AMD clang version 22.0.0git (... roc-7.2.4 26084 abc)",
            "arches": ["gfx942"],
            "generator": G.GENERATOR,
            "probe_cflags": ["-O0", "-nogpulib"],
            "probe_timeout_s": 60,
        }
        tc.update(toolchain)
        return json.dumps(
            {
                "schema": G.SCHEMA,
                "toolchain": tc,
                "keys": {"a.b": {"gfx942": {"status": G.STATUS_OK}}},
                "canonical": {},
            }
        )

    def test_a_different_compiler_build_is_not_drift(self):
        other = "AMD clang version 22.0.0git (... roc-7.2.0 26014 def)"
        self.assertIsNone(G._drift(self._doc(), self._doc(clang=other)))

    def test_a_changed_measurement_is_drift_and_says_which(self):
        fresh = json.loads(self._doc())
        fresh["keys"]["a.b"]["gfx942"]["status"] = G.STATUS_ARCH_ABSENT
        why = G._drift(self._doc(), json.dumps(fresh))
        self.assertIsNotNone(why)
        self.assertIn("a.b/gfx942", why)
        self.assertIn(G.STATUS_ARCH_ABSENT, why)

    def test_a_changed_probe_method_is_drift_even_with_identical_cells(self):
        """The `-O0` case. Same clang, same cells, different question -- and
        the cells are identical only because nobody re-ran them yet."""
        for field, value in (
            ("probe_cflags", ["-nogpulib"]),
            ("probe_timeout_s", 5),
            ("generator", G.GENERATOR + 1),
        ):
            with self.subTest(field=field):
                why = G._drift(self._doc(), self._doc(**{field: value}))
                self.assertIsNotNone(why, field)
                self.assertIn("toolchain", why)

    def test_an_unreadable_column_is_drift_not_a_crash(self):
        self.assertIn("not valid JSON", G._drift("{oops", self._doc()))


class ClassifyTest(unittest.TestCase):
    """A diagnostic this tool does not recognise becomes `probe_error`.

    That is the safe default -- guessing would bake a false negative into the
    artifact -- but it is not an answer, and it must never reach a committed
    column. Each new LLVM vintage words its diagnostics differently, so the
    wordings we have had to learn are pinned here: they are the difference
    between a measurement and a blank.
    """

    def test_a_target_that_says_so_in_words_is_arch_absent(self):
        """llvm23 states it outright where llvm20 and llvm22 only fail.

        Left unrecognised, twelve cells across `permlane16`, `permlanex16` and
        `mov.dpp8` recorded `probe_error` -- the generator reporting itself
        broken for targets that had answered the question clearly.
        """
        status, evidence = G._classify(
            1,
            "error: <unknown>:0:0: in function probe void (ptr, i32): "
            "intrinsic not supported on subtarget",
        )
        self.assertEqual(status, G.STATUS_ARCH_ABSENT)
        self.assertTrue(evidence)

    def test_an_illegal_transfer_size_asks_for_a_re_probe(self):
        """The same complaint as the other two, caught earlier.

        llvm23 rejects an out-of-range load-to-LDS size in the verifier, before
        codegen. The message is about our module, not the target, so it must
        route to the operand rescue rather than be recorded as an answer.
        """
        diag = (
            "error: invalid LLVM IR input: invalid data size for load-to-LDS "
            "intrinsic; must be 1, 2, 4, 12, or 16"
        )
        status, evidence = G._classify(1, diag)
        self.assertEqual(status, G.STATUS_PROBE_ERROR)
        self.assertTrue(G._wants_literals(status, evidence))

    def test_the_sizes_llvm23_names_are_the_ones_we_try(self):
        """The candidate list is not a guess; llvm23 enumerates the domain."""
        self.assertEqual(G._IMMARG_PROBE_VALUES, (1, 2, 4, 12, 16))


class ImmargSweepTest(unittest.TestCase):
    """A negative must survive being asked with a legal operand value.

    The probe passes 0 for every `immarg`, and 0 is not legal everywhere: it is
    not a transfer size. When it is rejected, ISel says `Cannot select` -- which
    is also what a genuinely unsupported target says, so the diagnostic cannot
    separate the two and the generator recorded three CDNA targets as incapable
    of an instruction that ships in production. Asking again with a different
    value is the only discriminator.

    Both directions matter. A sweep that only manufactures more `ok` cells has
    broken the table in the direction that is harder to notice: a false
    `arch_absent` breaks a build loudly, a false `ok` admits a kernel that
    cannot run.
    """

    _IMM_DECL = "declare void @llvm.probe.imm(ptr addrspace(1) nocapture, i32 immarg)"

    def test_an_immarg_defaults_to_zero_and_the_sweep_moves_it(self):
        base, _ = G._probe_module(self._IMM_DECL, "e-p:64:64-i64:64")
        self.assertIn("i32 0)", base)
        for imm in G._IMMARG_PROBE_VALUES:
            with self.subTest(imm=imm):
                text, why = G._probe_module(
                    self._IMM_DECL, "e-p:64:64-i64:64", imm_int=imm
                )
                self.assertEqual(why, "")
                self.assertIn(f"i32 {imm})", text)

    def test_a_declare_with_no_immarg_has_nothing_to_sweep(self):
        """The generator decides what is sweepable by comparing module text.

        So a declare the sweep cannot vary must produce an identical module --
        otherwise every key would get four extra probes on every negative, for
        candidates that change nothing.
        """
        decl = "declare i32 @llvm.probe.plain(ptr addrspace(1) nocapture, i32)"
        base, _ = G._probe_module(decl, "e-p:64:64-i64:64")
        for imm in G._IMMARG_PROBE_VALUES:
            text, _ = G._probe_module(decl, "e-p:64:64-i64:64", imm_int=imm)
            self.assertEqual(text, base, f"imm_int={imm} varied a non-immarg decl")

    def test_a_rescued_cell_says_which_value_answered(self):
        """Without it, a first-try `ok` and a fourth-candidate `ok` look alike.

        That is not bookkeeping: the value is the evidence that the cell was
        measured with a legal operand, and re-deriving it by hand is exactly
        what nobody did for two committed columns.
        """
        for flavor, path in _columns():
            doc = json.loads(path.read_text())
            for key, row in doc["keys"].items():
                for arch, cell in row.items():
                    if "probe_imm" not in cell:
                        continue
                    with self.subTest(flavor=flavor, key=key, arch=arch):
                        self.assertIn(cell["probe_imm"], G._IMMARG_PROBE_VALUES)
                        # The sweep only ever returns early on a win, so a
                        # recorded value on a negative cell would mean the
                        # value is being reported for a probe it did not
                        # decide.
                        self.assertEqual(cell["status"], G.STATUS_OK)

    def test_global_load_lds_is_available_on_cdna_and_not_on_rdna(self):
        """The case that motivated the sweep, pinned in both directions.

        The intrinsic takes a per-lane transfer size, legal at 1/2/4 on CDNA
        and at 16 additionally on gfx950. It has a builder entry point, a
        lowering handler in both engines and a production instance, so an
        all-`arch_absent` row is a false negative on a live path. The RDNA half
        is the other direction: those targets really cannot lower it, and they
        must still say so after the full sweep.
        """
        for flavor, path in _columns():
            row = json.loads(path.read_text())["keys"].get("global.load.lds")
            if row is None:
                continue
            for arch, cell in sorted(row.items()):
                if cell["status"] in (
                    G.STATUS_TARGET_UNSUPPORTED,
                    G.STATUS_TOOLCHAIN_CRASH,
                    G.STATUS_TOOLCHAIN_TIMEOUT,
                ):
                    continue  # no data, not an answer
                expect = (
                    G.STATUS_OK if arch.startswith("gfx9") else G.STATUS_ARCH_ABSENT
                )
                with self.subTest(flavor=flavor, arch=arch):
                    self.assertEqual(cell["status"], expect)


class RefusalTest(unittest.TestCase):
    """What the generator must refuse to write.

    Both cases here produced a committed-looking artifact and exit 0 before.
    That is the worst failure mode this tool has: the artifact carries its own
    provenance and reads as authoritative, so a column written without the
    measurement behind it is not caught downstream by anything. Refusing at the
    point of generation puts the stop in front of the only person who can still
    fix it.

    Driven through `main()` with stub tools rather than a real sweep: the
    policy being tested is "what do we do when the toolchain does not answer",
    and a real toolchain that does answer cannot exercise it.
    """

    def _stub(self, name: str, body: str) -> str:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        path = root / name
        path.write_text("#!/usr/bin/env python3\n" + body)
        path.chmod(0o755)
        return str(path)

    # Answers `--version` as an unrecognised vintage -- `flavor_of_clang`
    # returns None, so the flavor-mismatch exit does not fire and the run
    # reaches the stage under test.
    _CLANG_REJECTS = """
import sys
if "--version" in sys.argv:
    print("stub clang")
    sys.exit(0)
sys.stderr.write("error: something this tool has never seen before\\n")
sys.exit(1)
"""

    # Echoes the declare back with an attribute group, which is how
    # `_name_exists` recognises a resolved intrinsic.
    _OPT_RESOLVES = """
import sys
src = sys.argv[-1]
for line in open(src):
    if line.startswith("declare "):
        print(line.rstrip() + " #0")
"""

    def _run(self, tools: dict[str, str | None], out: Path) -> int:
        from unittest import mock

        import check_ir_validity as V

        argv = [str(_TOOL), "--out", str(out), "--arch", "gfx942", "--only", "fabs"]
        with (
            mock.patch.object(V, "_llvm_tool", lambda n: tools.get(n)),
            mock.patch.object(sys, "argv", argv),
        ):
            return G.main()

    def test_a_missing_opt_writes_nothing_and_fails(self):
        """No name check, no column.

        `opt` owns the flavor axis. Without it nothing separates "this LLVM has
        no such intrinsic" from "this target cannot lower it", and the sweep
        used to substitute "the name exists" for "we did not ask" -- which can
        record a `name_absent` key as `ok`, because an unresolved `llvm.*` name
        is an ordinary external call that a link may happen to accept.

        It fails rather than reporting itself unvalidated, unlike the no-clang
        path: a host with no compiler is honestly not a probing host, but no
        complete LLVM install ships clang without `opt`, so finding one and not
        the other means the environment is pointed at a partial toolchain. That
        is a broken invocation, and exiting 0 would let the caller believe a
        sweep had happened.
        """
        out = Path(tempfile.mkdtemp()) / "column.json"
        self.addCleanup(shutil.rmtree, out.parent, True)
        clang = self._stub("clang", self._CLANG_REJECTS)
        rc = self._run({"clang": clang, "opt": None}, out)
        self.assertEqual(rc, 1, "a partial toolchain is a broken env, not a no-op")
        self.assertFalse(out.exists(), "wrote a column with no name check behind it")

    def test_an_unclassified_probe_writes_nothing_and_fails(self):
        """`probe_error` is this tool reporting itself broken.

        It is not a fact about the toolchain, and the artifact gate forbids it
        in a committed column -- so a generator that emits one produces a file
        its own test rejects, with nothing but habit between that and a commit.
        """
        out = Path(tempfile.mkdtemp()) / "column.json"
        self.addCleanup(shutil.rmtree, out.parent, True)
        tools = {
            "clang": self._stub("clang", self._CLANG_REJECTS),
            "opt": self._stub("opt", self._OPT_RESOLVES),
        }
        rc = self._run(tools, out)
        self.assertEqual(rc, 1)
        self.assertFalse(out.exists(), "committed our own defect as a measurement")


class ValidityGateTest(unittest.TestCase):
    """`tools/check_ir_validity.py` -- the sibling gate, same two rules."""

    def test_both_gates_ask_the_question_at_O0(self):
        """The one flag that decides whether either gate proves anything.

        At `-O3` the IR pipeline can fold or delete the construct being asked
        about, and the module then links because the suspect instruction is no
        longer in it. The arch-domain generator has probed at `-O0` since the
        false `ok` that taught us; the validity gate did not, and its link
        oracle was chosen precisely to avoid that class of false positive.
        Pinned together so the two cannot drift apart again.
        """
        import check_ir_validity as V

        self.assertIn("-O0", V.COMPILE_CFLAGS)
        self.assertIn("-O0", G.PROBE_CFLAGS)
        argv = V._compile_argv("clang", Path("m.ll"), "gfx942", Path("m.hsaco"))
        self.assertIn("-O0", argv)
        self.assertNotIn("-O3", argv)

    def test_a_target_this_clang_cannot_build_is_not_an_emission_failure(self):
        """rocke wires arches ahead of the toolchains that can build for them.

        gfx1250 against the LLVM 20 in ROCm 7.1 is rejected in the driver,
        before clang reads an instruction -- so blaming the lowerer for it
        blames it for a module nobody compiled, and since this gate now runs by
        default it would red every host whose ROCm predates the newest wired
        arch. Same line the generator already draws with `target_unsupported`.
        """
        import check_ir_validity as V

        self.assertTrue(
            V._target_unsupported("clang: error: invalid target ID 'gfx1250'")
        )
        # The same wording routes to `target_unsupported` in the generator.
        self.assertEqual(
            G._classify(1, "clang: error: invalid target ID 'gfx1250'")[0],
            G.STATUS_TARGET_UNSUPPORTED,
        )
        # A real emission defect must keep its verdict.
        for diag in (
            "LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.foo",
            "ld.lld: error: undefined symbol: llvm.amdgcn.made.up",
            "error: invalid use of a non-immediate operand",
        ):
            with self.subTest(diag=diag):
                self.assertFalse(V._target_unsupported(diag))

    def test_an_instance_gated_on_a_newer_llvm_is_not_a_lowering_failure(self):
        """The corpus is flavor-wide; a host is not.

        Four gfx1250 `wmma_scale` cases decline to lower on anything older than
        llvm23, and that refusal is the emitter working. It used to abort the
        whole run with FATAL before a single module was compiled, which reds
        the now-default `run_all.py` gate on every llvm22 host.

        Matched on the gate's wording rather than the exception type, because
        a `NotImplementedError` from anywhere else in the lowerer is still the
        validity failure this gate exists to catch.
        """
        import check_ir_validity as V

        self.assertTrue(
            V._requires_newer_flavor(
                NotImplementedError(
                    "tile.wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32 "
                    "requires llvm23 (ROCm 7.13+), got llvm22"
                )
            )
        )
        for exc in (
            NotImplementedError("no handler for op 'tile.frobnicate'"),
            NotImplementedError("unsupported dtype combination"),
        ):
            with self.subTest(exc=str(exc)):
                self.assertFalse(V._requires_newer_flavor(exc))

    def test_both_gates_read_a_clang_banner_the_same_way(self):
        """The two tools act on a vintage mismatch differently on purpose --
        the generator refuses, this gate reports UNVALIDATED -- but they must
        not disagree about what a mismatch *is*. They share `_llvm_identity`
        for exactly that reason; this pins the sharing.

        An unrecognised banner must read as "unknown", never as "mismatch": a
        vendor rewording that line would otherwise red every gate at once.
        """
        import _llvm_identity as I
        import check_ir_validity as V

        self.assertIs(V.flavor_of_clang, I.flavor_of_clang)
        self.assertIs(G.flavor_of_clang, I.flavor_of_clang)
        self.assertEqual(
            I.flavor_of_clang(
                "AMD clang version 22.0.0git "
                "(https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014)"
            ),
            "llvm22",
        )
        for banner in ("(unknown)", "", "AMD clang, the good one"):
            with self.subTest(banner=banner):
                self.assertIsNone(I.flavor_of_clang(banner))

    def test_both_corpus_gates_skip_a_newer_flavor_the_same_way(self):
        """Both gates lower the shared corpus, so both meet an instance that
        declines the flavor they asked for -- check_ir_validity on a host whose
        LLVM is too old, check_arch_domain on every pass below the newest
        committed column. Neither may count that as a failure, and they must
        not disagree about which refusals qualify, so they share
        `_flavor_gate`. This pins the sharing.

        The match is on the emitter's wording, not the exception type: the
        refusal is a bare NotImplementedError, and so are real defects.
        """
        import _flavor_gate as F
        import check_arch_domain as A
        import check_ir_validity as V

        self.assertIs(V._requires_newer_flavor, F.requires_newer_flavor)
        self.assertIs(A.requires_newer_flavor, F.requires_newer_flavor)

        self.assertTrue(
            F.requires_newer_flavor(
                NotImplementedError(
                    "tile.wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32 "
                    "requires llvm23 (ROCm 7.13+), got llvm20"
                )
            )
        )
        # A lowering defect must stay fatal, however it is spelled.
        for other in (
            NotImplementedError("op 'tile.nope' has no lowering handler"),
            ValueError("requires gfx1250, got gfx942"),
            KeyError("llvm23"),
        ):
            with self.subTest(exc=other):
                self.assertFalse(F.requires_newer_flavor(other))


class ArchDomainRegenerationTest(unittest.TestCase):
    """Re-probe this host's flavor and require the committed column to match."""

    def test_regenerating_this_hosts_column_is_a_no_op(self):
        proc = subprocess.run(
            [sys.executable, str(_TOOL), "--check"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        out = proc.stdout + proc.stderr

        # Three distinct kinds of "we cannot answer here", all skips. Failing
        # any of them would red every CI machine whose ROCm differs from the one
        # that blessed the artifact, which is most of them.
        if "UNVALIDATED" in out:
            self.skipTest(out.strip().splitlines()[-1])
        if proc.returncode == 2 and "clang reports" in out:
            self.skipTest("toolchain flavor disagrees with the resolved flavor")

        self.assertEqual(proc.returncode, 0, out)


if __name__ == "__main__":
    unittest.main()
