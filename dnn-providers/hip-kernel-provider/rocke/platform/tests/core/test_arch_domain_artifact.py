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
  the committed JSON and checks it against the decl table it claims to describe,
  catching the realistic drift: an intrinsic key added, renamed or removed and
  the artifact silently no longer covering it while still looking authoritative.

* **Regeneration** re-probes the toolchain and asserts the committed column for
  *this host's* flavor comes back byte-identical. It is the only check that can
  catch a wrong measurement rather than a missing one. A full sweep is ~150 keys
  x 7 targets and finishes in seconds, so it is a test rather than a nightly.

A host can only ever measure its own LLVM, so a flavor with no matching
toolchain is skipped, never failed — "we did not get an answer" must not be
recorded as "the answer is no". Same reason the artifact distinguishes
``arch_absent`` (the target genuinely cannot lower it) from ``target_unsupported``,
``toolchain_crash`` and ``toolchain_timeout`` (no data).

This file is a **source-tree** gate, excluded from the installed test tree by
`CMakeLists.txt`: it imports the generator out of the uninstalled `tools/`, and
an uninstallable import is a collection error that takes the whole pytest
session down. Keep the exclusion in step if this file is ever renamed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import files as resource_files
from pathlib import Path
from unittest import mock

from rocke.core.isa.backend import wired_arches
from rocke.core.lower_llvm import _resolve_llvm_flavor

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parents[1]  # tests/core -> platform
_TOOL = _ROCKE / "tools" / "gen_arch_domain.py"

# Through the package, not the checkout layout: the artifacts are package data,
# so walking up from this file would check a directory that exists only in a
# source tree while the copy everything else imports sits elsewhere.
_DATA = Path(str(resource_files("rocke.core.arch") / "data"))

# Import the generator for its constants and its decl-table merge rule, so the
# gate cannot disagree with the tool about what "the decl table for a flavor"
# means. `_hostcaps` rides along on the same path entry; it is shared with
# `tools/check_byte_identity.py`, so the pool-sizing test below covers both
# callers.
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

                # Provenance must agree with the filename: a column filed under
                # the wrong LLVM vintage is a real measurement attributed to a
                # toolchain that never produced it, and nothing downstream can
                # tell.
                tc = doc.get("toolchain", {})
                self.assertEqual(tc.get("flavor"), flavor)
                clang_flavor = G.flavor_of_clang(tc.get("clang", ""))
                if clang_flavor is not None:
                    self.assertEqual(clang_flavor, flavor, tc.get("clang"))

                self.assertEqual(sorted(tc.get("arches", [])), sorted(wired_arches()))

    def test_every_decl_key_is_covered_on_every_wired_arch(self):
        """The drift this file exists to catch: a decl key the artifact has no
        opinion on, in JSON that is still valid and still looks authoritative.

        Enforced asymmetrically, for the same reason the artifact separates
        ``arch_absent`` from ``target_unsupported``:

        * a **stale** key -- in the artifact, gone from the decl table -- is
          always a defect, caught on every column from any host.
        * an **unmeasured** key can only be answered on a host running that
          flavor's toolchain, so full coverage is required of this host's
          flavor and reported as a named skip for the rest.
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
                # Raised *inside* `subTest`, which records against this column
                # without re-raising, so every later column is still checked --
                # a stale key must not be masked by an earlier column's missing
                # measurement.
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
                        # malformed -- a defect in the generator rather than a
                        # fact about the toolchain, so it must never survive
                        # into a committed column.
                        self.assertNotEqual(status, G.STATUS_PROBE_ERROR)
                        self.assertEqual(cell.get("verified_on"), flavor)
                        if status in _NEEDS_EVIDENCE:
                            self.assertTrue(cell.get("evidence"), "no diagnostic")

    def test_every_column_records_how_it_was_measured(self):
        """`clang` says which compiler answered, not which question it was asked.

        One toolchain build reports ``ok`` or not for the same key depending on
        the probe's optimisation level, because -O3 can delete the call being
        probed -- so a column carrying only the compiler identity cannot be
        told apart from one measured the wrong way. Asserted on the committed
        artifact, so a host with no toolchain can still make the check.
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

    `os.cpu_count()` reports the machine -- observed at 384 against an affinity
    of 2 on a scheduler-pinned host. Oversubscribing is not just slow: a
    load-dependent `probe_error` makes the artifact nondeterministic and
    `--check` flaky.
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

    def test_no_tool_sizes_a_pool_from_the_machine(self):
        """The helper only helps the callers that ask it.

        Each of these tools fans out subprocesses, and one left on
        `os.cpu_count()` launches a 384-way build against a 2-CPU allocation;
        the failures land as compile defects and read as bad IR. Asserted per
        file, since the next tool needing a worker count will be copied from
        one of these. Only code is searched -- a comment may name the rule.
        """
        for name in (
            "gen_arch_domain.py",
            "check_ir_validity.py",
            "check_byte_identity.py",
            "run_portable_ir_gates.py",
        ):
            with self.subTest(tool=name):
                src = (_ROCKE / "tools" / name).read_text()
                code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
                self.assertNotIn("cpu_count", code, f"{name} sizes from the machine")
                self.assertIn("available_cpus", code, f"{name} sizes from nothing")


class DriftTest(unittest.TestCase):
    """What `--check` forgives, and what it must not.

    The column embeds the exact ROCm build that measured it, so a byte
    comparison fails on any host at a different patch level -- not something
    anyone can act on. But the fields saying *how* the probe was posed are
    ours, not the host's, and forgiving those forgives real defects.
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

    The safe default -- guessing would bake a false negative into the artifact
    -- but it is not an answer and must never reach a committed column. Each
    LLVM vintage words its diagnostics differently, so the wordings we have had
    to learn are pinned here.
    """

    def test_a_target_that_says_so_in_words_is_arch_absent(self):
        """llvm23 states it outright where llvm20 and llvm22 only fail.

        Left unrecognised, twelve cells across `permlane16`, `permlanex16` and
        `mov.dpp8` recorded `probe_error` for targets that had answered.
        """
        status, evidence = G._classify(
            1,
            "error: <unknown>:0:0: in function probe void (ptr, i32): "
            "intrinsic not supported on subtarget",
        )
        self.assertEqual(status, G.STATUS_ARCH_ABSENT)
        self.assertTrue(evidence)

    def test_an_illegal_transfer_size_asks_for_a_re_probe(self):
        """llvm23 rejects an out-of-range load-to-LDS size in the verifier,
        before codegen. The message is about our module, not the target, so it
        must route to the operand rescue rather than be recorded as an answer.
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

    def test_selection_dag_addresses_are_not_persisted(self):
        """Allocator-dependent node addresses must not make --check flaky."""
        status, evidence = G._classify(
            1,
            "fatal error: error in backend: Cannot select: "
            "0x32c38e80: ch = AMDGPUISD::OP 0x32c39660:1",
        )
        self.assertEqual(status, G.STATUS_ARCH_ABSENT)
        self.assertEqual(
            evidence,
            "fatal error: error in backend: Cannot select: "
            "0xADDR: ch = AMDGPUISD::OP 0xADDR:1",
        )

    def test_selection_dag_result_zero_addresses_are_not_persisted(self):
        """Result-zero operands are bare addresses in SelectionDAG dumps."""
        diagnostic = "Cannot select: 0x1234: i32,ch = OP 0x5678, 0x9abc:1"
        self.assertLess(len(diagnostic), 200)

        status, evidence = G._classify(1, diagnostic)

        self.assertEqual(status, G.STATUS_ARCH_ABSENT)
        self.assertEqual(
            evidence,
            "Cannot select: 0xADDR: i32,ch = OP 0xADDR, 0xADDR:1",
        )

    def test_selection_dag_untyped_addresses_are_not_persisted(self):
        """Match the capitalization LLVM uses for its Untyped value type."""
        diagnostic = "Cannot select: 0x1234: Untyped = OP 0x5678"
        self.assertLess(len(diagnostic), 200)

        status, evidence = G._classify(1, diagnostic)

        self.assertEqual(status, G.STATUS_ARCH_ABSENT)
        self.assertEqual(
            evidence,
            "Cannot select: 0xADDR: Untyped = OP 0xADDR",
        )

    def test_semantic_hex_values_are_preserved(self):
        """Only DAG node references are unstable; constants are evidence."""
        cases = (
            (
                "error: immediate 0xff: invalid",
                "error: immediate 0xff: invalid",
                G.STATUS_PROBE_ERROR,
            ),
            (
                "error: immediate 0xff: invalid; Cannot select: 0x1234: i32",
                "error: immediate 0xff: invalid; Cannot select: 0xADDR: i32",
                G.STATUS_ARCH_ABSENT,
            ),
            (
                "Cannot select: 0x1234: i32 = OP Constant:i32<0xff>, "
                "mask 0xaa: invalid, 0xabcd:1",
                "Cannot select: 0xADDR: i32 = OP Constant:i32<0xff>, "
                "mask 0xaa: invalid, 0xADDR:1",
                G.STATUS_ARCH_ABSENT,
            ),
            (
                "Cannot select: 0x1234: i32 = OP Constant:i32<0xff>, 0x5678",
                "Cannot select: 0xADDR: i32 = OP Constant:i32<0xff>, 0xADDR",
                G.STATUS_ARCH_ABSENT,
            ),
            (
                "Cannot select: 0x1234: i32 = "
                "OP<(load from 0xff, align 4)> 0x5678, 0x9abc:1",
                "Cannot select: 0xADDR: i32 = "
                "OP<(load from 0xff, align 4)> 0xADDR, 0xADDR:1",
                G.STATUS_ARCH_ABSENT,
            ),
        )
        for diagnostic, expected, expected_status in cases:
            with self.subTest(diagnostic=diagnostic):
                status, evidence = G._classify(1, diagnostic)
                self.assertEqual(status, expected_status)
                self.assertEqual(evidence, expected)


class AssemblerCheckTest(unittest.TestCase):
    """A probe that links is assembled for its target before it counts as ok.

    The backend can select an intrinsic for a target that lacks the instruction
    and write the encoding into the object without complaint; only the target's
    text assembler checks every instruction against the target.
    """

    def test_the_rejection_names_the_instruction_not_the_scratch_file(self):
        # llvm-mc's diagnostic carries the scratch path; the evidence must not,
        # or the committed column would change from host to host.
        diag = (
            "/tmp/probe-x1y2/k.s:14:2: error: instruction not supported on this GPU\n"
            "\tv_cvt_scalef32_pk_f32_fp8 v[2:3], v1, v0\n"
            "\t^\n"
        )
        self.assertEqual(
            G._assembler_rejection(diag),
            "assembler: instruction not supported on this GPU (v_cvt_scalef32_pk_f32_fp8)",
        )

    def test_an_instruction_the_target_lacks_is_arch_absent_though_it_links(self):
        # The fused scale-and-convert links for gfx942 once its flag operand is
        # a constant, and the object then holds a gfx950-only instruction.
        from check_ir_validity import _llvm_tool

        clang = _llvm_tool("clang")
        if clang is None or not G._llvm_mc_beside(clang).is_file():
            self.skipTest("no clang with an llvm-mc beside it on this host")
        decl = G._decl_table(_resolve_llvm_flavor())["amdgcn.cvt.scalef32.pk.f32.fp8"]
        module, why = G._probe_module(decl, "", literal_ints=G._LITERAL_PROBE_VALUES[0])
        self.assertFalse(why)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "probe.ll"
            path.write_text(module)
            results = {
                arch: G._probe(clang, path, arch, Path(scratch) / f"{arch}.o")
                for arch in ("gfx942", "gfx950")
            }
        self.assertEqual(results["gfx950"], (G.STATUS_OK, ""))
        status, evidence = results["gfx942"]
        self.assertEqual(status, G.STATUS_ARCH_ABSENT, evidence)
        self.assertTrue(evidence.startswith("assembler: "), evidence)


class DeclareParseTest(unittest.TestCase):
    """Every probe is built from a declare, so the parse is the whole table.

    Since generator 5 the probe module is synthesised from the declare `opt`
    hands back, which makes these three functions load-bearing for all 166 keys
    at once: a leaked attribute is malformed IR, and the key's entire row
    measures probe_error.

    Attributes are the moving part. LLVM adds them freely (`noundef`,
    `range(...)`, `nofpclass(...)`, LLVM 21 respelling `nocapture` as
    `captures(none)`) on both sides of a parameter's type, so the cases below
    are mostly attributes the current toolchains do not emit yet -- the parse
    has to survive the one nobody has written down.
    """

    def test_return_attributes_are_dropped_and_the_type_kept(self):
        for ret, want in (
            ("i32", "i32"),
            ("void", "void"),
            ("noundef i32", "i32"),
            ("align 4 ptr", "ptr"),
            # `range` and `nofpclass` take arguments containing type names and
            # spaces, so this scans for the type rather than stripping
            # attributes.
            ("range(i32 0, 1024) i32", "i32"),
            ("range(i64 -1, 100) i64", "i64"),
            ("nofpclass(nan inf) float", "float"),
            ("dereferenceable(16) noundef align 8 ptr", "ptr"),
            ("noundef zeroext i1", "i1"),
            # Aggregates and vectors: the type is more than one token.
            ("<2 x i32>", "<2 x i32>"),
            ("align 8 <4 x float>", "<4 x float>"),
            ("{ i32, i32 }", "{ i32, i32 }"),
            ("[2 x i64]", "[2 x i64]"),
            ("ptr addrspace(8)", "ptr addrspace(8)"),
        ):
            with self.subTest(ret=ret):
                self.assertEqual(G._return_type(ret), want)

    def test_parameter_attributes_are_dropped_from_both_sides(self):
        """Parameter attributes come before *and* after the type.

        Deleting a fixed list of attribute names lets the rest through: `i1
        immarg zeroext` parses as the type `i1 zeroext`, which does not
        assemble, and the key's whole row records probe_error.
        """
        for param, want in (
            ("ptr nocapture readonly", ("ptr", False)),
            ("i32 immarg", ("i32", True)),
            ("noundef i32", ("i32", False)),
            ("align 8 ptr nocapture", ("ptr", False)),
            # Attributes outside any list this file could keep.
            ("i1 immarg zeroext", ("i1", True)),
            ("float nofpclass(nan)", ("float", False)),
            ("ptr captures(none)", ("ptr", False)),
            ("i32 immarg range(i32 0, 4)", ("i32", True)),
            # `addrspace(N)` is part of the type, not an attribute, and the
            # probe branches on it to emit an addrspacecast.
            ("ptr addrspace(8)", ("ptr addrspace(8)", False)),
            ("ptr addrspace(8) nocapture readonly", ("ptr addrspace(8)", False)),
            # Aggregates and vectors, attributed on the right.
            ("<2 x i32>", ("<2 x i32>", False)),
            ("<4 x float> noundef", ("<4 x float>", False)),
            ("{ i32, i32 }", ("{ i32, i32 }", False)),
            ("[2 x i64] align 8", ("[2 x i64]", False)),
            ("metadata", ("metadata", False)),
        ):
            with self.subTest(param=param):
                self.assertEqual(G._param_type(param), want)

    def test_an_unrecognised_type_is_handed_back_whole(self):
        """Fail loudly, not quietly.

        If a future type opener is missed, returning a prefix produces IR that
        assembles into something subtly different; returning the string whole
        produces IR that does not assemble, which is a diagnostic.
        """
        self.assertEqual(G._return_type("mysterytype"), "mysterytype")
        self.assertEqual(G._param_type("mysterytype attr"), ("mysterytype attr", False))

    def test_top_level_commas_split_but_nested_ones_do_not(self):
        for params, want in (
            ("", []),
            ("i32", ["i32"]),
            ("i32, i32", ["i32", "i32"]),
            ("{ i32, i32 }, ptr", ["{ i32, i32 }", "ptr"]),
            ("<2 x i32>, i32 immarg", ["<2 x i32>", "i32 immarg"]),
            ("ptr addrspace(8), i32, i32", ["ptr addrspace(8)", "i32", "i32"]),
            ("range(i32 0, 4) i32, ptr", ["range(i32 0, 4) i32", "ptr"]),
        ):
            with self.subTest(params=params):
                self.assertEqual(G._split_params(params), want)

    def test_an_attributed_declare_builds_a_probe_that_assembles(self):
        """End to end, because the three parsers only matter together.

        Asserted on the synthesised text: a leaked attribute shows up here as
        `i1 zeroext %a3` in a kernel signature, which is what the toolchain
        would choke on.
        """
        decl = (
            "declare range(i32 0, 1024) noundef i32 @llvm.amdgcn.probe."
            "example(ptr addrspace(8) nocapture readonly, "
            "i32 immarg zeroext, <2 x i32> noundef)"
        )
        text, err = G._probe_module(decl, "e-p:64:64", imm_int=4)
        self.assertEqual(err, "")
        self.assertIn("%r = call i32 @llvm.amdgcn.probe.example(", text)
        # The buffer fat pointer is the one operand still passed poison; the
        # `immarg` takes the swept value; everything else becomes a kernel arg.
        self.assertIn("ptr addrspace(8) poison", text)
        self.assertIn("i32 4", text)
        self.assertIn("<2 x i32> %a2", text)
        self.assertIn("store volatile i32 %r", text)
        for leaked in ("zeroext", "noundef", "nocapture", "readonly", "range("):
            with self.subTest(attr=leaked):
                self.assertNotIn(leaked, text.split("define", 1)[1])

    def test_an_attributed_addrspace_param_still_takes_the_cast_branch(self):
        """`ptr addrspace(N)` must survive its attributes as one type.

        The probe branches on the exact string `ptr addrspace(N)` to emit an
        `addrspacecast`. A trailing attribute misses that branch and falls
        through to the kernel-argument case, which is a different question.
        """
        decl = (
            "declare void @llvm.amdgcn.probe.cast("
            "ptr addrspace(1) nocapture writeonly, noundef i32)"
        )
        text, err = G._probe_module(decl, "e-p:64:64")
        self.assertEqual(err, "")
        self.assertIn("%p0 = addrspacecast ptr %base to ptr addrspace(1)", text)
        self.assertIn("call void @llvm.amdgcn.probe.cast(ptr addrspace(1) %p0,", text)


class ImmargSweepTest(unittest.TestCase):
    """A negative must survive being asked with a legal operand value.

    The probe passes 0 for every `immarg`, and 0 is not a legal transfer size.
    ISel rejects it with `Cannot select`, which is also what a genuinely
    unsupported target says -- so re-asking with a different value is the only
    discriminator. Both directions matter: a false `arch_absent` breaks a build
    loudly, a false `ok` admits a kernel that cannot run.
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
        """The generator decides what is sweepable by comparing module text, so
        a declare the sweep cannot vary must produce an identical module --
        otherwise every negative costs four probes that change nothing.
        """
        decl = "declare i32 @llvm.probe.plain(ptr addrspace(1) nocapture, i32)"
        base, _ = G._probe_module(decl, "e-p:64:64-i64:64")
        for imm in G._IMMARG_PROBE_VALUES:
            text, _ = G._probe_module(decl, "e-p:64:64-i64:64", imm_int=imm)
            self.assertEqual(text, base, f"imm_int={imm} varied a non-immarg decl")

    def test_a_rescued_cell_says_which_value_answered(self):
        """Without it, a first-try `ok` and a fourth-candidate `ok` look alike.

        The value is the evidence that the cell was measured with a legal
        operand, and it cannot be re-derived by hand afterwards.
        """
        for flavor, path in _columns():
            doc = json.loads(path.read_text())
            for key, row in doc["keys"].items():
                for arch, cell in row.items():
                    if "probe_imm" not in cell:
                        continue
                    with self.subTest(flavor=flavor, key=key, arch=arch):
                        self.assertIn(cell["probe_imm"], G._IMMARG_PROBE_VALUES)
                        # The sweep only returns early on a win, so a value on
                        # a negative cell would be reported for a probe it did
                        # not decide.
                        self.assertEqual(cell["status"], G.STATUS_OK)

    def test_global_load_lds_is_available_on_cdna_and_not_on_rdna(self):
        """The case that motivated the sweep, pinned in both directions.

        The intrinsic takes a per-lane transfer size, legal at 1/2/4 on CDNA
        and also 16 on gfx950, and it is on a live path (builder entry point,
        handler in both engines, production instance), so an all-`arch_absent`
        row is a false negative. RDNA really cannot lower it, and must still
        say so after the full sweep.
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

    A column written without the measurement behind it still carries its own
    provenance and reads as authoritative, so nothing downstream catches it;
    the stop has to be at generation time.

    Driven through `main()` with stub tools, because the policy under test is
    "what do we do when the toolchain does not answer".
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

        `opt` owns the flavor axis: without it nothing separates "this LLVM has
        no such intrinsic" from "this target cannot lower it", and a
        `name_absent` key can record `ok` because an unresolved `llvm.*` name
        is an ordinary external call a link may accept.

        It fails rather than reporting UNVALIDATED like the no-clang path: no
        complete LLVM install ships clang without `opt`, so finding one and not
        the other is a broken environment, not a non-probing host.
        """
        out = Path(tempfile.mkdtemp()) / "column.json"
        self.addCleanup(shutil.rmtree, out.parent, True)
        clang = self._stub("clang", self._CLANG_REJECTS)
        rc = self._run({"clang": clang, "opt": None}, out)
        self.assertEqual(rc, 1, "a partial toolchain is a broken env, not a no-op")
        self.assertFalse(out.exists(), "wrote a column with no name check behind it")

    def test_an_unclassified_probe_writes_nothing_and_fails(self):
        """`probe_error` is this tool reporting itself broken, not a fact about
        the toolchain, and the artifact gate forbids it in a committed column --
        so a generator that emits one writes a file its own test rejects.
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

    def test_a_disagreeing_opt_and_clang_write_nothing_and_fail(self):
        """One sweep, one toolchain.

        `opt` measures the flavor axis and `clang` the arch axis, found by
        independent searches. A prefix shipping clang but not `opt` lets `opt`
        fall through to PATH, and the provenance block records clang alone --
        filing the column under a vintage that measured half of it.
        """
        out = Path(tempfile.mkdtemp()) / "column.json"
        self.addCleanup(shutil.rmtree, out.parent, True)
        tools = {
            "clang": self._stub(
                "clang", 'import sys\nprint("AMD clang version 22.0.0git")\n'
            ),
            "opt": self._stub(
                "opt", 'import sys\nprint("AMD LLVM version 20.0.0git")\n'
            ),
        }
        # Pin the flavor so the earlier clang-vs-rocke check passes on any host
        # and the run reaches the pairing check.
        with mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm22"}):
            rc = self._run(tools, out)
        self.assertEqual(rc, 2, "two vintages measured one column")
        self.assertFalse(out.exists(), "wrote a column measured by two LLVMs")

    def test_an_unparseable_banner_is_not_a_disagreement(self):
        """Unknown must never read as mismatch: a vendor rewording its version
        banner would otherwise red every generating host.
        """
        self.assertIsNone(G.flavor_of_clang("some vendor reworded this"))
        for banner, flavor in (
            ("AMD clang version 22.0.0git (...)", "llvm22"),
            ("AMD LLVM version 22.0.0git", "llvm22"),
        ):
            with self.subTest(banner=banner):
                self.assertEqual(G.flavor_of_clang(banner), flavor)


class PartialSweepTest(unittest.TestCase):
    """A filtered sweep must never land on the committed column.

    `--only` and `--arch` measure a subset, but the document written is a whole
    column. Writing a subset to the default path deletes the unselected cells
    -- one documented `--only` invocation took a 166-key column down to one key
    -- and every lost lookup silently becomes no-data.

    The guard runs before any toolchain resolution, so these need no stubs and
    a bad invocation fails at once rather than after a full sweep.
    """

    def _run(self, *args: str) -> int:
        with mock.patch.object(sys, "argv", [str(_TOOL), *args]):
            return G.main()

    def _snapshot(self) -> dict[str, str]:
        return {p.name: p.read_text() for _f, p in _columns()}

    def test_a_filtered_sweep_without_an_out_refuses(self):
        for args in (
            ("--only", "fabs"),
            ("--arch", "gfx942"),
            ("--only", "fabs", "--arch", "gfx942"),
        ):
            with self.subTest(args=args):
                before = self._snapshot()
                self.assertEqual(self._run(*args), 2, f"{args} reached the writer")
                self.assertEqual(before, self._snapshot(), "a committed column moved")

    def test_a_filtered_check_refuses_rather_than_reporting_false_drift(self):
        """`--check` with a filter would diff a subset against the full column.

        Harmless to the file, but it reports the artifact stale when it is not,
        so both modes ask for a destination.
        """
        before = self._snapshot()
        self.assertEqual(self._run("--only", "fabs", "--check"), 2)
        self.assertEqual(before, self._snapshot())

    def test_a_filtered_sweep_with_an_out_is_let_through(self):
        """The guard keys on the missing destination, not on the filter.

        A guard that simply rejected `--only`/`--arch` would pass the tests
        above while deleting the documented way to probe one key. Driven to the
        no-clang exit (rc 0, UNVALIDATED): reaching a decision downstream of
        toolchain resolution proves it cleared the guard, at no probing cost.
        """
        out = Path(tempfile.mkdtemp()) / "column.json"
        self.addCleanup(shutil.rmtree, out.parent, True)
        argv = [str(_TOOL), "--out", str(out), "--only", "fabs", "--arch", "gfx942"]
        import check_ir_validity as V

        with (
            mock.patch.object(V, "_llvm_tool", lambda _n: None),
            mock.patch.object(sys, "argv", argv),
        ):
            self.assertEqual(G.main(), 0, "the guard swallowed a well-formed probe")


class ValidityGateTest(unittest.TestCase):
    """`tools/check_ir_validity.py` -- the sibling gate, same two rules."""

    def test_both_gates_ask_the_question_at_O0(self):
        """The one flag that decides whether either gate proves anything.

        At `-O3` the IR pipeline can fold or delete the construct being asked
        about, and the module then links because the suspect instruction is
        gone. Pinned together so the two gates cannot drift apart.
        """
        import check_ir_validity as V

        self.assertIn("-O0", V.COMPILE_CFLAGS)
        self.assertIn("-O0", G.PROBE_CFLAGS)
        argv = V._compile_argv("clang", Path("m.ll"), "gfx942", Path("m.hsaco"))
        self.assertIn("-O0", argv)
        self.assertNotIn("-O3", argv)

    def test_a_bundled_comgr_does_not_send_the_lookup_to_path(self):
        """A comgr bundled with another package (the ROCm torch wheel) has no LLVM
        beside it. The lookup must then find the installed ROCm toolchain, not the
        first clang on PATH, which is typically a distro clang that cannot link
        AMDGPU code objects."""
        from unittest import mock

        import check_ir_validity as V
        from rocke.runtime.runtime_coexistence import _rocm_root_libdirs

        # Only installs that ship a compiler can answer: a runtime-only ROCm has
        # no <root>/llvm/bin/clang, and the lookup then rightly finds none.
        installs = [
            d
            for d in _rocm_root_libdirs()
            if any(
                (root / "llvm" / "bin" / "clang").is_file()
                for root in list(Path(d).resolve().parents)[:3]
            )
        ]
        if not installs:
            self.skipTest("no ROCm install with an LLVM toolchain on this host")
        with tempfile.TemporaryDirectory() as bundled:
            fake_comgr = Path(bundled) / "lib" / "libamd_comgr.so"
            fake_comgr.parent.mkdir()
            fake_comgr.touch()
            env = {k: v for k, v in os.environ.items() if k != "ROCKE_LLVM_BIN"}
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch(
                    "rocke.runtime.comgr.resolved_lib_path",
                    return_value=str(fake_comgr),
                ),
            ):
                found = V._llvm_tool("clang")
        self.assertIsNotNone(found)
        self.assertTrue(
            any(
                Path(found).resolve().is_relative_to(Path(d).resolve().parents[1])
                for d in installs
            ),
            f"{found} is not inside a ROCm install {installs}",
        )

    def test_a_target_this_clang_cannot_build_is_not_an_emission_failure(self):
        """rocke wires arches ahead of the toolchains that can build for them.

        gfx1250 against the LLVM 20 in ROCm 7.1 is rejected in the driver,
        before clang reads an instruction, so blaming the lowerer would red
        every host whose ROCm predates the newest wired arch. Same line the
        generator draws with `target_unsupported`.
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
        llvm23, and that refusal is the emitter working, not a FATAL that reds
        every llvm22 host. Matched on the gate's wording rather than the
        exception type: a `NotImplementedError` from anywhere else in the
        lowerer is still a validity failure.
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
        # The llvm23-gated target ops (TDM tensor moves, ...) word it this way.
        self.assertTrue(
            V._requires_newer_flavor(
                ValueError("tensor_load_to_lds requires LLVM flavor llvm23, got llvm22")
            )
        )
        for exc in (
            NotImplementedError("no handler for op 'tile.frobnicate'"),
            NotImplementedError("unsupported dtype combination"),
            ValueError("tensor_load_to_lds requires a target with tdm, got gfx950"),
        ):
            with self.subTest(exc=str(exc)):
                self.assertFalse(V._requires_newer_flavor(exc))

    def test_both_gates_read_a_clang_banner_the_same_way(self):
        """The two tools act on a vintage mismatch differently on purpose --
        the generator refuses, this gate reports UNVALIDATED -- but must not
        disagree about what a mismatch *is*, so they share `_llvm_identity`.
        An unrecognised banner reads as "unknown", never as "mismatch".
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

    def test_an_opt_from_another_install_loses_l0_rather_than_the_run(self):
        """`opt` and clang are discovered independently, and only clang is
        asked what it is.

        A prefix shipping clang but not `opt` lets `opt` fall through to PATH,
        and an older verifier rejecting a construct its parser predates reds
        the case as L0-VERIFY. The generator refuses outright because it needs
        both tools; this gate only loses a lane clang re-reaches at L1, so it
        skips L0 and says so.
        """
        import check_ir_validity as V

        v20 = "AMD clang version 20.0.0git (roc-7.1.1)"
        v22 = "AMD clang version 22.0.0git (roc-7.2.0)"
        with mock.patch.object(V, "tool_identity", return_value=v20):
            reason = V._unpaired_opt("/a/opt", "/b/clang", v22, "llvm22")
        self.assertIsNotNone(reason)
        self.assertIn("SKIPPED", reason)
        # Both paths named, or the reader cannot act on it.
        self.assertIn("/a/opt", reason)
        self.assertIn("/b/clang", reason)

        # Matched pair: the lane stays.
        with mock.patch.object(V, "tool_identity", return_value=v22):
            self.assertIsNone(V._unpaired_opt("/a/opt", "/b/clang", v22, "llvm22"))

        # An unreadable banner is not evidence of a mismatch -- guessing would
        # disable L0 on every toolchain whose version line we have not seen.
        for banner, clang_flavor in ((v20, None), ("who knows", "llvm22")):
            with (
                self.subTest(banner=banner),
                mock.patch.object(V, "tool_identity", return_value=banner),
            ):
                self.assertIsNone(
                    V._unpaired_opt("/a/opt", "/b/clang", v22, clang_flavor)
                )

    def test_both_corpus_gates_skip_a_newer_flavor_the_same_way(self):
        """Both gates lower the shared corpus, so both meet an instance that
        declines the flavor they asked for. Neither may count that as a
        failure, and they must not disagree about which refusals qualify, so
        they share `_flavor_gate`.

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

        # "We cannot answer here" is a skip: failing it would red every CI
        # machine whose ROCm differs from the one that blessed the artifact.
        if "UNVALIDATED" in out:
            self.skipTest(out.strip().splitlines()[-1])
        if proc.returncode == 2 and "clang reports" in out:
            self.skipTest("toolchain flavor disagrees with the resolved flavor")

        self.assertEqual(proc.returncode, 0, out)


if __name__ == "__main__":
    unittest.main()
