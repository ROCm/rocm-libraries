# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The sweep, probe and audit CLIs, exercised as programs.

These are not unit tests of helper functions. Every case below runs the actual
`tools/*.py` entry point as a subprocess, from a directory unrelated to the
generator tree and to the sweep's own inputs, because the defects these guard
against were all in the wiring: a driver that resolved a path against the caller's
cwd, a gate that never ran because the phase short-circuited, a resume that trusted
a filename.

CONTROL-FLOW FIXTURES. `bin/rocminfo`, `bin/python3` and each install tree's
`bin/hipdnn_list_engines` below are FIXTURES, not models of the real tools. They
exist so the driver's control flow -- which gate fires, which phase re-runs, what
the resume decision is -- can be steered deterministically without a GPU. They
emit the field names and shapes the real `dnn_benchmarking` result schema defines
(`reporting/suite_results.py`: `SuiteResult.to_dict` ->
`{"metadata", "graphs"}`, `GraphResult.to_dict` ->
`{"graph_name", "graph_path", "results"}`, `ProviderEngineResult.to_dict` ->
`provider`/`engine_id`/`engine_name`/`status`/`role`/`plugin_path`/
`gpu_kernel_stats`/`correctness`, and `CorrectnessResult.to_dict` ->
`passed`/`execution_success`/`tolerance_match`/`rtol`/`atol`) and nothing more.
They prove NOTHING about the real benchmark, the real engine, or any device: no
case here establishes that a kernel ran, that a number is correct, or that an
install tree is loadable. What they establish is that when the evidence says a
given thing, the driver reaches the intended verdict.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_SWEEP = _TOOLS / "sweep.py"
_PROBE = _TOOLS / "device_probe.py"
_AUDIT = _TOOLS / "field_audit.py"

_ENGINE_NAME = "test:Engine"
_ENGINE_ID = 0x1A2B
_ARCH = "gfx942"

#: CONTROL-FLOW FIXTURE. Stands in for `rocminfo` so the arch gate has a definite
#: answer. Emits one agent line in the shape the token scan reads.
_ROCMINFO = """\
import os, sys
print("Agent 1")
print("  Name:                    AMD Ryzen")
print("Agent 2")
print("  Name:                    " + os.environ.get("FAKE_ROCMINFO_ARCH", "{arch}"))
sys.exit(int(os.environ.get("FAKE_ROCMINFO_RC", "0")))
"""

#: CONTROL-FLOW FIXTURE. Stands in for the installed engine registry so discovery
#: resolves one name to one ID, in the exact line shape the driver parses.
_LIST_ENGINES = """\
import sys
print("Engines:")
print("  {name} (0x{eid:x})")
"""

#: CONTROL-FLOW FIXTURE, and deliberately TWO programs in one file.
#:
#: The driver identifies the benchmark's Python environment by running the same
#: executable with `-c <probe>`, then runs it again as the benchmark. A single file
#: answering both is what lets a test control the whole benchmark side without a
#: venv: the identity answer stays cheap and content-bound, and the benchmark answer
#: is steered entirely by the scenario JSON at $FAKE_SCENARIO.
_FAKE_BENCH = """\
import glob, json, os, sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    print(json.dumps({"python": sys.executable, "roots": [],
                      "files": [str(Path(__file__).resolve())]}))
    raise SystemExit(0)

scenario = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
args = sys.argv[1:]


def value(flag):
    return args[args.index(flag) + 1] if flag in args else None


graphs = sorted(glob.glob(value("--graph")))
plugin_dir = Path(value("--plugin-path"))
engine_id = int(value("--engine"))
validating = "--validate" in args

log = os.environ.get("HIPDNN_LOG_FILE")
if log and not scenario.get("skip_provenance"):
    Path(log).write_text(
        "info: load plugin from [" + str(plugin_dir / "engine.so") + "]\\n")
elif log:
    Path(log).write_text("info: no plugin was loaded\\n")

stats = {"mean_ms": 1.5, "median_ms": 1.5, "std_ms": 0.0, "min_ms": 1.4,
         "max_ms": 1.6, "p95_ms": 1.6, "p99_ms": 1.6, "total_ms": 3.0}
served_limit = scenario.get("served", len(graphs))
results = []
passed = failed = skipped = errored = 0
for index, path in enumerate(graphs):
    name = json.loads(Path(path).read_text())["name"]
    rows = []
    if index < served_limit:
        row = {"provider": "{engine}", "engine_id": engine_id,
               "engine_name": "{engine}", "engine_version": "1.0",
               "started_at": "2026-01-01T00:00:00+00:00", "status": "success",
               "plugin_path": str(plugin_dir / "engine.so"),
               "cpu_build_time_ms": 2.0, "host_stats": stats,
               "elapsed_time_ms": 9.0}
        timing = scenario.get("timing", "ok")
        if timing == "ok":
            row["gpu_kernel_stats"] = stats
        elif timing == "null":
            row["gpu_kernel_stats"] = None
        elif timing == "zero":
            row["gpu_kernel_stats"] = dict(stats, mean_ms=0.0)
        if validating:
            match = scenario.get("tolerance", True)
            row["correctness"] = {"passed": bool(match), "execution_success": True,
                                  "tolerance_match": match, "rtol": 0.01,
                                  "atol": 0.01, "max_abs_diff": 0.001,
                                  "max_rel_diff": 0.002}
            if match is False:
                failed += 1
        if scenario.get("tolerance", True) is not False:
            passed += 1
        rows.append(row)
    else:
        rows.append({"provider": "{engine}", "engine_id": engine_id,
                     "engine_name": "{engine}", "engine_version": "1.0",
                     "started_at": "2026-01-01T00:00:00+00:00",
                     "status": "skipped",
                     "skip_reason": "head_size unsupported by this variant set"})
        skipped += 1
    if validating:
        reference = {"provider": "pytorch", "engine_id": 0,
                     "engine_name": "pytorch", "engine_version": "2.0",
                     "started_at": "2026-01-01T00:00:00+00:00",
                     "role": "reference", "status": "success",
                     "gpu_kernel_stats": stats, "host_stats": stats,
                     "elapsed_time_ms": 9.0, "cpu_build_time_ms": 1.0,
                     "correctness": {"passed": False, "execution_success": True,
                                     "tolerance_match": None, "rtol": 0.01,
                                     "atol": 0.01}}
        if scenario.get("reference") == "skipped":
            reference = {"provider": "pytorch", "engine_id": 0,
                         "engine_name": "pytorch", "engine_version": "2.0",
                         "started_at": "2026-01-01T00:00:00+00:00",
                         "role": "reference", "status": "skipped",
                         "skip_reason": "torch is not available"}
        rows.append(reference)
    results.append({"graph_name": name, "graph_path": path, "results": rows})

document = {
    "metadata": {"timestamp": "2026-01-01T00:00:00+00:00", "hostname": "fixture",
                 "total_graphs": scenario.get("total_graphs", len(graphs)),
                 "total_combinations": passed + failed + skipped + errored,
                 "pass_combinations": passed,
                 "fail_combinations": scenario.get("fail_combinations", failed),
                 "skip_combinations": skipped,
                 "error_combinations": scenario.get("error_combinations", errored),
                 "gpu_arch": scenario.get("gpu_arch", "{arch}")},
    "graphs": results,
}
out = Path(value("-o"))
text = json.dumps(document, indent=2)
if scenario.get("truncate_output"):
    text = text[: len(text) // 2]
out.write_text(text)
raise SystemExit(scenario.get("rc", 0))
"""


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _graph(index: int) -> dict:
    return {
        "name": f"graph_{index}",
        "tensors": [
            {"uid": 1, "name": "query", "dims": [1, 8, 512, 128]},
            {"uid": 2, "name": "key", "dims": [1, 8, 512, 128]},
            {"uid": 3, "name": "value", "dims": [1, 8, 512, 128]},
        ],
        "nodes": [{"type": "SdpaAttributes", "attributes": {"q_tensor_uid": 1}}],
    }


class Sweep:
    """One fully-staged sweep: fixtures, inputs, config, and a way to run it."""

    def __init__(self, tmp_path: Path, *, correctness: bool = False, graphs: int = 3):
        self.root = tmp_path / "sweeps"
        self.bin = tmp_path / "bin"
        self.elsewhere = tmp_path / "unrelated-cwd"
        self.elsewhere.mkdir(parents=True)
        self.scenario_path = tmp_path / "scenario.json"
        self.scenario_path.write_text("{}")

        _script(self.bin / "rocminfo", _ROCMINFO.format(arch=_ARCH))
        _script(
            self.bin / "python3",
            _FAKE_BENCH.replace("{engine}", _ENGINE_NAME).replace("{arch}", _ARCH),
        )

        self.corpus = self.root / "corpora" / "main"
        self.corpus.mkdir(parents=True)
        for index in range(graphs):
            (self.corpus / f"g{index}.json").write_text(json.dumps(_graph(index)))

        self.install = self.root / "arm-a"
        _script(
            self.install / "bin" / "hipdnn_list_engines",
            _LIST_ENGINES.format(name=_ENGINE_NAME, eid=_ENGINE_ID),
        )
        (self.install / "lib" / "hipdnn_plugins" / "engines").mkdir(parents=True)
        (self.install / "lib" / "hipdnn_plugins" / "engines" / "engine.so").write_text(
            "so"
        )
        (self.install / "descriptors").mkdir()
        self.descriptors = self.install / "descriptors" / "pack.kdp.json"
        self.descriptors.write_text(
            json.dumps({"kernelDescriptors": [{"a": 1}, {"b": 2}]})
        )

        self.config_path = tmp_path / "sweep.yaml"
        self.config_path.write_text(
            json.dumps(
                {
                    "sweep_root": str(self.root),
                    "output_dir": str(self.root / "results"),
                    "corpus_dir": str(self.root / "corpora"),
                    "arch": _ARCH,
                    "engine_name": _ENGINE_NAME,
                    "engine_ued_name": _ENGINE_NAME,
                    "corpora": [
                        {
                            "name": "main",
                            "path": str(self.corpus),
                            "expected_graphs": graphs,
                        }
                    ],
                    "arms": [
                        {
                            "name": "a",
                            "install_tree": str(self.install),
                            "expected_descriptors": 2,
                        }
                    ],
                    "warmup_arm": None,
                    "rounds": 1,
                    "min_served": 2,
                    "exclude_tensors": "none",
                    "benchmark": {
                        "argv": [str(self.bin / "python3")],
                        "warmup": 1,
                        "iters": 2,
                    },
                    "correctness": {
                        "enabled": correctness,
                        "reference": "pytorch",
                        "warmup": 1,
                        "iters": 1,
                    },
                },
                indent=2,
            )
        )

    def scenario(self, **keys) -> None:
        self.scenario_path.write_text(json.dumps(keys))

    def run(self) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PATH"] = str(self.bin) + os.pathsep + env.get("PATH", "")
        env["FAKE_SCENARIO"] = str(self.scenario_path)
        return subprocess.run(
            [sys.executable, str(_SWEEP), "--config", str(self.config_path)],
            cwd=self.elsewhere,
            env=env,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def gates(result: subprocess.CompletedProcess, kind: str = "timing") -> dict:
        """The gate map the driver printed for one phase, parsed from its own line."""
        pattern = rf"^{kind}__\S+: (?:PASS|FAIL) (\{{.*\}})$"
        matches = re.findall(pattern, result.stdout, flags=re.MULTILINE)
        assert matches, f"no {kind} phase line in:\n{result.stdout}\n{result.stderr}"
        return json.loads(matches[-1])

    @staticmethod
    def resumed(result: subprocess.CompletedProcess, kind: str = "timing") -> bool:
        return bool(
            re.search(rf"^{kind}__\S+: SKIP", result.stdout, flags=re.MULTILINE)
        )


@pytest.fixture
def sweep(tmp_path):
    return Sweep(tmp_path)


class TestTheCLIsRunAsPrograms:
    """From an unrelated cwd, with no inherited environment doing the work."""

    def test_a_clean_timing_sweep_completes(self, sweep):
        result = sweep.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SWEEP_TIMING_ONLY" in result.stdout
        assert all(sweep.gates(result).values())

    def test_a_correctness_sweep_reports_a_validated_run(self, tmp_path):
        staged = Sweep(tmp_path, correctness=True)
        result = staged.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SWEEP_DONE" in result.stdout
        assert staged.gates(result, "correctness")["correctness"] is True

    def test_the_device_probe_refuses_an_inexact_arch(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(_PROBE),
                "--mode",
                "early",
                "--arch",
                "gfx9",
                "--sweep-root",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "exact gfx architecture token" in result.stderr

    def test_early_mode_makes_no_installation_claim(self, tmp_path):
        """Early feasibility must not accept --install, and must ignore an inherited
        INSTALL: the runbook's early gate precedes any build."""
        env = dict(os.environ, INSTALL=str(tmp_path / "not-a-tree"))
        rejected = subprocess.run(
            [
                sys.executable,
                str(_PROBE),
                "--mode",
                "early",
                "--arch",
                _ARCH,
                "--sweep-root",
                str(tmp_path),
                "--install",
                str(tmp_path),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "early mode rejects --install" in rejected.stderr

        accepted = subprocess.run(
            [
                sys.executable,
                str(_PROBE),
                "--mode",
                "early",
                "--arch",
                _ARCH,
                "--sweep-root",
                str(tmp_path),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert "not-a-tree" not in accepted.stdout + accepted.stderr

    def test_installed_mode_requires_an_install_tree(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(_PROBE),
                "--mode",
                "installed",
                "--arch",
                _ARCH,
                "--sweep-root",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "installed mode requires --install" in result.stderr

    def test_the_field_audit_names_every_unreferenced_field(self, tmp_path):
        schema = tmp_path / "op_attributes.fbs"
        schema.write_text(
            "// a comment: ignored_by_comment_stripping:int;\n"
            "table OpAttributes {\n"
            "  head_size:int;\n"
            "  sliding_window:int;\n"
            "}\n"
        )
        source = tmp_path / "Native.cpp"
        source.write_text(
            "auto h = attrs.head_size();\n// sliding_window mentioned only in prose\n"
        )
        result = subprocess.run(
            [sys.executable, str(_AUDIT), str(schema), str(source)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "UNCHECKED: sliding_window" in result.stdout
        assert "UNCHECKED: head_size" not in result.stdout
        assert "ignored_by_comment_stripping" not in result.stdout

    def test_the_field_audit_refuses_more_than_one_schema(self, tmp_path):
        first = tmp_path / "a.fbs"
        first.write_text("table A { x:int; }\n")
        second = tmp_path / "b.fbs"
        second.write_text("table B { y:int; }\n")
        result = subprocess.run(
            [sys.executable, str(_AUDIT), str(first), str(second)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "exactly one .fbs schema" in result.stderr


class TestGatesFailIndependently:
    """Each gate must be able to be the ONLY thing that failed.

    A driver where one gate's failure suppresses another's evaluation reports a
    single cause for a run with several, and the second one is then found only by a
    later, more expensive step.
    """

    def _only_failing(self, gates: dict) -> set:
        return {name for name, value in gates.items() if not value}

    def test_a_nonzero_benchmark_status_fails_the_run_despite_valid_evidence(
        self, sweep
    ):
        """rc 7 with a complete, well-formed, fully served timing artifact. The
        artifact is exactly what a passing run produces; only the status differs."""
        sweep.scenario(rc=7)
        result = sweep.run()
        assert result.returncode == 1
        assert "SWEEP_INCOMPLETE" in result.stdout
        assert self._only_failing(sweep.gates(result)) == {"command"}

    def test_a_missing_timing_number_fails_only_the_outcome_gate(self, sweep):
        sweep.scenario(timing="null")
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"served", "outcomes"}

    def test_a_nonpositive_timing_number_is_not_a_measurement(self, sweep):
        sweep.scenario(timing="zero")
        result = sweep.run()
        assert result.returncode == 1
        assert "outcomes" in self._only_failing(sweep.gates(result))

    def test_an_unloaded_plugin_fails_only_the_provenance_gate(self, sweep):
        """Every number is present and plausible; nothing in the log says this
        engine's plugin was ever loaded, so the numbers are another engine's."""
        sweep.scenario(skip_provenance=True)
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"provenance"}

    def test_too_few_served_graphs_fails_only_the_served_gate(self, sweep):
        """Declines are a legitimate outcome, so the ledger is clean -- the count is
        the only thing wrong, and it is what catches a dropped engine."""
        sweep.scenario(served=1)
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"served"}

    def test_a_wrong_descriptor_census_fails_only_the_descriptor_gate(self, sweep):
        sweep.descriptors.write_text(json.dumps({"kernelDescriptors": [{"a": 1}]}))
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"descriptors"}

    def test_a_truncated_result_document_is_a_failure_not_an_absence(self, sweep):
        sweep.scenario(truncate_output=True)
        result = sweep.run()
        assert result.returncode == 1
        failing = self._only_failing(sweep.gates(result))
        assert {"parsed_inventory", "metadata", "served", "outcomes"} <= failing

    def test_a_suite_reporting_another_arch_fails_the_metadata_gate(self, sweep):
        """The device gate established the arch on this host. A result document
        claiming a different one is not this sweep's evidence."""
        sweep.scenario(gpu_arch="gfx950")
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"metadata"}

    def test_a_suite_reporting_a_failed_combination_fails_the_metadata_gate(
        self, sweep
    ):
        sweep.scenario(fail_combinations=1)
        result = sweep.run()
        assert result.returncode == 1
        assert self._only_failing(sweep.gates(result)) == {"metadata"}


class TestCorrectnessEvidenceIsRequiredNotOptional:
    def test_a_tolerance_mismatch_fails_the_correctness_phase(self, tmp_path):
        staged = Sweep(tmp_path, correctness=True)
        staged.scenario(tolerance=False)
        result = staged.run()
        assert result.returncode == 1
        assert staged.gates(result, "correctness")["correctness"] is False

    def test_an_unreported_comparison_is_not_a_pass(self, tmp_path):
        """`tolerance_match: null` is what the suite emits when no comparison ran,
        and it counts it as a pass. The sweep must not."""
        staged = Sweep(tmp_path, correctness=True)
        staged.scenario(tolerance=None)
        result = staged.run()
        assert result.returncode == 1
        assert staged.gates(result, "correctness")["correctness"] is False

    def test_a_skipped_reference_provider_fails_the_reference_gate(self, tmp_path):
        """The reference silently skipping is the dangerous shape: every engine row
        still says success, the suite exits 0, and nothing was compared."""
        staged = Sweep(tmp_path, correctness=True)
        staged.scenario(reference="skipped")
        result = staged.run()
        assert result.returncode == 1
        gates = staged.gates(result, "correctness")
        assert gates["reference"] is False

    def test_a_timing_only_run_never_claims_validation(self, sweep):
        result = sweep.run()
        assert "SWEEP_DONE" not in result.stdout
        summary = json.loads((sweep.root / "results" / "summary.json").read_text())
        assert summary["validated_complete"] is False
        assert summary["timing_only_complete"] is True


class TestResume:
    """A completed phase is reusable only when it is bound to the CURRENT inputs and
    passed EVERY gate. Each of the three ways that binding is broken is tested."""

    def test_an_unchanged_rerun_resumes(self, sweep):
        """The control. Without it, every assertion below passes vacuously."""
        assert sweep.run().returncode == 0
        again = sweep.run()
        assert again.returncode == 0
        assert sweep.resumed(again), again.stdout

    def test_a_failed_phase_is_rerun_not_resumed(self, sweep):
        """A failed served-count gate leaves no reusable record, so fixing the cause
        must re-measure rather than adopt the failure or skip past it."""
        sweep.scenario(served=1)
        assert sweep.run().returncode == 1
        sweep.scenario()
        recovered = sweep.run()
        assert recovered.returncode == 0
        assert not sweep.resumed(recovered)
        assert all(sweep.gates(recovered).values())

    def test_an_edited_corpus_invalidates_the_resume(self, sweep):
        """Same filenames, same count, same timestamps-as-far-as-anyone-checks --
        different content. Binding to names would resume a different experiment."""
        assert sweep.run().returncode == 0
        graph = json.loads((sweep.corpus / "g0.json").read_text())
        graph["tensors"][0]["dims"] = [2, 8, 512, 128]
        (sweep.corpus / "g0.json").write_text(json.dumps(graph))
        again = sweep.run()
        assert again.returncode == 0
        assert not sweep.resumed(again), "an edited corpus must not resume"

    def test_an_edited_install_tree_invalidates_the_resume(self, sweep):
        assert sweep.run().returncode == 0
        plugin = sweep.install / "lib" / "hipdnn_plugins" / "engines" / "engine.so"
        plugin.write_text("rebuilt")
        again = sweep.run()
        assert again.returncode == 0
        assert not sweep.resumed(again), "a rebuilt install tree must not resume"

    def test_an_edited_config_invalidates_the_resume(self, sweep):
        assert sweep.run().returncode == 0
        config = json.loads(sweep.config_path.read_text())
        config["benchmark"]["iters"] = 5
        sweep.config_path.write_text(json.dumps(config))
        again = sweep.run()
        assert again.returncode == 0
        assert not sweep.resumed(again), "a changed measurement config must not resume"

    def test_an_interrupted_write_is_never_treated_as_success(self, sweep):
        """A completion record that was being written when the job died. Half a
        record is not a phase that passed."""
        assert sweep.run().returncode == 0
        sidecars = sorted((sweep.root / "results").glob("timing__*.complete.json"))
        assert sidecars, "the control run wrote no completion record"
        text = sidecars[0].read_text()
        sidecars[0].write_text(text[: len(text) // 2])
        again = sweep.run()
        assert again.returncode == 0
        assert not sweep.resumed(again), "a truncated record must not resume"

    def test_a_record_whose_evidence_was_edited_is_not_resumed(self, sweep):
        """The record is intact and its gates all passed; the artifact it points at
        no longer hashes to what it recorded."""
        assert sweep.run().returncode == 0
        results = sorted((sweep.root / "results" / "attempts").rglob("results.json"))
        assert results
        document = json.loads(results[-1].read_text())
        document["graphs"][0]["results"][0]["gpu_kernel_stats"]["mean_ms"] = 0.001
        results[-1].write_text(json.dumps(document, indent=2))
        again = sweep.run()
        assert again.returncode == 0
        assert not sweep.resumed(again), "edited evidence must not resume"


class TestTheDriverRefusesAnUnsafeConfig:
    def test_a_shell_launcher_is_not_a_benchmark_executable(self, sweep):
        config = json.loads(sweep.config_path.read_text())
        config["benchmark"]["argv"] = ["/bin/sh"]
        sweep.config_path.write_text(json.dumps(config))
        result = sweep.run()
        assert result.returncode == 2
        assert "shell launchers are not sweep executables" in result.stderr

    def test_a_wrapper_that_selects_a_shell_is_refused_by_its_shebang(self, sweep):
        """Refusing only argv[0]'s FILENAME lets any differently-named wrapper
        through the config gate; it is then caught much later by the interpreter
        identity gate, which declines the sweep (exit 1) instead of rejecting the
        config (exit 2). What makes it unusable is what it SELECTS, not its name."""
        wrapper = _script(sweep.bin / "run-benchmark", "")
        wrapper.write_text('#!/bin/sh\nexec "%s" "$@"\n' % (sweep.bin / "python3"))
        config = json.loads(sweep.config_path.read_text())
        config["benchmark"]["argv"] = [str(wrapper)]
        sweep.config_path.write_text(json.dumps(config))
        result = sweep.run()
        assert result.returncode == 2, result.stdout + result.stderr
        assert "INVALID CONFIG" in result.stderr
        assert "shell launchers are not sweep executables" in result.stderr
        assert "SWEEP_INCOMPLETE" not in result.stderr

    def test_a_config_may_not_redirect_a_driver_owned_option(self, sweep):
        """`--engine` in the config would let the evidence come from a different
        engine than the one the ledger attributes it to."""
        config = json.loads(sweep.config_path.read_text())
        config["benchmark"]["argv"] = [str(sweep.bin / "python3"), "--engine", "1"]
        sweep.config_path.write_text(json.dumps(config))
        result = sweep.run()
        assert result.returncode == 2
        assert "phase-owned options" in result.stderr

    def test_an_unknown_key_is_refused_rather_than_ignored(self, sweep):
        config = json.loads(sweep.config_path.read_text())
        config["profiles"] = ["fast"]
        sweep.config_path.write_text(json.dumps(config))
        result = sweep.run()
        assert result.returncode == 2
        assert "unknown keys" in result.stderr

    def test_a_corpus_that_grew_is_not_the_same_experiment(self, sweep):
        (sweep.corpus / "extra.json").write_text(json.dumps(_graph(99)))
        result = sweep.run()
        assert result.returncode == 2
        assert "expected 3" in result.stderr

    def test_a_missing_device_declines_the_sweep_at_the_device_gate(self, sweep):
        """Wrong-arch host: the sweep may not silently measure whatever is present.
        A gate that declined is an ordinary incomplete outcome, exit 1 -- distinct
        from the operational failure below, which is exit 2."""
        env_result = subprocess.run(
            [sys.executable, str(_SWEEP), "--config", str(sweep.config_path)],
            cwd=sweep.elsewhere,
            env=dict(
                os.environ,
                PATH=str(sweep.bin) + os.pathsep + os.environ.get("PATH", ""),
                FAKE_SCENARIO=str(sweep.scenario_path),
                FAKE_ROCMINFO_ARCH="gfx90a",
            ),
            capture_output=True,
            text=True,
        )
        assert env_result.returncode == 1
        assert "SWEEP_INCOMPLETE" in env_result.stderr
        assert "gfx942" in env_result.stderr
        assert "SWEEP ERROR" not in env_result.stderr

    def test_an_unwritable_output_root_is_an_operational_error_not_an_incomplete_sweep(
        self, sweep
    ):
        """An OSError is a broken execution host, not a measured decline. Reported
        as SWEEP_INCOMPLETE it reads as an ordinary gated-out result, and a harness
        driving several arches carries on as though this one had simply produced
        nothing to compare."""
        sweep.root.chmod(0o555)
        try:
            result = sweep.run()
        finally:
            sweep.root.chmod(0o755)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "SWEEP ERROR" in result.stderr
        assert "SWEEP_INCOMPLETE" not in result.stderr
