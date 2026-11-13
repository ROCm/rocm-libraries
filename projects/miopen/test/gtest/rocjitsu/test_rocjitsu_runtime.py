# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Architecture/config, installed category and non-widening phase boundaries."""

from copy import deepcopy
from fnmatch import fnmatchcase
import json
import os
import subprocess
import signal
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_rocjitsu import (
    CATEGORY,
    SUITE,
    load_policy,
    resolve_arch,
    runtime_paths,
    run_logged,
    split_execution_commands,
    suite_environment,
    suite_metadata,
    validated_config,
)
from rocjitsu_cpu import available_cpu_count, run_cpu_shards
from rocjitsu_selection import export_policy, target_version


def matches(command, name):
    value = next(
        arg.split("=", 1)[1] for arg in command if arg.startswith("--gtest_filter=")
    )
    positive, _, negative = value.partition("-")
    return any(
        fnmatchcase(name, part) for part in (positive or "*").split(":")
    ) and not any(fnmatchcase(name, part) for part in negative.split(":") if part)


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            export_policy(Path(__file__).parent.parent / "test_categories.yaml", policy)
            cls.policy = load_policy(policy)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.targets = deepcopy(self.policy["targets"])

    def test_config_requires_exact_target_version_for_registered_and_override_arches(
        self,
    ):
        for arch, version in (("gfx1250", 120500), ("gfx950", 90500)):
            with self.subTest(arch=arch):
                profile = {
                    "vm": {
                        "gpu": {
                            "device": {
                                "gfx_target_version": version,
                                "num_cu_per_sh": 9,
                            }
                        }
                    },
                    "topology": {
                        "root": {"name": "soc", "children": [{"name": "cu[0:9]"}]}
                    },
                }
                config = self.directory / "config.json"
                for bad_version in (None, "120500", True, 90000):
                    bad = deepcopy(profile)
                    bad["vm"]["gpu"]["device"]["gfx_target_version"] = bad_version
                    config.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError, "does not target"):
                        validated_config(config, arch, 4)
                bad = deepcopy(profile)
                del bad["vm"]["gpu"]["device"]["gfx_target_version"]
                config.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "does not target"):
                    validated_config(config, arch, 4)
                bad = deepcopy(profile)
                bad["vm"]["target"] = "gfx950" if arch == "gfx1250" else "gfx1250"
                config.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "does not target"):
                    validated_config(config, arch, 4)

    def test_version_conversion_and_unsafe_arch_names(self):
        for arch, version in (
            ("gfx950", 90500),
            ("gfx1250", 120500),
            ("gfx90a", 90010),
        ):
            with self.subTest(arch=arch):
                self.assertEqual(target_version(arch), version)
        for arch in ("gfx950:gfx1250", "gfx950/../gfx1250", "gfx", "950", "gfx0950"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                target_version(arch)

    def test_ci_selection_uses_unique_exact_target_before_family(self):
        for value in ("gfx1250", " gfx950,gfx1250;gfx1100 ", "gfx1250;gfx1250"):
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_arch(
                        None,
                        self.targets,
                        {"AMDGPU_TARGETS": value, "AMDGPU_FAMILIES": "unrelated"},
                    ),
                    "gfx1250",
                )
        for value in ("gfx125X-dcgpu", "unrelated; gfx125X-dcgpu,gfx125X-dcgpu"):
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_arch(None, self.targets, {"AMDGPU_FAMILIES": value}),
                    "gfx1250",
                )

    def test_ci_selection_rejects_missing_unknown_and_strict_inputs(self):
        for env in (
            {},
            {"AMDGPU_FAMILIES": "gfx125X-dcgpu-strict"},
            {"AMDGPU_FAMILIES": "gfx1250"},
            {"AMDGPU_TARGETS": "gfx1250-strict"},
            {"AMDGPU_TARGETS": "gfx950", "AMDGPU_FAMILIES": "gfx125X-dcgpu"},
            {"AMDGPU_TARGETS": "gfx1250:gfx950"},
            {"HSA_OVERRIDE_GFX_VERSION": "12.5.0"},
        ):
            with self.subTest(env=env), self.assertRaisesRegex(
                ValueError, "Expected one policy-supported"
            ):
                resolve_arch(None, self.targets, env)

    def test_new_targets_and_family_variants_require_unambiguous_selection(self):
        self.targets["gfx950"] = {"family": "gfx950-dcgpu", "profile": "gfx950.json"}
        for env in (
            {"AMDGPU_TARGETS": "gfx1250,gfx950"},
            {"AMDGPU_FAMILIES": "gfx125X-dcgpu;gfx950-dcgpu"},
        ):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "matched"):
                resolve_arch(None, self.targets, env)
        self.assertEqual(
            resolve_arch("gfx1250", self.targets, {"AMDGPU_TARGETS": "gfx1250;gfx950"}),
            "gfx1250",
        )
        self.targets["gfx1251"] = {
            "family": "gfx125X-dcgpu",
            "profile": "gfx1251.json",
        }
        with self.assertRaisesRegex(ValueError, "Expected one policy-supported"):
            resolve_arch(None, self.targets, {"AMDGPU_FAMILIES": "gfx125X-dcgpu"})
        self.assertEqual(
            resolve_arch(None, self.targets, {"AMDGPU_TARGETS": "gfx1251"}),
            "gfx1251",
        )

    def test_explicit_unregistered_target_requires_paired_real_overrides(self):
        with self.assertRaisesRegex(ValueError, "Unregistered rocjitsu target gfx950"):
            resolve_arch("gfx950", self.targets, {})
        self.assertEqual(
            resolve_arch("gfx950", self.targets, {}, overrides=True), "gfx950"
        )
        for arch in ("gfx1250-strict", "../gfx1250"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                resolve_arch(arch, self.targets, {}, overrides=True)
        launcher = self.directory / "launcher"
        config = self.directory / "local.json"
        for supplied in ((launcher, None), (None, config)):
            with self.subTest(supplied=supplied), self.assertRaisesRegex(
                ValueError, "must be supplied together"
            ):
                runtime_paths(self.directory, "gfx950", self.targets, *supplied)
        with self.assertRaisesRegex(RuntimeError, "launcher missing or not executable"):
            runtime_paths(self.directory, "gfx950", self.targets, launcher, config)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "config missing"):
            runtime_paths(self.directory, "gfx950", self.targets, launcher, config)
        config.write_text('{"vm":{"gpu":{"device":{"gfx_target_version":90500}}}}')
        _, found_config = runtime_paths(
            self.directory, "gfx950", self.targets, launcher, config
        )
        with self.assertRaisesRegex(ValueError, "does not target gfx1250"):
            validated_config(found_config, "gfx1250", 4)

    def test_installed_package_requires_executable_launcher_and_policy_profile(self):
        launcher = self.directory / "bin/rocjitsu"
        launcher.parent.mkdir()
        profiles = self.directory / "share/rocjitsu/configs"
        profiles.mkdir(parents=True)
        profile = profiles / self.targets["gfx1250"]["profile"]
        # A previous prepared tree must not satisfy installed artifact discovery.
        old_launcher = self.directory / "build/tools/rocjitsu/rocjitsu"
        old_launcher.parent.mkdir(parents=True)
        old_launcher.write_text("#!/bin/sh\nexit 0\n")
        old_launcher.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "launcher missing or not executable"):
            runtime_paths(self.directory, "gfx1250", self.targets)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(RuntimeError, "launcher missing or not executable"):
            runtime_paths(self.directory, "gfx1250", self.targets)
        launcher.chmod(0o755)
        (profiles / "other.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "config missing"):
            runtime_paths(self.directory, "gfx1250", self.targets)
        profile.write_text('{"vm":{"gpu":{"device":{"gfx_target_version":90500}}}}')
        _, config = runtime_paths(self.directory, "gfx1250", self.targets)
        with self.assertRaisesRegex(ValueError, "does not target gfx1250"):
            validated_config(config, "gfx1250", 4)
        profile.write_text(
            '{"vm":{"gpu":{"num_gpus":4,"device":{"gfx_target_version":120500}}}}'
        )
        normalized = validated_config(config, "gfx1250", 4)
        self.assertEqual(normalized["vm"]["gpu"]["num_gpus"], 1)

    def test_malformed_configuration_fails_as_target_error(self):
        config = self.directory / "config.json"
        for data in (
            [],
            None,
            {},
            {"vm": None},
            {"vm": {"gpu": []}},
            {"vm": {"gpu": {"device": None}}},
        ):
            with self.subTest(data=data):
                config.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "does not target gfx1250"):
                    validated_config(config, "gfx1250", 4)

    def test_failed_policy_loading_invalidates_only_the_requested_verdict(self):
        runner = self.directory / "installed-runner"
        runner.mkdir()
        for filename in (
            "run_rocjitsu.py",
            "rocjitsu_cpu.py",
            "rocjitsu_reports.py",
            "rocjitsu_selection.py",
        ):
            shutil.copy2(Path(__file__).with_name(filename), runner / filename)
        policy = runner / "rocjitsu_policy.json"
        report = (
            self.directory
            / "rocjitsu-work/gfx1250/reports/miopen_gtest_rocjitsu-quick.xml"
        )
        report.parent.mkdir(parents=True)
        unrelated = (
            self.directory
            / "rocjitsu-work/gfx950/reports/miopen_gtest_rocjitsu-quick.xml"
        )
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('<testsuites tests="1" passed="1"/>')
        for arguments in (["--arch", "gfx1250"], []):
            for content in (None, "{", '{"category":"quick"}'):
                with self.subTest(arguments=arguments, policy=content):
                    policy.unlink(missing_ok=True)
                    if content is not None:
                        policy.write_text(content)
                    report.write_text('<testsuites tests="1" passed="1"/>')
                    result = subprocess.run(
                        [sys.executable, str(runner / "run_rocjitsu.py"), *arguments],
                        cwd=self.directory,
                        env={**os.environ, "AMDGPU_TARGETS": "gfx1250"},
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertFalse(report.exists())
                    self.assertTrue(unrelated.exists())

    def test_wrong_or_missing_exported_category_cannot_fall_back(self):
        path = self.directory / "policy.json"
        for policy in (
            [],
            None,
            {"category": "quick", "representative_cases": {}},
            {"representative_cases": {}},
            {"category": CATEGORY},
            {"category": CATEGORY, "representative_cases": []},
        ):
            with self.subTest(policy=policy):
                path.write_text(json.dumps(policy))
                with self.assertRaisesRegex(ValueError, "Expected exported"):
                    load_policy(path)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_policy(path)
        policy = deepcopy(self.policy)
        for targets in (None, {}, [], {"gfx1250": {"family": "gfx125X-dcgpu"}}):
            with self.subTest(targets=targets):
                policy["targets"] = targets
                path.write_text(json.dumps(policy))
                with self.assertRaises(ValueError):
                    load_policy(path)

    def test_cpu_gpu_partition_does_not_reintroduce_unselected_cpu_cases(self):
        inventory = [
            "CPU_Config_NONE.retained",
            "CPU_Config_NONE.costly",
            "Smoke/CPU_Host_NONE.retained/0",
            "Smoke/CPU_Host_NONE.costly/1",
            "Smoke/GPU_Op_FP32.retained/0",
            "Smoke/GPU_Op_FP32.costly/1",
            "Full/GPU_Other_FP32.case",
        ]
        command = [
            "miopen_gtest",
            "--gtest_filter=CPU_Config_NONE.retained:Smoke/CPU_Host_NONE.retained/*:Smoke/GPU_Op_FP32.*"
            "-Smoke/GPU_Op_FP32.costly/*",
        ]
        selected = [inventory[0], inventory[2], inventory[4]]
        cpu, gpu, cpu_names, gpu_names = split_execution_commands(command, selected)
        actual_cpu = {name for name in inventory if matches(cpu, name)}
        actual_gpu = {name for name in inventory if matches(gpu, name)}
        self.assertEqual(actual_cpu, {inventory[0], inventory[2]})
        self.assertEqual(actual_gpu, {inventory[4]})
        self.assertEqual(actual_cpu & actual_gpu, set())
        self.assertEqual(actual_cpu | actual_gpu, set(selected))
        self.assertEqual(set(cpu_names), actual_cpu)
        self.assertEqual(set(gpu_names), actual_gpu)

    def test_empty_cpu_phase_never_means_all_tests(self):
        command = ["miopen_gtest", "--gtest_filter=Smoke/GPU_Op_FP32.*"]
        cpu, gpu, cpu_names, gpu_names = split_execution_commands(
            command, ["Smoke/GPU_Op_FP32.case"]
        )
        self.assertFalse(matches(cpu, "CPU_A_NONE.case"))
        self.assertFalse(matches(cpu, "Smoke/GPU_Op_FP32.case"))
        self.assertTrue(matches(gpu, "Smoke/GPU_Op_FP32.case"))
        self.assertEqual(cpu_names, [])
        self.assertEqual(gpu_names, ["Smoke/GPU_Op_FP32.case"])

    def test_ctest_environment_modifications_and_architecture_overrides(self):
        with patch.dict(
            os.environ,
            {
                "ROCJITSU_TEST_LIST": "host",
                "HSA_OVERRIDE_GFX_VERSION": "12.5.0",
                "HSA_DISABLE_GFX12_STRICT": "1",
            },
        ):
            env = suite_environment(
                {
                    "ENVIRONMENT": [
                        "ROCJITSU_TEST_LIST=packaged",
                        "ROCJITSU_TEMP=remove",
                    ],
                    "ENVIRONMENT_MODIFICATION": [
                        "ROCJITSU_TEST_LIST=path_list_prepend:first",
                        "ROCJITSU_TEST_LIST=reset:",
                        "ROCJITSU_TEST_LIST=path_list_append:last",
                        "ROCJITSU_TEMP=unset:",
                        "HSA_OVERRIDE_GFX_VERSION=set:9.5.0",
                        "HSA_DISABLE_GFX12_STRICT=set:1",
                    ],
                }
            )
        self.assertEqual(env["ROCJITSU_TEST_LIST"], f"packaged{os.pathsep}last")
        self.assertNotIn("ROCJITSU_TEMP", env)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", env)
        self.assertNotIn("HSA_DISABLE_GFX12_STRICT", env)
        with self.assertRaisesRegex(RuntimeError, "Unsupported CTest"):
            suite_environment({"ENVIRONMENT_MODIFICATION": ["X=unknown:value"]})

    def metadata(self, name=SUITE, label=CATEGORY, properties=""):
        (self.directory / "CTestTestfile.cmake").write_text(
            f'add_test({name} "{sys.executable}" "--gtest_filter=Smoke/*:CPU_*")\n'
            f'set_tests_properties({name} PROPERTIES LABELS "{label}" {properties})\n'
        )
        return suite_metadata(self.directory, self.directory)

    def test_ordinary_quick_metadata_cannot_fall_back(self):
        for name, label in (("miopen_gtest_quick_suite", "quick"), (SUITE, "quick")):
            with self.subTest(name=name, label=label), self.assertRaisesRegex(
                RuntimeError, "Expected exactly one"
            ):
                self.metadata(name, label)

    def test_disabled_or_missing_working_directory_cannot_run(self):
        with self.assertRaisesRegex(RuntimeError, "is disabled"):
            self.metadata(properties="DISABLED TRUE")
        with self.assertRaisesRegex(RuntimeError, "working directory missing"):
            self.metadata(properties=f'WORKING_DIRECTORY "{self.directory / "absent"}"')

    def test_real_child_failure_is_propagated_with_saved_diagnostics(self):
        log = self.directory / "failed.log"
        with self.assertRaises(subprocess.CalledProcessError) as context:
            run_logged(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('numerical failure', flush=True); sys.exit(7)",
                ],
                os.environ.copy(),
                self.directory,
                log,
                10,
            )
        self.assertEqual(context.exception.returncode, 7)
        self.assertIn("numerical failure", log.read_text())

    def test_real_child_timeout_is_an_error_not_a_verdict(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            run_logged(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                os.environ.copy(),
                self.directory,
                self.directory / "timeout.log",
                0.1,
            )

    def test_interruption_terminates_and_reaps_the_real_child(self):
        ready = self.directory / "ready.pipe"
        os.mkfifo(ready)
        interrupted = []
        original_wait = subprocess.Popen.wait

        def interrupt_once(process, *args, **kwargs):
            if not interrupted:
                interrupted.append(process)
                with ready.open("rb") as pipe:
                    self.assertEqual(pipe.read(1), b"1")
                raise KeyboardInterrupt
            return original_wait(process, *args, **kwargs)

        # Closing stdout before the handshake keeps the unfixed path from
        # blocking on its reader; the child still needs termination and reaping.
        program = (
            "import os, sys, time\n"
            "os.close(1)\nos.close(2)\n"
            "with open(sys.argv[1], 'wb') as ready:\n"
            "    ready.write(b'1')\n"
            "time.sleep(60)\n"
        )
        try:
            with patch.object(subprocess.Popen, "wait", interrupt_once):
                with self.assertRaises(KeyboardInterrupt):
                    run_logged(
                        [sys.executable, "-c", program, str(ready)],
                        os.environ.copy(),
                        self.directory,
                        self.directory / "interrupted.log",
                        10,
                    )
            with self.assertRaises(ChildProcessError):
                os.waitpid(interrupted[0].pid, os.WNOHANG)
        finally:
            for process in interrupted:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                original_wait(process)

    def test_cpu_shard_launch_failures_keep_every_expected_missing_report(self):
        config = self.directory / "config.json"
        config.write_text(json.dumps({"vm": {"gpu": {"num_gpus": 1}}}))
        workers = min(2, available_cpu_count())
        # Python is deliberately an invalid launcher: its real nonzero exit must
        # reach aggregation, rather than an earlier successful XML being reused.
        for index in range(workers):
            (self.directory / f"cpu-shard-{index}.xml").write_text("old verdict")
        paths, errors = run_cpu_shards(
            [sys.executable],
            Path(sys.executable),
            config,
            os.environ.copy(),
            self.directory,
            self.directory,
            workers,
            10,
            1,
            run_logged,
        )
        self.assertEqual(
            set(paths),
            {self.directory / f"cpu-shard-{index}.xml" for index in range(workers)},
        )
        self.assertEqual(len(errors), workers)
        for index, error in enumerate(errors):
            self.assertIn(f"CPU shard {index}", error)
            self.assertIn("non-zero exit status", error)
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertTrue(
            all(
                (self.directory / f"cpu-shard-{i}.log").is_file()
                for i in range(workers)
            )
        )

    def test_cpu_masks_cannot_oversubscribe_affinity_or_dispatch_limit(self):
        with self.assertRaisesRegex(ValueError, "allocation"):
            run_cpu_shards(
                [],
                Path(sys.executable),
                self.directory / "absent.json",
                {},
                self.directory,
                self.directory,
                len(os.sched_getaffinity(0)) + 1,
                10,
                1,
                run_logged,
            )
        with self.assertRaisesRegex(ValueError, "dispatch"):
            run_cpu_shards(
                [],
                Path(sys.executable),
                self.directory / "absent.json",
                {},
                self.directory,
                self.directory,
                1,
                10,
                5,
                run_logged,
            )

    def test_cpu_count_respects_cgroup_quota_affinity_and_fractional_limits(self):
        for affinity, files, expected in (
            (8, {"/sys/fs/cgroup/cpu.max": "250000 100000"}, 2),
            (2, {"/sys/fs/cgroup/cpu.max": "800000 100000"}, 2),
            (8, {"/sys/fs/cgroup/cpu.max": "50000 100000"}, 1),
            (3, {"/sys/fs/cgroup/cpu.max": "max 100000"}, 3),
            (
                8,
                {
                    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "300000",
                    "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
                },
                3,
            ),
        ):

            def read_cgroup(path, *args, **kwargs):
                if str(path) not in files:
                    raise FileNotFoundError(path)
                return files[str(path)]

            with self.subTest(files=files), patch(
                "rocjitsu_cpu.os.sched_getaffinity", return_value=set(range(affinity))
            ), patch.object(Path, "read_text", read_cgroup):
                self.assertEqual(available_cpu_count(), expected)


if __name__ == "__main__":
    unittest.main()
