# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Path resolution and the configuration's checks (rkmon_lib/paths.py, rkmon_lib/config.py): a blank configuration,
the package's defaults, and the ROCm LLVM tools' folder a configuration written before the package gets.
"""

import ast
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import home_repository  # noqa: E402
from rkmon_lib import config, doctor, paths, review, verifier_file  # noqa: E402
from rkmon_lib.checks import Steps  # noqa: E402

D = "/home/op/rocke_review"
# A configuration written before the package: `paths.root`, no `paths.working_dir`.
LEGACY = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "paths": {"root": D, "dashboard": "/srv/out/rocke_merge_queue.md"},
}


def write(d: str, cfg: dict) -> pathlib.Path:
    p = pathlib.Path(d) / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class TestWorkerModule(unittest.TestCase):
    def test_a_worker_module_is_imported_with_worker_on_no_path(self):
        """paths.worker_module imports worker/<name>.py when asked, with worker/ on no path before: docker/rkmon's
        deploy path imports modules that read worker/ only this way."""
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import paths; "
            "print(paths.worker_module('perms').__file__)"
        )
        p = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), str(ROOT / "worker" / "perms.py"))


class TestLegacyShape(unittest.TestCase):
    def test_an_older_configuration_gets_the_old_machines_llvm_tools(self):
        """A configuration written before the package that names no ROCm LLVM tools gets the folder the code used
        when a configuration named none. A current configuration that names none has none.
        """
        self.assertTrue(paths.is_legacy(LEGACY))
        self.assertEqual(
            paths.resolve(LEGACY)["rocm_llvm_bin"], "/opt/rocm/core-10.1/lib/llvm/bin"
        )
        current = dict(LEGACY, paths={"working_dir": D})
        self.assertFalse(paths.is_legacy(current))
        self.assertEqual(paths.resolve(current)["rocm_llvm_bin"], "")


class TestSparseWorktrees(unittest.TestCase):
    def test_a_worktree_checks_out_rockes_parent_and_github(self):
        cfg = {
            "repo": "acme/widgets",
            "rocke_path": "a/b/rocke",
            "paths": {"working_dir": D},
        }
        self.assertEqual(paths.sparse_dirs(cfg), ["a/b", ".github"])
        script = review.checks_script(
            cfg,
            1,
            "a" * 40,
            "first-round",
            {
                "families": [],
                "tests": [],
                "build": False,
                "golden": False,
                "engine_wide": False,
                "outside_rocke": [],
                "libparity": False,
                "_dirs": {},
            },
            None,
        )
        self.assertIn("sparse-checkout set --no-cone a/b .github", script)
        self.assertIn(f"OUT={D}/workers/pr1/checks;", script)


class TestTheRepositorysFolder(unittest.TestCase):
    def test_one_function_reads_it_from_the_configuration(self):
        """paths.repository_folder is the one home of the folder's reading (config rocke_path, the deploy settings'
        repository.folder): no other code reads the key, so each reader gets it with no slash at either end.
        """
        from rkmon_lib import package_files

        reads = []
        for path in package_files.tracked(ROOT):
            rel = (path.relative_to(ROOT) if path.is_absolute() else path).as_posix()
            if not rel.endswith(".py") or rel.startswith("tests/"):
                continue
            if rel == "rkmon_lib/paths.py":
                continue  # the home
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                key = None
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                    key = node.slice
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args
                ):
                    key = node.args[0]
                if isinstance(key, ast.Constant) and key.value == "rocke_path":
                    reads.append(f"{rel}:{node.lineno}")
        self.assertEqual(reads, [])
        self.assertEqual(paths.repository_folder({"rocke_path": "/lib/k/"}), "lib/k")


class TestBlankConfiguration(unittest.TestCase):
    def test_nothing_resolves_and_writers_refuse(self):
        self.assertEqual(
            {k for k, v in paths.resolve({}).items() if v}, {"package"}
        )  # only the code's own folder
        self.assertEqual(paths.resolve({})["package"], str(ROOT))
        with self.assertRaises(paths.PathsError):
            paths.get({}, "dashboard")
        with self.assertRaises(paths.PathsError):
            paths.shell_vars({})

    def test_the_package_defaults_name_every_blank_required_key(self):
        defaults = config.load_defaults()
        bad = config.problems(defaults)
        from rkmon_lib import deploy_tree

        for key in (
            "repo",
            "rocke_path",
            "operator.github_login",
            "paths.working_dir",
            "scope.labels",
        ):
            key = deploy_tree.name(key)  # as the deploy settings name it
            self.assertTrue(
                any(line.startswith(key + " is blank") for line in bad), (key, bad)
            )
        self.assertFalse(
            any(line.startswith("model_bindings.") for line in bad)
        )  # the default first pass needs no model
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(config.ConfigError) as cm:
                config.load(write(d, defaults))
        self.assertIn(
            "integrations.github_cli.username is blank: the GitHub login the monitor posts as",
            str(cm.exception),
        )
        # its fix, which the error carries: the settings fixed in the deploy settings file, then confirmed and applied
        fix = str(cm.exception.fix("rkmon"))
        self.assertEqual(fix, str(doctor.cannot_run_fix()))
        self.assertTrue(fix.endswith(f" {Steps(doctor.apply_commands())}"), fix)

    def test_safe_defaults(self):
        from rkmon_lib import deploy_tree, features

        d = config.load_defaults()
        self.assertEqual(d["features"], deploy_tree.internal_defaults()["features"])
        self.assertEqual(
            (
                d["features"]["review_automation"]["mode"],
                d["features"]["merge_queue"]["mode"],
                d["features"]["merge_queue"]["auto_merge"],
            ),
            ("prototype", "prototype", "hints_only"),
        )
        c = config.normalize(d)
        # nothing outward: no feature that sends is live
        self.assertEqual([f for f in features.SENDS if features.live(c, f)], [])
        self.assertEqual(
            (features.guard_mode(c), features.active(c, "queue-notices")),
            ("alert", False),
        )
        self.assertEqual(
            (
                d["teams"]["channel"],
                d["publish"]["html"]["host"],
                d["jira"]["project"],
                d["jira"]["accounts"],
            ),
            ("", "", "", {}),
        )
        for key in ("repo", "rocke_path", "claude_workspace"):
            self.assertEqual(d[key], "", key)
        self.assertEqual(d["operator"]["github_login"], "")
        # Every folder is the machine's own (blank).
        self.assertEqual(set(d["paths"].values()), {""})

    def test_every_key_has_one_line_of_documentation(self):
        """Every key of the defaults has its one line in the registry (rkmon_lib/registry.py): an advanced key its
        own; a key that a key of the deploy settings file feeds, its file key's, from the schema
        (tests/test_schema.py).
        """
        from rkmon_lib import deploy_tree, registry

        d = config.load_defaults()
        doc = config.doc()
        fed = deploy_tree.held()
        self.assertTrue(
            all(
                doc[k]
                for k, key in fed.items()
                if not key.startswith("model_bindings.")
            )
        )  # each fed key's description is its file key's, from the schema
        names = {k.name for k in registry.keys()}

        def keys(node, prefix=""):
            """Each value of the defaults, by the key that holds it (a value under no key, by its own path)."""
            for k, v in node.items():
                if prefix + k not in names and isinstance(v, dict) and v:
                    yield from keys(v, prefix + k + ".")
                else:
                    yield prefix + k

        self.assertEqual(
            sorted(set(keys(d)) - set(doc)), [], "keys without a description"
        )
        self.assertEqual(
            sorted(n for n in doc if registry.kind(n) is None),
            [],
            "descriptions of keys that do not exist",
        )  # a key's, a section's or a name's inside a key (registry.PARTS)
        self.assertTrue(all(v and "\n" not in v for v in doc.values()))


class TestPackageDefaults(unittest.TestCase):
    def cfg(self, **over):
        c = config.load_defaults()
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"], "path_labels": []},
        )
        c["paths"] = dict(c["paths"], working_dir="/w")
        c.update(over)
        return c

    def test_everything_lives_under_the_working_folder(self):
        p = paths.resolve(self.cfg())
        self.assertFalse(paths.is_legacy(self.cfg()))
        self.assertEqual(
            (
                p["toolkit"],
                p["repo"],
                p["worker_tools"],
                p["outputs_dir"],
                p["html"],
                p["weekly_dir"],
                p["review_book"],
                p["rocm_llvm_bin"],
                p["secrets_dir"],
                p["package"],
            ),
            (
                "/w/monitor",
                "/w/repo",
                "/w/tools",
                "/w",
                "/w/rocke_monitor.html",
                "/w/rocke_weekly",
                "/w/agentic/review_book.md",
                "",
                "/w/.config",
                str(ROOT),
            ),
        )
        self.assertEqual(config.problems(self.cfg()), [])

    def test_an_outputs_folder_moves_only_the_operators_files(self):
        c = self.cfg()
        c["paths"]["outputs_dir"] = "~/rkmon-outputs"
        p = paths.resolve(c)
        home = os.path.expanduser("~")
        self.assertEqual(
            (p["dashboard"], p["daily_dir"], p["posted_dir"], p["workers"]),
            (
                f"{home}/rkmon-outputs/rocke_merge_queue.md",
                f"{home}/rkmon-outputs/rocke_daily",
                f"{home}/rkmon-outputs",
                "/w/workers",
            ),
        )

    def test_reviews_need_the_model_not_the_llvm_tools(self):
        """A blank paths.rocm_llvm_bin refuses no rk command while reviews run, on a machine with no ROCm and a remote
        first pass. It leaves the develop guard incomplete and the compile gate pending instead.
        """
        from rkmon_lib import features

        self.assertEqual(
            config.problems(self.cfg()), []
        )  # the default first pass needs no model
        bad = config.problems(self.cfg(reviewer="worker"))
        self.assertEqual(
            [line.split(" is blank")[0] for line in bad],
            ["model_bindings.worker.model_id", "model_bindings.worker"],
        )
        self.assertIn(
            "ROCm's LLVM tools",
            " ".join(features.missing(self.cfg(), "develop-checks")),
        )

    def test_bad_values_are_named(self):
        bad = " | ".join(
            config.problems(
                self.cfg(
                    features={"review_automation": {"mode": "go"}},
                    repo="widgets",
                )
            )
        )
        for word in (
            "features.review_automation.mode is 'go'",
            "repository.name is 'widgets'",
        ):
            self.assertIn(word, bad)
        c = self.cfg()
        c["paths"]["workers"] = "relative/dir"
        self.assertIn(
            "paths.workers is 'relative/dir'; it must be an absolute path",
            " ".join(config.problems(c)),
        )

    def test_derived_values(self):
        c = config.normalize(
            self.cfg(
                worker=dict(
                    config.load_defaults()["worker"],
                    model="m:1",
                    url="http://127.0.0.1:9",
                ),
                hot_spots=home_repository.VALUES["hot_spots"],
            )
        )
        self.assertEqual(
            c["review_first_line"],
            f"Automated review by rocke-monitor: Model - m:1, effort: {config.MODEL_DEFAULT}",
        )
        self.assertEqual(c["codeowner_org"], "acme")
        self.assertEqual(c["weekly_report"]["dir"], "/w/rocke_weekly")
        from rkmon_lib import queue

        self.assertEqual(
            queue.hot_paths(c, "ABI"), ["a/b/rocke/platform/cpp/include/rocke/abi.h"]
        )

    def test_the_verifier_s_team_file_is_the_agentic_install_s_copy(self):
        """paths.verifier_team_file, blank: the verifier's team member file (VERIFIER_FILE, verifier_file.FILE) in the
        Claude workspace's .claude/, where the agentic install puts it; none without a workspace.
        """
        self.assertEqual(verifier_file.FILE, paths.VERIFIER_FILE)
        with mock.patch.object(paths, "VERIFIER_FILE", "team_members/v.md"):
            self.assertEqual(
                paths.resolve(self.cfg(claude_workspace="/ws"))["verifier_team_file"],
                "/ws/.claude/team_members/v.md",
            )
        self.assertEqual(
            paths.resolve(self.cfg(claude_workspace=""))["verifier_team_file"], ""
        )

    def test_pr_code_learns_where_the_reviewer_model_runs_never_its_address(self):
        """tools/worker_grace.py runs as PR code, whose environment (worker/pr_env.py) holds no address: the monitor's
        jobs say where the reviewer model runs (RKMON_WORKER_REMOTE), and the tool reads that; run by hand without it,
        the tool reads the address."""
        import importlib.util

        sys.path.insert(0, str(ROOT / "worker"))
        import pr_env

        spec = importlib.util.spec_from_file_location(
            "worker_grace", ROOT / "tools" / "worker_grace.py"
        )
        grace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grace)
        far = "https://models.example.com/v1"
        for url, remote in ((far, True), ("http://127.0.0.1:9", False)):
            env = config.env(
                config.normalize(self.cfg(worker={"model": "m:1", "url": url}))
            )
            self.assertEqual(env["RKMON_WORKER_REMOTE"], "1" if remote else "0")
            pr = pr_env.allowed(env)
            self.assertNotIn("RKMON_WORKER_URL", pr)
            self.assertEqual(grace.remote(pr), remote, url)
        self.assertTrue(grace.remote({"RKMON_WORKER_URL": far}))
        self.assertFalse(
            grace.remote({"RKMON_WORKER_URL": far, "RKMON_WORKER_REMOTE": "0"})
        )

    def test_the_worker_environment(self):
        env = config.env(
            config.normalize(
                self.cfg(
                    worker={"model": "m:1", "url": "http://127.0.0.1:9"},
                    tracker_keys=["abc"],
                    jira={"project": "XYZ", "component": "", "labels": []},
                )
            )
        )
        self.assertEqual(
            (
                env["RKMON_WORKING_DIR"],
                env["RKMON_REPO"],
                env["RKMON_ROCKE_PATH"],
                env["RKMON_SPARSE"],
                env["RKMON_OPERATOR_LOGIN"],
                env["RKMON_WORKER_MODEL"],
                env["RKMON_TRACKER_KEYS"],
                env["RKMON_REVIEW_BOOK"],
            ),
            (
                "/w",
                "acme/widgets",
                "a/b/rocke",
                "a/b .github",
                "op-login",
                "m:1",
                "ABC,XYZ",
                "/w/agentic/review_book.md",
            ),
        )


class TestCommandLine(unittest.TestCase):
    """rk refuses a command when a required key is blank, naming it; doctor, env and paths still run."""

    def run_rk(self, cfg_path, *args):
        import rkmon

        out = io.StringIO()
        with mock.patch.dict(
            os.environ, {"RKMON_HOME": str(pathlib.Path(cfg_path).parent)}
        ), contextlib.redirect_stdout(out):
            rc = rkmon.main(list(args))
        return rc, out.getvalue()

    def test_refused_with_the_key_named(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, config.load_defaults())
            rc, text = self.run_rk(p, "status")
            self.assertEqual(rc, 4)
            self.assertIn("REFUSED:", text)
            self.assertIn("paths.working_dir is blank", text)
            rc, text = self.run_rk(p, "paths")
            self.assertEqual(rc, 0)
            self.assertIn("working_dir    (not set)", text)
            rc, text = self.run_rk(p, "env")
            self.assertEqual(rc, 0)
            self.assertIn("RKMON_WORKER_JSON=", text)
            self.assertIn('"backend": "ollama"', text)

    def test_no_configuration_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            rc, text = self.run_rk(pathlib.Path(d) / "config.json", "board")
        self.assertEqual(rc, 4)
        self.assertIn("REFUSED: no configuration at ", text)
        self.assertIn(f"\nfix: {Steps(doctor.apply_commands())}\n", text)


class TestConfiguredPaths(unittest.TestCase):
    """store.configured_paths: the folders the newer modules ask for by name, resolved by paths.resolve over the
    configuration they are given (or the monitor home's), so they agree with every other path.
    """

    def test_the_outputs_folder_is_the_dashboards_folder(self):
        from rkmon_lib import store

        p = store.configured_paths(
            {
                "paths": {
                    "working_dir": "/w/rr",
                    "dashboard": "/out/q/rocke_merge_queue.md",
                }
            }
        )
        self.assertEqual(
            (p["root"], p["tmp"], p["scratch"]), ("/w/rr", "/w/rr/tmp", "/w/rr/scratch")
        )
        self.assertEqual(
            (
                p["outputs_dir"],
                p["daily_dir"],
                p["html"],
                p["insights_dir"],
                p["posted_dir"],
                p["spend_dir"],
            ),
            (
                "/out/q",
                "/out/q/rocke_daily",
                "/out/q/rocke_monitor.html",
                "/out/q",
                "/out/q",
                "/out/q/rocke_spend",
            ),
        )
        self.assertEqual(p["posts"], "/w/rr/scratch/post")

    def test_a_set_key_wins_and_a_blank_one_takes_its_default(self):
        from rkmon_lib import store

        p = store.configured_paths(
            {
                "paths": {
                    "working_dir": "/w/rr/",
                    "outputs_dir": "/o",
                    "daily_dir": "/d/",
                    "html": " ",
                    "posts": "/p",
                    "tmp": "",
                }
            }
        )
        self.assertEqual(
            (
                p["root"],
                p["outputs_dir"],
                p["daily_dir"],
                p["html"],
                p["posts"],
                p["tmp"],
            ),
            ("/w/rr", "/o", "/d", "/o/rocke_monitor.html", "/p", "/w/rr/tmp"),
        )

    def test_no_working_folder_resolves_nothing(self):
        from rkmon_lib import store

        p = store.configured_paths({})
        self.assertEqual((p["root"], p["outputs_dir"], p["posted_dir"]), ("", "", ""))

    def test_no_code_names_a_machine_folder(self):
        """Module defaults come from the configuration, never from a home or Windows folder written in the code."""
        from rkmon_lib import daylog, guard, insights, post, premises, review, webdash

        for mod in (daylog, guard, insights, post, premises, review, webdash):
            src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
            self.assertNotIn("/mnt/c/Users/", src, mod.__name__)
            self.assertNotRegex(src, r"\"/home/[a-z]", mod.__name__)


class TestWorkerTool(unittest.TestCase):
    def test_the_installed_tool_else_the_packages_own(self):
        """A job runs a worker tool from paths.worker_tools ($T) when it is installed there, and the package's own
        worker/ copy while that install lacks the file (one made before the tool existed), so no job fails on it.
        """
        from rkmon_lib import paths

        with tempfile.TemporaryDirectory() as d:
            t, m = pathlib.Path(d) / "tools", pathlib.Path(d) / "pkg"
            t.mkdir()
            (m / "worker").mkdir(parents=True)

            def where(name):
                return subprocess.run(
                    ["bash", "-c", f"T={t}; M={m}; echo {paths.worker_tool(name)}"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()

            self.assertEqual(where("mkwt.py"), f"{m}/worker/mkwt.py")
            (t / "mkwt.py").write_text("")
            self.assertEqual(where("mkwt.py"), f"{t}/mkwt.py")
            # Inside a double-quoted argument (installed_tests.py's --dev-build) it resolves the same way.
            out = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"T={t}; M={m}; printf '%s' \"x {paths.worker_tool('build_native.py')}\"",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(out, f"x {m}/worker/build_native.py")


class TestIsPackage(unittest.TestCase):
    """paths.is_package: a folder with the monitor's rkmon.py and rkmon_lib/ is a package; the running code's own
    folder is one."""

    def test_rkmon_py_and_rkmon_lib_make_a_package(self):
        self.assertTrue(paths.is_package(paths.PACKAGE))
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            self.assertFalse(paths.is_package(d))
            (d / "rkmon.py").write_text("", encoding="utf-8")
            self.assertFalse(paths.is_package(d))  # no rkmon_lib/
            (d / "rkmon_lib").mkdir()
            self.assertTrue(paths.is_package(d))
            self.assertTrue(paths.is_package(str(d)))
            (d / "rkmon.py").unlink()
            (d / "rkmon.py").mkdir()  # a folder by that name is no rkmon.py
            self.assertFalse(paths.is_package(d))
            self.assertFalse(paths.is_package(d / "missing"))


if __name__ == "__main__":
    unittest.main()
