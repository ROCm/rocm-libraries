# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Remote GPU tests (rkmon_lib/gpu.py), against fakes only: an ssh that runs the
login host's side here and answers only an example host, a fake Slurm (tests/fakes/gpu/), and a local repository that
stands in for GitHub. No real host is ever contacted.

What the operator's rules ask of it, each tested here: only the GPU-only cases go (those the checks skipped for want
of a GPU); jobs queue and never hold a review (the post carries a pending row); the row is updated in place when the
job ends, through a guarded, ledgered write that only journals while review automation is in prototype; a failure is
the PR's only after
develop passed it on the same node, and then gets a drafted notice; accounts are tried in order; a node that fails the
GPU probe is avoided and reported; leftovers are cancelled; a long wait is reported once and never resubmitted; a
feature that is on but misconfigured skips cleanly."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    deploy_tree,
    features,
    gh,
    gpu,
    post,
    report,
    settings_file,
)
import home_repository  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

FAKES = HERE / "fakes" / "gpu"
HOST = "login.example.com"
# The login host's own key, as the state records it (an invented key).
HOST_KEY = f"{HOST} ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIInventedForTheTestsOnly"
K = "a/b/rocke"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_AUTHOR_DATE": "2002-02-02T00:00:00Z",
    "GIT_COMMITTER_DATE": "2002-02-02T00:00:00Z",
}
CASES = [
    "platform/tests/test_gpu_x.py::test_value",
    "platform/tests/test_gpu_x.py::test_ok",
]
# The checks' skip lines (tools/pytest_plugins/rkmon_skips.py: "rkmon-skip:<TAB>tool<TAB>test id<TAB>reason").
SKIPS = (
    f"rkmon-skip:\t-\t{CASES[0]}\tneeds a gfx942 GPU\nrkmon-skip:\t-\t{CASES[1]}\tneeds a gfx942 GPU\n"
    "rkmon-skip:\t-\tplatform/tests/test_slow.py::test_big\tslow; run with --slow\n"
)
TEST_FILE = """import os

import pytest

import kern

gpu = pytest.mark.skipif(os.environ.get("FAKE_GPU") != "gfx942", reason="needs a gfx942 GPU")


@gpu
def test_value():
    assert kern.VALUE == 1


@gpu
def test_ok():
    assert True


def test_cpu():
    assert True
"""
SKIPPING = """import pytest


@pytest.mark.skip(reason="needs a gfx950 GPU")
def test_x():
    pass


def test_y():
    pass


@pytest.mark.xfail(reason="known")
def test_z():
    assert 0
"""
REVIEW = (
    "Automated review: Model - m, effort: max\n\n## PR #7 — t — 🟢 First-round: fine\n\n| Prior item | Status |\n"
    "|---|---|\n| Bots / CI | 🟢 Green |\n| Our checks | 🟢 Clean<br>• merge vs develop: clean (ran) |\n\n"
    "**Utility:** 🟢 Positive\n"
)
CLUSTER = {
    "partitions": ["gpu-q"],
    "accounts": ["acct-b"],
    "nodes": [
        {
            "name": "node-a01",
            "gres": "gpu:type-a:8",
            "state": "idle",
            "features": "SITE-A",
            "partition": "gpu-q",
        },
        {
            "name": "node-a02",
            "gres": "gpu:type-a:8(S:0-1)",
            "state": "mixed",
            "features": "SITE-A",
            "partition": "gpu-q",
        },
        {
            "name": "node-b01",
            "gres": "gpu:type-b:4",
            "state": "alloc",
            "features": "SITE-B",
            "partition": "gpu-q",
        },
    ],
}

try:
    import pytest  # noqa: F401

    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, **GIT_ENV),
    ).stdout.strip()


def make_origin(base: pathlib.Path) -> tuple[str, str, str]:
    """The repository GitHub would serve: develop, and PR 7 at refs/pull/7/head, whose change makes test_value fail.
    Returns (its file:// URL, develop's SHA, the PR's head)."""
    o = base / "origin"
    for d in (f"{K}/platform/python", f"{K}/platform/tests", f"{K}/library", ".github"):
        (o / d).mkdir(parents=True, exist_ok=True)
    (o / K / "platform/python/kern.py").write_text("VALUE = 1\n")
    (o / K / "platform/tests/test_gpu_x.py").write_text(TEST_FILE)
    (o / K / "library/README").write_text("l\n")
    (o / ".github/CODEOWNERS").write_text(f"/{K}/ @acme/team-p\n")
    git(o, "init", "-q", "-b", "develop")
    git(o, "add", "-A")
    git(o, "commit", "-qm", "develop: first")
    git(o, "checkout", "-qb", "pr7")
    (o / K / "platform/python/kern.py").write_text("VALUE = 2\n")
    git(o, "commit", "-qam", "pr: change the value")
    git(o, "update-ref", "refs/pull/7/head", "pr7")
    head = git(o, "rev-parse", "pr7")
    git(o, "checkout", "-q", "develop")
    (o / K / "library/README").write_text("l2\n")
    git(o, "commit", "-qam", "develop: another change (#1)")
    git(o, "config", "uploadpack.allowFilter", "true")
    git(o, "config", "uploadpack.allowAnySHA1InWant", "true")
    return o.resolve().as_uri(), git(o, "rev-parse", "develop"), head


class _Base(unittest.TestCase):
    """A working folder, a state store, a fake cluster and its login host, and (once asked) the origin repository."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.work = self.base / "work"
        self.remote_home = self.base / "remote-home"
        self.slurm = self.base / "slurm"
        for d in (self.work, self.remote_home, self.slurm):
            d.mkdir()
        (self.slurm / "cluster.json").write_text(json.dumps(CLUSTER))
        node = self.base / "node" / "bin"
        node.mkdir(parents=True)
        # The node's Python is this one, its venv included (a link to a venv's python finds no pyvenv.cfg beside it,
        # and runs as the base Python, which may lack pytest).
        (node / "python").write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8"
        )
        (node / "python").chmod(0o755)
        self.ssh_log = self.base / "ssh.log"
        self.env = mock.patch.dict(
            os.environ,
            {
                "RKMON_GPU_SSH": str(FAKES / "ssh"),
                "FAKE_SSH_HOST": HOST,
                "FAKE_REMOTE_HOME": str(self.remote_home),
                "FAKE_SLURM_DIR": str(self.slurm),
                "FAKE_SSH_LOG": str(self.ssh_log),
            },
        )
        self.env.start()
        self.store = Store(self.base / "state")
        self.st = empty_state()
        self.url, self.dev, self.head = "file:///nonexistent", "d" * 40, "e" * 40
        # The key, in the secret's own file: the load derives gpu.ssh_key and keeps no place config.json gives.
        self.key = pathlib.Path(self.cfg()["gpu"]["ssh_key"])
        self.key.parent.mkdir(parents=True, exist_ok=True)
        self.key.write_text("not a real key\n")
        self.key.chmod(0o600)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def origin(self):
        self.url, self.dev, self.head = make_origin(self.base)

    def raw(self, mode: str = "live", **over) -> dict:
        """A configuration with the remote GPU tests on (gpu.enabled, review automation in `mode`) against the fake
        cluster, whose login host's own key the state records."""
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path=K,
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
            clone_since="2001-01-01",
        )
        c["paths"] = dict(c["paths"], working_dir=str(self.work))
        c["features"] = {"review_automation": {"enabled": True, "mode": mode}}
        c["gpu"] = dict(
            c["gpu"],
            enabled=True,
            login_host=HOST,
            ssh_user="op",
            partition="gpu-q",
            accounts=["acct-a", "acct-b"],
            gres={"gfx942": "type-a", "gfx950": "type-b"},
            remote_python=str(self.base / "node" / "bin" / "python"),
            clone_url=self.url,
            env={"FAKE_GPU": "gfx942"},
        )
        c["gpu"].update(over)
        c["remote_gpu_tests"] = {"login_host_keys": [HOST_KEY]}
        return c

    def cfg(self, mode: str = "live", **over) -> dict:
        return config.normalize(self.raw(mode, **over))

    def track(self, n: int = 7):
        self.st["prs"][str(n)] = {"n": n, "head": self.head, "status": "open"}
        self.st["develop"]["sha"] = self.dev

    def checks(self, n: int = 7, skips: str = SKIPS):
        d = self.work / "workers" / f"pr{n}" / "checks"
        d.mkdir(parents=True, exist_ok=True)
        (d / gpu.SKIPS_FILE).parent.mkdir(exist_ok=True)
        (d / gpu.SKIPS_FILE).write_text(skips)
        (d / "develop_sha").write_text(self.dev + "\n")
        (d / "head_sha").write_text(self.head + "\n")
        (d / "checks.md").write_text("# Checks\n")
        return d

    def staged(self):
        """Wait for every staging job to end (they run in the background, as in the monitor)."""
        for j in gpu.records(self.st, active=True):
            if j["state"] != "staging":
                continue
            rc = self.store.dir / "jobs" / j["stage_job"]["id"] / "rc"
            for _ in range(1200):
                if rc.exists():
                    break
                time.sleep(0.1)
            self.assertTrue(rc.exists(), "the staging job did not end")

    def slurm_cmd(self, *args) -> str:
        p = subprocess.run(
            [sys.executable, str(FAKES / "slurm.py"), *args],
            capture_output=True,
            text=True,
            env=dict(os.environ, USER="op"),
        )
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return p.stdout

    def node(self, name: str, gfx: str = "gfx942") -> str:
        """A node's own tools, first on its PATH: a rocminfo that lists `gfx` (none: a broken node) and a hipcc that
        compiles. They shadow this machine's own ROCm tools, so nothing here ever touches its GPU.
        """
        d = self.base / f"node-tools-{name}"
        d.mkdir(exist_ok=True)
        (d / "rocminfo").write_text(
            f"#!/bin/sh\necho '  Name:                    {gfx}'\n"
            if gfx
            else "#!/bin/sh\necho 'no agents'\n"
        )
        (d / "hipcc").write_text("#!/bin/sh\nexit 0\n")
        for f in ("rocminfo", "hipcc"):
            (d / f).chmod(0o755)
        return str(d)

    def run_job(self, sid: str, node: str, gfx: str = "gfx942") -> None:
        self.slurm_cmd("run", sid, "--node", node, "--path", self.node(node, gfx))

    def slurm_job(self, sid: str) -> dict:
        return json.loads((self.slurm / "jobs" / f"{sid}.json").read_text())


class TestChoosing(unittest.TestCase):
    def cfg(self, **over):
        c = home_repository.config(
            config.load_defaults()
        )  # its skip pattern: a GPU, a gfx target, HIP, ROCm
        c["gpu"] = dict(c["gpu"], gres={"gfx942": "type-a", "gfx950": "type-b"}, **over)
        return c

    def test_only_gpu_only_cases_go_each_to_the_target_its_reason_names(self):
        skips = [
            ("t/a.py::x", "needs a gfx942 GPU"),
            ("t/a.py::y", "requires gfx950 hardware"),
            ("t/a.py::z", "slow; run with --slow"),
            ("t/b.py", "no GPU available"),
            ("t/c.py::w", "needs gfx1201"),
            ("t/d.py::v", "only on gfx94x"),
            ("t/a.py::x", "needs a gfx942 GPU"),
            (
                "t/e.py::u",
                "rkmon: this host's hipcc cannot compile <hip/hip_runtime.h>",
            ),
        ]
        got = gpu.select(self.cfg(), skips)
        self.assertEqual(
            got["by_arch"],
            {
                "gfx942": ["t/a.py::x", "t/b.py", "t/d.py::v", "t/e.py::u"],
                "gfx950": ["t/a.py::y"],
            },
        )
        self.assertEqual(
            got["no_type"], {"gfx1201": 1}
        )  # no GPU type configured for it: counted, not run

    def test_caps_run_files_and_limit_the_targets(self):
        skips = [(f"t/f{i % 3}.py::c{i}", "needs a gfx942 GPU") for i in range(9)] + [
            ("t/g.py::k", "needs gfx950")
        ]
        got = gpu.select(self.cfg(max_cases=5, max_jobs_per_pr=1), skips)
        self.assertEqual(
            got["by_arch"], {"gfx942": ["t/f0.py", "t/f1.py", "t/f2.py"]}
        )  # past the cap: the files
        self.assertEqual((got["files"], got["over"]), ({"gfx942": True}, {"gfx950": 1}))
        self.assertEqual(
            gpu.select(self.cfg(default_arch="gfx950"), [("t/a.py::x", "no GPU here")])[
                "by_arch"
            ],
            {"gfx950": ["t/a.py::x"]},
        )

    def test_bad_settings_are_named(self):
        bad = gpu.problems(
            {
                "gpu": {
                    "login_host": "op@host",
                    "accounts": ["ok", "bad name"],
                    "gres": {"942": "x"},
                    "time_limit": "an hour",
                    "skip_pattern": "(",
                    "max_cases": 0,
                    "env": {"1X": "y"},
                    "default_arch": "gfx90a",
                    "setup": "a\nb",
                }
            }
        )
        for key in (
            "login_host",
            "accounts",
            "gres",
            "time_limit",
            "skip_pattern",
            "max_cases",
            "env",
            "default_arch",
            "setup",
        ):
            self.assertTrue(
                any(p.startswith(f"{deploy_tree.name('gpu.' + key)} is ") for p in bad),
                (key, bad),
            )
        self.assertEqual(gpu.problems({"gpu": dict(config.load_defaults()["gpu"])}), [])
        self.assertIn(
            "features.review_automation.remote_gpu_tests.cluster.time_limit is 'x'",
            "; ".join(
                config.problems(dict(config.load_defaults(), gpu={"time_limit": "x"}))
            ),
        )

    def test_each_value_takes_the_form_of_its_deploy_settings_key(self):
        """gpu.problems reads each form from the deploy schema (gpu.forms): a key's type, its bounds, its pattern and a
        map's names, each said with the words of the key's description; a placeholder left in a value is refused, as the
        file's checks refuse it, so no ssh option comes in through one; gpu.base_env takes gpu.env's form; a blank value
        is the default's."""
        held = deploy_tree.held()
        given = {
            "ssh_port": "22",
            "enabled": "yes",
            "time_limit": 60,
            "gpus_per_job": 0,
            "login_host": "-oProxyCommand=x<host>",
            "base_env": {"1X": "y"},
        }
        bad = gpu.problems({"gpu": given})
        for key in given:
            file_key = held["gpu.env" if key == "base_env" else f"gpu.{key}"]
            form = settings_file.closing_of(deploy_tree.description(file_key))[0]
            lines = [
                p for p in bad if p.startswith(f"{deploy_tree.name('gpu.' + key)} is ")
            ]
            self.assertEqual(len(lines), 1, (key, bad))
            self.assertTrue(lines[0].endswith(f": {form}"), (key, lines))
        self.assertEqual(len(bad), len(given), bad)
        self.assertIn(
            '"-oProxyCommand=x<host>" has a placeholder, <host>', "\n".join(bad)
        )
        blank = {"login_host": "", "qos": None, "remote_dir": "", "accounts": []}
        self.assertEqual(gpu.problems({"gpu": blank}), [])

    def test_each_job_sets_the_repositorys_variables_first(self):
        """gpu.env lays the configuration's variables over gpu.base_env, the repository defaults': a configuration that
        leaves them out still has each job set them. The generic defaults set none."""
        base = {"base_env": {"ROCKE_BACKEND": "python"}}
        for env, want in (
            (None, {"ROCKE_BACKEND": "python"}),
            ({}, {"ROCKE_BACKEND": "python"}),
            ({"EXTRA": "1"}, {"ROCKE_BACKEND": "python", "EXTRA": "1"}),
            ({"ROCKE_BACKEND": "cpp"}, {"ROCKE_BACKEND": "cpp"}),
        ):
            with self.subTest(env=env):
                self.assertEqual(
                    gpu.settings({"gpu": dict(base, env=env)})["env"], want
                )
        home = home_repository.config(config.load_defaults())
        self.assertEqual(gpu.settings(home)["env"], {"ROCKE_BACKEND": "python"})
        self.assertEqual(gpu.settings(config.load_defaults())["env"], {})

    def test_a_blank_skip_pattern_sends_no_case(self):
        c = config.load_defaults()
        c["gpu"] = dict(c["gpu"], skip_pattern="", gres={"gfx942": "type-a"})
        got = gpu.select(c, [("t/a.py::x", "needs a gfx942 GPU")])
        self.assertEqual(got["by_arch"], {})


class TestRow(unittest.TestCase):
    def job(self, **over):
        j = {
            "id": "j1",
            "pr": 7,
            "head": "abcdef1234" * 4,
            "arch": "gfx942",
            "cases": CASES,
            "state": "queued",
            "result": None,
            "node": "",
        }
        j.update(over)
        return j

    def test_the_row_is_replaced_in_place_or_added_after_our_checks(self):
        body = gpu.with_row(REVIEW, "⏳ Pending<br>• x")
        lines = body.splitlines()
        self.assertEqual(
            lines[
                lines.index(next(x for x in lines if x.startswith("| Our checks |")))
                + 1
            ],
            "| GPU tests | ⏳ Pending<br>• x |",
        )
        again = gpu.with_row(body, "🟢 Passed<br>• y")
        self.assertEqual(again.count("| GPU tests |"), 1)
        self.assertIn("| GPU tests | 🟢 Passed<br>• y |", again)
        bare = "Automated review\n\n| A | B |\n|---|---|\n| x | y |\n\nend\n"
        self.assertIn("| x | y |\n| GPU tests | c |\n\nend", gpu.with_row(bare, "c"))
        self.assertEqual(gpu.with_row(REVIEW, ""), REVIEW)

    def test_the_cell_says_which_cases_where_and_never_a_node_name(self):
        st = {"gpu": {"jobs": [self.job()]}}
        cell = gpu.row_cell(st, 7, "abcdef1234" * 4)
        self.assertTrue(
            cell.startswith(
                "⏳ Pending<br>• gfx942: 2 GPU-only cases from `test_gpu_x.py`: queued"
            )
        )
        res = {
            "status": "pr-fail",
            "pr_failed": [CASES[0]],
            "old_failed": [],
            "unrun": [],
            "mode": "merge",
            "detail": "1 failing",
        }
        st["gpu"]["jobs"][0].update(state="ended", node="node-a01", result=res)
        cell = gpu.row_cell(st, 7, "abcdef1234" * 4, "red")
        self.assertTrue(cell.startswith("🔴 Failed<br>• gfx942: 1 fail at"), cell)
        self.assertNotIn("(node", cell)  # the architecture only, never a node
        self.assertIn("develop passes them on the same node (ran)", cell)
        self.assertNotIn("node-a01", cell)
        self.assertEqual(post.public(gpu.with_row(REVIEW, cell)), [])
        st["gpu"]["jobs"][0]["result"] = dict(res, status="pass", detail="2 passed")
        self.assertTrue(
            gpu.row_cell(st, 7, "abcdef1234" * 4).startswith(
                "🟢 Passed<br>• gfx942: 2 passed at"
            )
        )
        st["gpu"]["jobs"][0]["result"] = {
            "status": "error",
            "detail": "not queued: fatal: unable to access "
            "'https://git.example.com/x': timed out",
        }
        cell = gpu.row_cell(st, 7, "abcdef1234" * 4)
        self.assertEqual(
            cell,
            "🟡 Not conclusive<br>• gfx942: not run (it could not be queued; the operator is told)",
        )


class TestFlow(_Base):
    """The whole path against the fakes: staged and queued (account fallback), the pending row in the post, run on a
    node, collected, cleaned up, and the row and the notice."""

    def setUp(self):
        super().setUp()
        self.origin()
        self.track()
        self.checks()

    def queue_it(self, cfg):
        lines = gpu.after_checks(cfg, self.st, self.store, 7, self.head)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(
            lines[0].startswith("GPU_STAGING #7 gfx942: 2 GPU-only case(s)"), lines
        )
        self.staged()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(len(lines), 1, (lines, self.stage_out()))
        return lines[0]

    def stage_out(self) -> str:
        j = gpu.records(self.st)[0]
        d = gpu.local_dir(self.cfg(), j["id"])
        return "".join(
            (d / f).read_text() for f in ("stage.out", "stage.err") if (d / f).exists()
        )

    def test_a_work_folder_with_a_space_reaches_the_job_whole(self):
        """gpu.remote_dir is one line, which may hold a space: the job's folder reaches job.sh whole, as the argument
        sbatch passes it, never written into the script, so the job runs in its own folder on the node.
        """
        cfg = self.cfg(remote_dir="rkmon gpu")
        self.queue_it(cfg)
        j = gpu.records(self.st)[0]
        folder = self.remote_home / "rkmon gpu" / "jobs" / j["id"]
        self.assertEqual(self.slurm_job(j["slurm"])["args"], [str(folder)])
        self.run_job(j["slurm"], "node-a01")
        self.assertEqual(
            ((folder / "node").read_text(), (folder / "probe").read_text()),
            ("node-a01\n", "ok\n"),
        )

    @unittest.skipUnless(HAVE_PYTEST, "the fake node runs the cases with pytest")
    def test_queued_run_on_a_node_collected_and_reported(self):
        cfg = self.cfg()
        line = self.queue_it(cfg)
        j = gpu.records(self.st)[0]
        self.assertTrue(
            line.startswith(
                f"GPU_QUEUED #7 gfx942 at {self.head[:10]}: 2 GPU-only case(s), job 1001 "
                "(account acct-b; acct-a refused it)"
            ),
            line,
        )
        self.assertEqual(
            (j["state"], j["slurm"], j["account"], j["refused"][0][0]),
            ("queued", "1001", "acct-b", "acct-a"),
        )
        checks = (self.work / "workers/pr7/checks/checks.md").read_text()
        self.assertIn("## GPU tests (remote)", checks)
        self.assertIn(
            "2 GPU-only case(s) this machine skipped are queued on a remote GPU node: gfx942 (2)",
            checks,
        )
        sj = self.slurm_job(
            "1001"
        )  # one GPU, the configured partition and type, the job's own folder
        self.assertEqual(
            (sj["partition"], sj["gres"], sj["name"], sj["state"]),
            ("gpu-q", "gpu:type-a:1", j["id"], "PENDING"),
        )
        self.assertTrue(j["id"].startswith(gpu.prefix(cfg) + "-pr7-gfx942-"))
        hosts = {ln.split("\t")[0] for ln in self.ssh_log.read_text().splitlines()}
        self.assertEqual(hosts, {HOST})
        # Pending: the review is posted on time with a pending row.
        self.assertEqual(gpu.tick(cfg, self.st, self.store), [])
        body = gpu.for_post(self.st, 7, self.head, REVIEW)
        self.assertIn(
            "| GPU tests | ⏳ Pending<br>• gfx942: 2 GPU-only cases from `test_gpu_x.py`: queued on a remote "
            "GPU node; this row is updated when they finish |",
            body,
        )
        url = "https://github.com/ROCm/r/pull/7#issuecomment-4242"
        gpu.mark_posted(self.st, 7, self.head, url, body)
        # The node runs it: the merge fails test_value, develop passes it there.
        self.run_job("1001", "node-a01")
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(
            lines[0].startswith(
                f"GPU_DONE #7 gfx942 at {self.head[:10]} on node node-a01: pr-fail"
            ),
            lines,
        )
        self.assertIn("update the review's row: `rk gpu-row 7`", lines[0])
        self.assertTrue(lines[1].startswith("GPU_FAIL #7 gfx942"), lines)
        j = gpu.records(self.st)[0]
        self.assertEqual(
            (
                j["state"],
                j["node"],
                j["result"]["status"],
                j["result"]["pr_failed"],
                j["result"]["mode"],
            ),
            ("ended", "node-a01", "pr-fail", [CASES[0]], "merge"),
        )
        notice = pathlib.Path(j["notice"]).read_text()
        self.assertTrue(notice.startswith(post.notice_line(cfg, "gpu") + "\n"))
        self.assertIn(f"| `{CASES[0]}` | fails | passes |", notice)
        self.assertNotIn("node-a01", notice)
        self.assertEqual(
            post.check(
                dict(cfg, review_first_line=post.notice_line(cfg, "gpu")), notice
            ),
            [],
        )
        # Cleaned up: the job folder on the login host is gone, and so are its worktrees.
        self.assertFalse((self.remote_home / "rkmon-gpu" / "jobs" / j["id"]).exists())
        wts = git(self.remote_home / "rkmon-gpu" / "repo", "worktree", "list")
        self.assertEqual(len(wts.splitlines()), 1, wts)
        self.assertEqual(gpu.queue_lines(cfg), ["no job of op in the queue: clean"])
        # The row, in place: in prototype it journals only; live, it edits our comment, once.
        calls = []

        def fake_run(args, *a, **kw):
            calls.append(args)
            body_file = next(a for a in args if a.startswith("body=@"))[6:]
            calls.append(pathlib.Path(body_file).read_text())
            return subprocess.CompletedProcess(args, 0, "", "")

        boom = mock.Mock(side_effect=AssertionError("prototype wrote to GitHub"))
        with mock.patch.object(gh, "rest", boom), mock.patch.object(gh, "write", boom):
            out = gpu.update_row(self.cfg(mode="prototype"), self.st, self.store, 7)
        self.assertTrue(
            out[0].startswith("PROTOTYPE: #7's GPU tests row would become: 🟠 Failed"),
            out,
        )
        self.assertEqual(self.store.ledger(), [])
        with mock.patch.object(gh, "rest", return_value=body), mock.patch.object(
            gh, "write", side_effect=fake_run
        ):
            out = gpu.update_row(cfg, self.st, self.store, 7, "red")
        self.assertTrue(out[0].startswith("updated: #7's GPU tests row"), out)
        self.assertEqual(
            calls[0][:4],
            ["api", "-X", "PATCH", "repos/acme/widgets/issues/comments/4242"],
        )
        self.assertIn(
            f"| GPU tests | 🔴 Failed<br>• gfx942: 1 fail at "
            f"`{self.head[:8]}`; develop passes them on the same node (ran): `{CASES[0]}` |",
            calls[1],
        )
        self.assertEqual(calls[1].count("| GPU tests |"), 1)
        self.assertEqual([r["kind"] for r in self.store.ledger()], ["gpu-row"])
        self.assertEqual(
            gpu.update_row(cfg, self.st, self.store, 7, "red"),
            ["#7: no posted GPU tests row is out of date"],
        )

    @unittest.skipUnless(HAVE_PYTEST, "the fake node runs the cases with pytest")
    def test_develops_own_case_by_hand(self):
        """`rk gpu-run --develop --arch A --case ID`: the operator's first real job is this one."""
        cfg = self.cfg()
        lines = gpu.run_by_hand(
            cfg, self.st, self.store, None, "gfx942", [CASES[1]], develop_only=True
        )
        self.assertEqual(
            lines,
            [
                "GPU_STAGING develop gfx942: 1 GPU-only case(s) being queued on a remote GPU node"
            ],
        )
        self.staged()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertTrue(
            lines[0].startswith(
                f"GPU_QUEUED develop gfx942 at {self.dev[:10]}: 1 GPU-only case(s), job "
                "1001 (account acct-b"
            ),
            lines,
        )
        self.run_job("1001", "node-a01")
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(
            lines[0].startswith(
                f"GPU_DONE develop gfx942 at {self.dev[:10]} on node node-a01: pass: "
                "1 passed"
            ),
            lines,
        )
        self.assertEqual(gpu.records(self.st)[0]["result"]["mode"], "develop")
        self.assertEqual(gpu.queue_lines(cfg), ["no job of op in the queue: clean"])

    def test_a_long_wait_is_reported_once_and_never_resubmitted(self):
        cfg = self.cfg(pending_report_min=30)
        self.queue_it(cfg)
        j = gpu.records(self.st)[0]
        j["submitted"] = "2000-01-01T00:00:00Z"
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(
            lines[0].startswith("GPU_PENDING_LONG #7 gfx942 job 1001: waiting"), lines
        )
        self.assertIn("(Resources); reported once, not resubmitted", lines[0])
        self.assertEqual(gpu.tick(cfg, self.st, self.store), [])  # once
        self.assertEqual(
            len((self.slurm / "sbatch.log").read_text().splitlines()), 2
        )  # acct-a refused, acct-b took it
        row, _key, needs = gpu.board_row(cfg, self.st)
        self.assertEqual(row[0], "GPU tests")
        self.assertIn("waiting long: #7 gfx942", row[2])
        self.assertIn("keep waiting, or give up with `rk gpu-cancel 7`?", needs[0])
        lines = gpu.cancel(cfg, self.st, self.store, "7")
        self.assertTrue(
            lines[0].startswith("GPU_CANCELLED #7 gfx942 job 1001: given up"), lines
        )
        self.assertEqual(self.slurm_job("1001")["state"], "CANCELLED")

    def test_a_node_that_fails_the_probe_is_avoided_and_reported(self):
        cfg = self.cfg()
        self.queue_it(cfg)
        self.run_job("1001", "node-a02", gfx="")  # its rocminfo lists no device
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertTrue(
            lines[0].startswith(
                "GPU_NODE_BROKEN node-a02 (gfx942): no gfx942 device visible (rocminfo): "
                "add it to features.review_automation.remote_gpu_tests.cluster.excluded_nodes; #7's job goes again "
                "without it"
            ),
            lines,
        )
        self.staged()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertTrue(lines[0].startswith("GPU_QUEUED #7 gfx942"), lines)
        self.assertEqual(self.slurm_job("1002")["exclude"], "node-a02")
        self.assertIn("node-a02", gpu.status_lines(cfg, self.st)[-1])
        self.run_job(
            "1002", "node-a01", gfx=""
        )  # the one resend is spent: reported, not sent again
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertIn(
            "is not sent again (features.review_automation.remote_gpu_tests.bad_node_resubmits 1 reached)",
            lines[0],
        )
        self.assertEqual(gpu.records(self.st)[0]["result"]["status"], "error")
        self.assertEqual(
            len((self.slurm / "sbatch.log").read_text().splitlines()), 4
        )  # two tries, two accounts each

    def test_a_rerun_asks_for_the_node_of_the_last_run(self):
        """The node's name is kept locally, so a re-run can ask for the same machine;
        posts never name it."""
        cfg = self.cfg()
        with self.assertRaisesRegex(
            gpu.GpuError, "no earlier job of #7 on gfx942 ran on a known node"
        ):
            gpu.run_by_hand(
                cfg, self.st, self.store, 7, "gfx942", [CASES[0]], same_node=True
            )
        self.queue_it(cfg)
        self.run_job(
            "1001", "node-a01", gfx=""
        )  # the probe fails there: the job goes again without it
        gpu.tick(cfg, self.st, self.store)
        self.staged()
        gpu.tick(cfg, self.st, self.store)
        self.run_job("1002", "node-a02")
        gpu.tick(cfg, self.st, self.store)
        self.assertEqual(gpu.last_node(self.st, 7, "gfx942"), "node-a02")
        lines = gpu.run_by_hand(
            cfg, self.st, self.store, 7, "gfx942", [CASES[0]], same_node=True
        )
        self.assertEqual(
            lines,
            [
                "GPU_STAGING #7 gfx942: 1 GPU-only case(s) being queued on a remote GPU node (on "
                "node node-a02, as asked)"
            ],
        )
        self.staged()
        gpu.tick(cfg, self.st, self.store)
        self.assertEqual(
            (self.slurm_job("1003")["nodelist"], self.slurm_job("1003")["exclude"]),
            ("node-a02", ""),
        )
        self.run_job(
            "1003", "node-a02", gfx=""
        )  # the node asked for fails the probe: not sent elsewhere
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertIn("asked for this node, so it is not sent again", lines[0])
        self.assertNotIn("node-a0", gpu.row_cell(self.st, 7, self.head))
        with self.assertRaisesRegex(gpu.GpuError, "is not a node name"):
            gpu.run_by_hand(
                cfg, self.st, self.store, 7, "gfx942", [CASES[0]], node="n1;reboot"
            )
        with self.assertRaisesRegex(gpu.GpuError, "not both"):
            gpu.run_by_hand(
                cfg,
                self.st,
                self.store,
                7,
                "gfx942",
                [CASES[0]],
                same_node=True,
                node="node-a01",
            )
        with self.assertRaisesRegex(
            gpu.GpuError,
            r"node node-a03 is in features\.review_automation\.remote_gpu_tests\.cluster\.excluded_nodes:",
        ):
            gpu.run_by_hand(
                self.cfg(exclude=["node-a03"]),
                self.st,
                self.store,
                7,
                "gfx942",
                [CASES[0]],
                node="node-a03",
            )

    def test_a_pr_that_moves_on_cancels_its_jobs(self):
        cfg = self.cfg()
        self.queue_it(cfg)
        self.st["prs"]["7"]["head"] = "f" * 40
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertTrue(
            lines[0].startswith(
                "GPU_CANCELLED #7 gfx942 job 1001: superseded by `ffffffff`"
            ),
            lines,
        )
        self.assertEqual(self.slurm_job("1001")["state"], "CANCELLED")

    def test_our_leftovers_are_cancelled_and_others_jobs_left_alone(self):
        cfg = self.cfg()
        self.queue_it(cfg)
        mine = self.slurm_cmd(
            "sbatch",
            "--parsable",
            f"--job-name={gpu.prefix(cfg)}-pr9-gfx942-old",
            "--partition=gpu-q",
            "--account=acct-b",
            "--gres=gpu:type-a:1",
            "/bin/true",
        ).strip()
        theirs = self.slurm_cmd(
            "sbatch",
            "--parsable",
            "--job-name=interactive",
            "--partition=gpu-q",
            "--account=acct-b",
            "--gres=gpu:type-a:1",
            "/bin/true",
        ).strip()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(
            lines, [f"GPU_CLEANUP: cancelled 1 leftover job(s) of this monitor: {mine}"]
        )
        self.assertEqual(
            [self.slurm_job(sid)["state"] for sid in (mine, theirs, "1001")],
            ["CANCELLED", "PENDING", "PENDING"],
        )

    def test_no_account_takes_it(self):
        cfg = self.cfg(accounts=["acct-a", "acct-c"])
        gpu.after_checks(cfg, self.st, self.store, 7, self.head)
        self.staged()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertEqual(
            lines,
            [
                "GPU_NOT_QUEUED #7 gfx942: every account was refused (rk gpu-status; tell the user if "
                "it repeats)"
            ],
        )
        self.assertEqual(
            [a for a, _w in gpu.records(self.st)[0]["refused"]], ["acct-a", "acct-c"]
        )

    def test_the_job_asks_for_its_quality_of_service_and_its_gpu_count(self):
        """Each sbatch of a job carries the quality of service (gpu.qos) as --qos, and the GPU count
        (gpu.gpus_per_job) in --gres; a blank quality of service sends no --qos, so the account's default holds.
        """
        self.queue_it(self.cfg(qos="normal", gpus_per_job=2))
        sent = [
            json.loads(line)
            for line in (self.slurm / "sbatch.log").read_text().splitlines()
        ]
        self.assertEqual(len(sent), 2)  # acct-a refused, then acct-b
        for argv in sent:
            self.assertIn("--qos=normal", argv)
            self.assertIn("--gres=gpu:type-a:2", argv)
        self.assertEqual(self.slurm_job("1001")["gres"], "gpu:type-a:2")
        script = gpu.stage_script(self.cfg(), gpu.records(self.st)[0])
        self.assertNotIn("--qos", script)
        self.assertIn("--gres=gpu:type-a:1 ", script)

    def test_each_cluster_value_reaches_sbatch_whole(self):
        """The quality of service, the partition and the constraint may hold & | and ! (Slurm's constraint
        syntax), which the shell reads: the sbatch line quotes each, so the shell hands each to sbatch whole. The
        line runs in bash with an sbatch that prints its arguments."""
        self.queue_it(self.cfg())
        odd = {"qos": "a&b|!c", "partition": "p&q|!r", "constraint": "x&y|!z"}
        script = gpu.stage_script(self.cfg(**odd), gpu.records(self.st)[0])
        [line] = [s for s in script.splitlines() if s.startswith("sub() { sbatch ")]
        shell = "\n".join(['sbatch() { printf "%s\\n" "$@"; }', line, "sub acct-a"])
        argv = subprocess.run(
            ["bash", "-c", shell], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
        for key, value in odd.items():
            self.assertIn(f"--{key}={value}", argv)
        self.assertIn("--account=acct-a", argv)

    def test_only_the_configured_host_is_ever_contacted(self):
        raw = self.raw("live", login_host="other.example.com")
        # Its own key recorded: one recorded for the login host named before is none of its keys (host_keys.recorded).
        raw["remote_gpu_tests"] = {
            "login_host_keys": [HOST_KEY.replace(HOST, "other.example.com")]
        }
        cfg = config.normalize(raw)
        gpu.after_checks(cfg, self.st, self.store, 7, self.head)
        self.staged()
        lines = gpu.tick(cfg, self.st, self.store)
        self.assertTrue(
            lines[0].startswith(
                "GPU_NOT_QUEUED #7 gfx942: ssh: connect to host other.example.com port "
                "22: refused by the test's fake"
            ),
            lines,
        )


class TestTheSettings(unittest.TestCase):
    def test_each_key_of_the_block_reaches_the_jobs(self):
        """Each key of the remote GPU tests' block (features.review_automation.remote_gpu_tests) feeds the
        configuration key gpu.py reads, and none waits for its step (deploy_tree.NOT_CARRIED); the quality of service
        and the GPU count a deploy settings file sets reach the jobs' settings."""
        from rkmon_lib import deploy_tree, file_types, settings_file

        block = "features.review_automation.remote_gpu_tests."
        fed = dict(deploy_tree.feeds())
        keys = [
            path
            for path, _node, _required in settings_file.schema_keys(
                file_types.DEPLOY.load()
            )
            if path.startswith(block)
        ]
        self.assertGreater(len(keys), 20)
        for path in keys:
            with self.subTest(key=path):
                self.assertTrue(fed.get(path, "").startswith("gpu."))
                self.assertNotIn(path, deploy_tree.NOT_CARRIED)
        cluster = {"qos": "normal", "gpus_per_job": 2}
        cfg = deploy_tree.to_config(
            {
                "features": {
                    "review_automation": {"remote_gpu_tests": {"cluster": cluster}}
                }
            }
        )
        g = gpu.settings(cfg)
        self.assertEqual((g["qos"], g["gpus_per_job"]), ("normal", 2))


class TestStartingAndSkipping(_Base):
    def test_off_without_review_automation_or_gpu_enabled(self):
        self.track()
        self.checks()
        for c in (
            dict(self.cfg(), features={"review_automation": {"enabled": False}}),
            dict(self.cfg(), gpu=dict(self.cfg()["gpu"], enabled=False)),
        ):
            self.assertFalse(features.on(c, "remote-gpu-tests"))
            self.assertEqual(gpu.after_checks(c, self.st, self.store, 7, self.head), [])
        self.assertEqual(gpu.records(self.st), [])
        # In prototype the jobs still queue: they write nothing to GitHub.
        self.assertTrue(features.on(self.cfg(mode="prototype"), "remote-gpu-tests"))

    def test_no_gpu_only_case_no_job(self):
        self.track()
        d = self.checks(skips="rkmon-skip:\t-\tt/a.py::x\tslow\n")
        self.assertEqual(
            gpu.after_checks(self.cfg(), self.st, self.store, 7, self.head), []
        )
        self.assertIn(
            "No GPU-only case was skipped here", (d / "checks.md").read_text()
        )

    def test_on_but_misconfigured_skips_cleanly_and_says_what_is_missing(self):
        c = self.cfg()
        c["gpu"] = dict(c["gpu"], login_host="", accounts=[], gres={})
        self.key.unlink()
        gone = features.missing(c, "remote-gpu-tests")
        cluster = "features.review_automation.remote_gpu_tests.cluster."
        for words in (
            cluster + "login_host",
            cluster + "accounts",
            cluster + "gpu_types",
            "the SSH key",
        ):
            self.assertTrue(any(words in g for g in gone), (words, gone))
        self.assertFalse(features.active(c, "remote-gpu-tests"))
        self.track()
        self.checks()
        self.assertEqual(
            gpu.after_checks(c, self.st, self.store, 7, self.head), []
        )  # nothing, no traceback
        self.assertEqual(
            features.filter_lines(c, ["GPU_DONE #7 x", "REVIEW_READY #7"]),
            ["REVIEW_READY #7"],
        )
        self.assertTrue(
            any(
                x.startswith(
                    f"FEATURE_SKIPPED remote GPU tests: on but needs {cluster}login_host"
                )
                for x in features.tick_reports(c, self.st)
            )
        )

    def test_lacking_only_its_key_it_is_set_up_by_the_key_s_command(self):
        """Remote GPU tests that lack only the SSH key for the login host are set up by that secret's own command, as
        the operator types it on the host wherever this runs: no settings edit."""
        c = self.cfg()
        self.key.unlink()
        self.assertEqual(len(features.missing(c, "remote-gpu-tests")), 1)
        for env, w in (
            ({}, "rkmon"),
            ({"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}, "./rkmon"),
        ):
            with self.subTest(w):
                self.assertEqual(
                    str(features.fix(c, "remote-gpu-tests", environ=env)),
                    f"{w} secret gpu_ssh_key, and give an SSH private key: the whole key file, from its BEGIN line "
                    "to its END line",
                )

    def test_a_key_others_may_read_is_used_through_a_private_copy(self):
        cfg = self.cfg()
        self.key.chmod(0o644)  # as a read-only mount from a Windows host shows it
        used = pathlib.Path(gpu.usable_key(cfg))
        self.assertNotEqual(used, self.key)
        self.assertEqual(
            (used.read_text(), used.stat().st_mode & 0o777), ("not a real key\n", 0o600)
        )
        self.assertEqual(used, self.key.with_name(self.key.name + ".private-copy"))
        argv = gpu.ssh_argv(cfg, "true")
        self.assertEqual(argv[argv.index("-i") + 1], str(used))
        self.assertIn("BatchMode=yes", argv)

    def test_the_private_copy_is_never_written_through_a_link(self):
        """The copy is a private file: refreshed when the key changes, set to 600 when others may read it, and a link
        in its place is refused: the file it points to stays as it was, and ssh is given the configured key.
        """
        cfg = self.cfg()
        self.key.chmod(0o644)
        used = pathlib.Path(gpu.usable_key(cfg))
        self.key.write_text("another made-up key\n")
        self.assertEqual(gpu.usable_key(cfg), str(used))
        self.assertEqual(used.read_text(), "another made-up key\n")
        used.chmod(0o644)
        self.assertEqual(gpu.usable_key(cfg), str(used))
        self.assertEqual(used.stat().st_mode & 0o777, 0o600)
        elsewhere = self.base / "elsewhere"
        elsewhere.write_text("left alone\n")
        used.unlink()
        used.symlink_to(elsewhere)
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(self.base / "home")}):
            self.assertEqual(gpu.usable_key(cfg), str(self.key))
        self.assertTrue(used.is_symlink())
        self.assertEqual(elsewhere.read_text(), "left alone\n")

    def test_ssh_trusts_the_login_hosts_own_keys_the_state_records_alone(self):
        """ssh checks the login host's key strictly against one known_hosts file, which holds the keys the state
        records (remote_gpu_tests.login_host_keys) and nothing else, reads neither its own known_hosts nor the
        system's, and asks no command (KnownHostsCommand) and no DNS record (VerifyHostKeyDNS) for a key, whatever
        the ssh configuration says. A login host whose key is not recorded is refused before ssh runs: when no key is
        recorded (a known_hosts file the configuration names is not read), and when only another host's key is, which
        is none of its keys (host_keys.recorded).
        """
        cfg = self.cfg()
        known = self.work / "gpu_known_hosts"
        argv = gpu.ssh_argv(cfg, "true")
        options = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-o"]
        self.assertIn("StrictHostKeyChecking=yes", options)
        self.assertEqual(
            [o for o in options if "KnownHosts" in o],
            [
                f"UserKnownHostsFile={known}",
                "GlobalKnownHostsFile=/dev/null",
                "KnownHostsCommand=none",
            ],
        )
        self.assertIn("VerifyHostKeyDNS=no", options)
        self.assertEqual(known.read_text(), HOST_KEY + "\n")
        p = gpu.remote(cfg, "echo reached\n")
        self.assertEqual((p.returncode, p.stdout), (0, "reached\n"), p.stderr)

        given = self.base / "known_hosts"
        given.write_text(HOST_KEY + "\n")
        bare = self.cfg(known_hosts=str(given))
        bare["remote_gpu_tests"] = {"login_host_keys": []}
        contacted = self.ssh_log.read_text()
        p = gpu.remote(bare, "echo reached\n")
        self.assertEqual((p.returncode, p.stdout), (255, ""))
        self.assertIn("the GPU login host's own key is not recorded: ", p.stderr)
        self.assertIn(
            " login shows it, and records it once you accept it",
            p.stderr,
        )
        self.assertEqual(self.ssh_log.read_text(), contacted)  # ssh did not run
        self.assertIn("UserKnownHostsFile=/dev/null", gpu.ssh_argv(bare, "true"))
        self.assertTrue(
            gpu.check(bare)[0].startswith(
                "FAIL the GPU login host's own key is not recorded: "
            )
        )
        gone = features.missing(bare, "remote-gpu-tests")
        self.assertTrue(
            any(g.startswith("the login host's own key (") for g in gone), gone
        )
        self.assertFalse(features.active(bare, "remote-gpu-tests"))

        other = self.cfg()
        other["remote_gpu_tests"] = {
            "login_host_keys": [HOST_KEY.replace(HOST, "other.example.com")]
        }
        p = gpu.remote(other, "echo reached\n")
        self.assertEqual((p.returncode, p.stdout), (255, ""))
        self.assertIn("the GPU login host's own key is not recorded: ", p.stderr)
        self.assertEqual(self.ssh_log.read_text(), contacted)  # ssh did not run
        self.assertIn("UserKnownHostsFile=/dev/null", gpu.ssh_argv(other, "true"))

    def test_the_staging_script_tries_the_accounts_in_order_and_keeps_the_key_here(
        self,
    ):
        self.track()
        cfg = self.cfg(constraint="SITE-A", exclude=["node-x1"])
        job = {
            "id": "rkmon-abc-pr7-gfx942-1",
            "pr": 7,
            "head": self.head,
            "develop": self.dev,
            "arch": "gfx942",
            "gres": "type-a",
            "cases": CASES,
            "mode": "merge",
            "what": "#7",
            "excluded": ["node-y2"],
        }
        s = gpu.stage_script(cfg, job)
        self.assertIn("for A in acct-a acct-b; do", s)
        self.assertIn("--constraint=SITE-A --exclude=node-x1,node-y2", s)
        self.assertIn("--gres=gpu:type-a:1", s)
        self.assertIn("+refs/pull/7/head:refs/rkmon/pr7", s)
        self.assertNotIn(str(self.key), s)  # the key never leaves this machine
        js = gpu.job_script(cfg, job)
        self.assertIn("export FAKE_GPU=gfx942", js)
        self.assertIn(
            "export ROCKE_REQUIRE_GFX942=1", js
        )  # a case for this target fails, never skips, without its GPU
        self.assertIn(
            'python -I -B "$J/tools/pytest_vs_base.py"', js
        )  # the node's python runs it
        self.assertIn(
            "\"$J/tools/pytest_vs_base.py\" <<'RKMON_EOF'", s
        )  # staged beside its plugins

    def test_the_job_starts_in_its_folder_with_the_nodes_temp_folder(self):
        """sbatch starts the job in its own folder, and the job's temp folder is under TMPDIR, else under the
        system's own temp folder (where mktemp puts its files): the same folders a node gave the job before.
        """
        self.track()
        job = {
            "id": "rkmon-abc-pr7-gfx942-1",
            "pr": 7,
            "head": self.head,
            "develop": self.dev,
            "arch": "gfx942",
            "gres": "type-a",
            "cases": CASES,
            "mode": "merge",
            "what": "#7",
        }
        self.assertIn(
            '--chdir="$J" --output="$J/slurm.out"', gpu.stage_script(self.cfg(), job)
        )
        line = next(
            ln
            for ln in gpu.job_script(self.cfg(), job).splitlines()
            if ln.startswith("export TMPDIR=")
        )
        probe = line.split(";")[0] + '; echo "$TMPDIR"'
        env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}
        old_default = (
            os.sep + "tmp"
        )  # the folder the job script named before (a Linux node's temp folder)
        for extra in ({}, {"TMPDIR": str(self.base)}):
            with self.subTest(tmpdir=extra.get("TMPDIR")):
                p = subprocess.run(
                    ["bash", "-c", probe],
                    env=dict(env, SLURM_JOB_ID="42", **extra),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    p.stdout.strip(), f"{extra.get('TMPDIR', old_default)}/rkmon-42"
                )

    def test_the_job_removes_its_temp_folder_however_it_ends(self):
        """The job's folder under the node's temp folder goes when the job ends: after its run, and after a node probe
        that failed. A setup that points TMPDIR elsewhere keeps that folder: the job removes only the one it made.
        """
        job = {
            "id": "rkmon-abc-pr7-gfx942-1",
            "pr": 7,
            "head": self.head,
            "develop": self.dev,
            "arch": "gfx942",
            "gres": "type-a",
            "cases": CASES,
            "mode": "merge",
            "what": "#7",
        }
        node_tmp, elsewhere = self.base / "node-tmp", self.base / "setup-tmp"
        node_tmp.mkdir()
        elsewhere.mkdir()
        (elsewhere / "keep").write_text("the setup's own\n")
        for i, (what, gfx, setup) in enumerate(
            (
                ("a run", "gfx942", ""),
                ("a failed probe", "", ""),
                ("a setup's TMPDIR", "gfx942", f"export TMPDIR={elsewhere}"),
            )
        ):
            with self.subTest(what):
                jdir = self.base / f"job{i}"
                jdir.mkdir()
                (jdir / "cases.txt").write_text("\n".join(CASES) + "\n")
                (jdir / "mode").write_text("merge\n")
                script = jdir / "job.sh"
                script.write_text(gpu.job_script(self.cfg(setup=setup), job))
                p = subprocess.run(
                    [
                        "bash",
                        str(script),
                        str(jdir),
                    ],  # the job folder, as sbatch passes it
                    cwd=jdir,
                    env=dict(
                        os.environ,
                        TMPDIR=str(node_tmp),
                        SLURM_JOB_ID="42",
                        SLURMD_NODENAME="node-a01",
                        PATH=f"{self.node('node-a01', gfx)}{os.pathsep}{os.environ['PATH']}",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertTrue((jdir / "ended").is_file())  # the job ran to its end
                self.assertEqual(list(node_tmp.iterdir()), [])
        self.assertTrue((elsewhere / "keep").is_file())

    def test_the_commands_of_the_feature(self):
        home = self.base / "home"
        home.mkdir()
        # gpu-run works in prototype too: it writes nothing to GitHub.
        (home / "config.json").write_text(json.dumps(self.raw(mode="prototype")))

        def rk(*args):
            out = io.StringIO()
            with mock.patch.dict(
                os.environ, {"RKMON_HOME": str(home)}
            ), contextlib.redirect_stdout(out):
                rc = rkmon.main(list(args))
            return rc, out.getvalue()

        rc, out = rk("gpu-status")
        self.assertEqual((rc, out.strip()), (0, "no GPU jobs recorded"))
        rc, out = rk("gpu-run", "7")
        self.assertEqual(rc, 4)
        self.assertIn("REFUSED: develop's SHA is not known yet", out)
        st = empty_state()
        st["develop"]["sha"] = self.dev
        Store(home / "state").save(st)
        rc, out = rk("gpu-run", "7")
        self.assertIn("REFUSED: #7 is not a tracked open PR", out)
        rc, out = rk("gpu-run", "--develop")
        self.assertIn(
            "REFUSED: --develop needs the cases to run (--case ID, repeatable)", out
        )
        rc, out = rk("gpu-run", "--develop", "--arch", "gfx1201", "--case", "t.py::x")
        self.assertIn(
            "REFUSED: gfx1201 has no GPU type in features.review_automation.remote_gpu_tests.cluster.gpu_types",
            out,
        )
        rc, out = rk("gpu-check")
        self.assertIn("PASS gfx942 (type-a): 2 node(s) in gpu-q", out)
        self.assertIn("PASS account acct-b: yours", out)
        self.assertIn("FAIL account acct-a: not among your associations", out)
        rc, out = rk("gpu-queue")
        self.assertEqual((rc, out.strip()), (0, "no job of op in the queue: clean"))
        # In prototype the notice and the row write nothing to GitHub.
        note = self.base / "notice.md"
        note.write_text(
            post.notice_line(self.cfg(mode="prototype"), "gpu") + "\n\nbody\n"
        )
        rc, out = rk("notice", "7", str(note), "--kind", "gpu", "--key", "j1")
        self.assertEqual(rc, 0, out)
        self.assertIn("OFF: review posting is off, so nothing was done", out)
        rc, out = rk("gpu-row", "7")
        self.assertEqual(
            (rc, out.strip()), (0, "#7: no posted GPU tests row is out of date")
        )
        self.assertEqual(Store(home / "state").ledger(), [])

    def test_the_board_shows_the_jobs_while_the_feature_is_on(self):
        cfg = self.cfg()
        self.track()
        self.st["gpu"] = {
            "jobs": [
                {
                    "id": "j1",
                    "pr": 7,
                    "head": self.head,
                    "arch": "gfx942",
                    "state": "running",
                    "cases": CASES,
                    "node": "node-a01",
                    "slurm": "1001",
                }
            ]
        }
        bd = report.board(cfg, self.st, self.store, [], [])
        self.assertIn("| GPU tests | 0 queued, 1 running: #7 gfx942 |", bd["md"])
        off = dict(cfg, gpu=dict(cfg["gpu"], enabled=False))
        self.assertNotIn(
            "| GPU tests |", report.board(off, self.st, self.store, [], [])["md"]
        )


@unittest.skipUnless(HAVE_PYTEST, "needs pytest")
class TestSkipsAreRecorded(unittest.TestCase):
    def test_the_checks_plugin_writes_each_skip_and_its_reason(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "test_a.py").write_text(SKIPPING)
            (d / "test_b.py").write_text(
                "import pytest\n\npytest.skip('no GPU device', allow_module_level=True)\n"
            )
            tsv = d / "skips.tsv"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "tools" / "pytest_plugins"))
            r = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "rkmon_env",
                    "--rootdir",
                    str(d),
                    "test_a.py",
                    "test_b.py",
                ],
                cwd=d,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            # pytest_vs_base.py keeps the plugin's skip lines in tests_skipped.tsv
            tsv.write_text(
                "".join(
                    ln + "\n"
                    for ln in r.stdout.splitlines()
                    if ln.startswith("rkmon-skip:\t")
                )
            )
            self.assertEqual(
                sorted(gpu.read_skips(tsv)),
                [
                    ("test_a.py::test_x", "needs a gfx950 GPU"),
                    ("test_b.py", "no GPU device"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
