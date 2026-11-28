# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk provision (rkmon_lib/provision.py): each start's setup of the work volume from your confirmed
settings, asking nothing. A fresh work volume gets its folders, the worker tools, the clone and its develop
worktree, the checks' virtual environment and the agentic files, and the internal state the clone's start date, the
code owners and the record, with no config.json written; a run cut short, at a step that fails or in the middle of the
clone, leaves no record, and the next run finishes it; a failure one of your settings causes (a repository git does not
find, no operator login) names that setting, then confirm and apply, and any other the start; a run after one
that finished changes nothing, and so does each step in place; once the record is there, a start puts back a piece
that has gone missing and changes nothing else, and leaves a clone or a worktree that is there but not whole as it is,
its fix the removal by hand, then the stop and the start; --check says whether the work volume is provisioned; a work
volume an older package set up (its
config.json, or that file as the one-time move renames it) counts as provisioned and gets its record when first read,
which says it is an older install's, while a config.json written once a fresh work volume is set up changes nothing;
rk adopt before the first start, then the start, sets the work volume up around the adopted state, the earlier work
volume's record left behind and the adoption recorded, whatever config.json adopt copied (the one-time move's, never an
older install's setup), and an apply that cannot make the move due again, or that an OSError stops, says what it did,
then ERROR with its fix, as a plan that cannot read the stage does, nothing changed;
and where the local review loop's server answers is recorded in the internal state, while a server that does not answer
is a warning that holds nothing up, its fix the stop and the start again, whose probe asks it again until it answers,
and one that answers the test chat with an error refuses the model, its fix the binding put right.
"""

import contextlib
import errno
import fcntl
import hashlib
import http.client
import io
import json
import os
import pathlib
import shutil
import socket
import ssl
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "fakes"))
sys.dont_write_bytecode = True

import fake_models  # noqa: E402
import fakerepo  # noqa: E402
import rkmon  # noqa: E402
import sandbox  # noqa: E402
from rendered import one_line  # noqa: E402
from rkmon_lib import (  # noqa: E402
    adopt,
    agentic,
    checks,
    config,
    container_layout,
    deploy,
    doctor,
    file_types,
    gitops,
    image,
    models,
    paths,
    provision,
    runtime_config,
    secrets_registry,
    state_store,
    steps,
    store,
)

# A layer's settings whose configuration runs: the repository acme/widgets, its folder a/b/rocke, as fakerepo's; and
# the keys of a layer's settings that it lists as set in your file.
from test_runtime_config import SETTINGS, set_in  # noqa: E402

LOGIN = "op-login"  # the operator's login, which the agentic files' render fills in

RC = runtime_config
VERSION = image.package_version(paths.PACKAGE)
WRAPPER = "./rkmon"  # the host's command, as the container's wrapper passes it in (RKMON_WRAPPER)
# Step 7's fixes (doctor.probe_fix): a refused model's, by what refused it: a model that cannot call tools, a server
# error, any other (its binding put right, then confirm and apply); and one not checked's (the stop and the start
# again, whose provision probes the server again).
REBIND = f"fix: edit model_bindings.worker in your deploy settings file, then {WRAPPER} confirm and {WRAPPER} apply"
TOOLS = (
    "fix: set model_bindings.worker.model_id in your deploy settings file to a model that can call tools, then "
    f"{WRAPPER} confirm and {WRAPPER} apply"
)
SERVER = (
    f"fix: put the model server right (its log says what failed), then {WRAPPER} stop --drain, then "
    f"{WRAPPER} start"
)
RESTART = f"fix: {WRAPPER} stop --drain, then {WRAPPER} start"
ABSENT = "absent-model"  # a model the fake gateway's model list does not name
# The fix of a model the server does not serve (doctor.not_served_fix), the same in step 7's row and in the doctor's
# reviewer row: the server's side, then the stop and the start; or the binding's, then confirm and apply.
NOT_SERVED = (
    f"fix: pull or serve {ABSENT} there, then {WRAPPER} stop --drain, then {WRAPPER} start; or set "
    f"model_bindings.worker.model_id in your deploy settings file to a model it serves, then {WRAPPER} confirm and "
    f"{WRAPPER} apply"
)
# The header side of a refused key's fix (doctor.probe_fix) while the binding sends the key as a bearer token alone,
# which goes in no other header: how the key is sent, after the header's edit.
SENT_AS = (
    ', and set model_bindings.worker.api_key_sent_as in your deploy settings file to "header" (or "header_and_bearer" '
    "to send it as a bearer token as well)"
)
# What step 7's row says it recorded: the places of a gateway's model list and chat route, and with the name its list
# gives the model.
RECORDED = "recorded: where its model list answers and its chat route"
RECORDED_NAME = "recorded: where its model list answers, its chat route and the name its model list gives the model"
AT = "2026-01-02T03:00:00Z"  # store.now, the record's time
RECORD = {"at": AT, "package_version": VERSION}
# The code owners of fakerepo's CODEOWNERS under a/b/rocke: its root, and each area under it.
TEAMS = {
    "root": ["team-l", "team-p"],
    "platform": ["team-p"],
    "style": ["team-p"],
    "library": ["team-l"],
}
STEPS = (
    "folders",
    "worker tools",
    "clone",
    "code owners",
    "checks' virtual environment",
    "agentic files",
    "local review loop's server",
)
# The rows of the steps before step 7, each in place, as a start once the work volume is set up prints them.
IN_PLACE = [("PASS", s) for s in STEPS[:-1]]


def files(*roots) -> dict:
    """Every entry under ROOTS by its path: a file by its content's digest, a folder as such."""
    out = {}
    for root in roots:
        for p in sorted(pathlib.Path(root).rglob("*")):
            out[str(p)] = (
                "folder" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
            )
    return out


def rows(lines: list[str]) -> list[tuple[str, str]]:
    """The rows' statuses and names among LINES (checks.render's `  STATUS  NAME: FOUND`), in order."""
    out = []
    for line in lines:
        head = line[:8].strip()
        if line.startswith("  ") and head in ("PASS", "WARN", "FAIL", "OFF", "SKIP"):
            out.append((head, line[8:].split(":", 1)[0]))
    return out


def list_open(h) -> None:
    """A fake gateway's GET (fake_models.Gateway) whose model list answers without the key; any other as the fake's."""
    if h.path.split("?", 1)[0] != h.prefix + "/v1/models":
        return fake_models.Gateway.do_GET(h)
    h._t0 = time.time()
    h._send(200, {"object": "list", "data": h._listed()})


def chat_answers(status: int) -> object:
    """A fake gateway's POST, answered with the HTTP status STATUS."""

    def do_POST(h) -> None:
        h._t0 = time.time()
        h._req = h._body()
        h._send(status, {"error": {"message": f"the gateway answered {status}"}})

    return do_POST


def kept(seen: list, fake) -> object:
    """A POST of the fake FAKE (fake_models.Gateway or Ollama) that keeps each request's body in SEEN, then answers as
    FAKE does."""

    def do_POST(h) -> None:
        body = fake_models.Gateway._body(h)
        seen.append(body)
        h._body = lambda: body
        fake.do_POST(h)

    return do_POST


def drops_the_tool_call_probe(h) -> None:
    """A fake gateway's POST (fake_models.Gateway) that closes the connection with no answer to the tool-call probe's
    request, the one that offers tools; any other as the fake's."""
    body = fake_models.Gateway._body(h)
    if body.get("tools"):
        h.close_connection = True
        return
    h._body = lambda: body
    fake_models.Gateway.do_POST(h)


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = sandbox.own_home(self)
        self.settings_dir = self.dir / "settings"
        self.settings_dir.mkdir()
        self.work = self.dir / "work"
        self.secrets = self.dir / "secrets"
        self.secrets.mkdir()
        # The image's checks venv, as the image has one.
        self.venv = self.dir / "image-venv"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "python").write_text("")
        # The repository, as GitHub serves it: develop, a/b/rocke, CODEOWNERS.
        self.url = fakerepo.make(self.dir)
        load = RC.load
        for patch in (
            # The container's load, its settings folder the test's; nothing else of the load changes.
            mock.patch.object(
                RC,
                "load",
                lambda home=None, settings_dir=None, config_file=None: load(
                    home, self.settings_dir, config_file
                ),
            ),
            mock.patch.object(doctor, "deployment_image", return_value=True),
            mock.patch.object(container_layout, "WORK_DIR", str(self.work)),
            mock.patch.object(container_layout, "SECRETS_DIR", str(self.secrets)),
            mock.patch.object(container_layout, "IMAGE_VENV", str(self.venv)),
            mock.patch.object(deploy, "clone_url", return_value=self.url),
            # The clone's start date when the state holds none: fakerepo's commits are older than the default's.
            mock.patch.object(deploy, "default_since", return_value=fakerepo.SINCE),
            mock.patch.object(store, "now", return_value=AT),
            mock.patch.dict(
                os.environ,
                {"RKMON_WRAPPER": WRAPPER, "XDG_CONFIG_HOME": str(self.dir / "xdg")},
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.layer()

    def layer(self, **settings) -> None:
        """The delivered layer, as deliver writes it: SETTINGS with LOGIN, and the keys of SETTINGS (the argument) over
        them, each of their keys listed as set in your file."""
        values = dict(
            SETTINGS, integrations={"github_cli": {"username": LOGIN}}, **settings
        )
        layer = {
            "kind": RC.LAYER_KIND,
            "schema_version": RC.LAYER_SCHEMA_VERSION,
            "deploy_schema_version": file_types.DEPLOY.version,
            "package_version": VERSION,
            "confirmed_at": "2026-01-02T01:00:00Z",
            "fingerprint": "sha256:" + "1" * 64,
            "settings": values,
            "set_in_file": set_in(values),
            "container": {},
        }
        new = self.settings_dir / "deploy.json.new"
        new.write_text(json.dumps(layer), encoding="utf-8")
        new.replace(self.settings_dir / RC.LAYER_FILE)

    def rk(self, *argv) -> tuple[int, list[str]]:
        """rk ARGV as the container runs it: its exit and the lines it printed."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(list(argv))
        return code, out.getvalue().splitlines()

    def state(self) -> dict:
        return state_store.read(self.home)

    def where(self) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """The clone, its develop worktree and the Claude workspace's .claude, as the configuration places them."""
        cfg = config.load()
        return (
            pathlib.Path(paths.get(cfg, "repo")),
            pathlib.Path(paths.worktree(cfg, "wtdev")),
            pathlib.Path(cfg["claude_workspace"]) / ".claude",
        )

    def assert_set_up(self, record=RECORD, config_json=False) -> None:
        """The work volume is set up: the clone and its develop worktree, checked out; the agentic files; the
        discoveries and RECORD, the record, in the internal state; no config.json, but with CONFIG_JSON the one adopt
        copied, which the start's one-time move splits."""
        repo, wt, claude = self.where()
        self.assertTrue(str(repo).startswith(str(self.work)), repo)
        self.assertIsNone(deploy.clone_check(config.load()))
        self.assertEqual(
            (wt / ".github" / "CODEOWNERS").read_text(),
            (self.dir / "origin" / ".github" / "CODEOWNERS").read_text(),
        )
        self.assertEqual(
            fakerepo.git(repo, "config", "extensions.worktreeConfig"), "true"
        )
        self.assertTrue(
            (claude / "skills" / agentic.SKILL / "SKILL.md").is_file(), claude
        )
        state = self.state()
        self.assertEqual(state["clone_since"], fakerepo.SINCE)
        self.assertEqual(state["codeowner_teams"], TEAMS)
        self.assertEqual(state["provisioned"], record)
        self.assertEqual(os.path.lexists(self.home / RC.CONFIG_JSON), config_json)


class TestFresh(Base):
    def test_a_fresh_work_volume_is_set_up_from_the_layer(self):
        self.assertFalse(self.work.exists())
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertEqual(
            rows(lines),
            [("PASS", s) for s in STEPS[:-1]] + [("OFF", STEPS[-1])],
            "\n".join(lines),
        )
        self.assertEqual(lines[-1], provision.DONE)
        # On one line: a row breaks where its temporary paths make it.
        text = one_line("\n".join(lines))
        self.assertIn(f"{self.url} cloned into ", text)
        self.assertIn(f"checks' virtual environment: in place: {self.venv}", text)
        self.assertIn("rendered with this configuration into ", text)
        self.assertIn("team-l", text)  # the code owners found
        self.assert_set_up()
        # The checks' virtual environment is the image's: none is made in the working folder.
        self.assertEqual(paths.get(config.load(), "venv"), str(self.venv))

    def test_a_second_run_changes_nothing(self):
        """Every start runs the steps: each in place is left as it is."""
        self.assertEqual(self.rk("provision")[0], 0)
        before = files(self.work, self.home)
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertEqual(rows(lines), IN_PLACE + [("OFF", STEPS[-1])], lines)
        self.assertEqual(lines[-1], provision.DONE)
        for step in (
            "folders",
            "worker tools",
            "clone",
            "checks' virtual environment",
            "agentic files",
        ):
            self.assertTrue(
                any(line.startswith(f"  PASS  {step}: in place") for line in lines),
                (step, lines),
            )
        self.assertIn("in the internal state: 4 code-owner areas", "\n".join(lines))
        self.assertEqual(files(self.work, self.home), before)

    def test_a_config_json_written_once_the_volume_is_set_up_leaves_its_record_alone(
        self,
    ):
        """The work volume is an older install's only when its config.json was there before any record."""
        self.assertEqual(self.rk("provision")[0], 0)
        self.assertFalse(provision.older_install(self.home))
        (self.home / RC.CONFIG_JSON).write_text("{}", encoding="utf-8")
        self.assertEqual(
            self.rk("provision", "--check"), (0, [provision.words(RECORD)])
        )
        self.assertEqual(self.state()["provisioned"], RECORD)
        self.assertFalse(provision.older_install(self.home))

    def test_check_says_whether_the_work_volume_is_provisioned(self):
        self.assertEqual(
            self.rk("provision", "--check"),
            (1, [provision.NOT_YET.format(program=WRAPPER)]),
        )
        self.assertFalse(self.work.exists())  # --check sets nothing up
        self.assertEqual(self.rk("provision")[0], 0)
        self.assertEqual(
            self.rk("provision", "--check"), (0, [provision.words(RECORD)])
        )


class TestCutShort(Base):
    def test_a_run_stopped_at_a_step_is_finished_by_the_next(self):
        with mock.patch.object(deploy, "venv_commands", return_value=[["false"]]):
            code, lines = self.rk("provision")
        self.assertEqual(code, 1, "\n".join(lines))
        self.assertEqual(
            rows(lines),
            [("PASS", s) for s in STEPS[:4]] + [("FAIL", STEPS[4])],
        )
        self.assertEqual(lines[-1].strip(), f"fix: {WRAPPER} start")
        self.assertNotIn("provisioned", self.state())
        self.assertEqual(self.rk("provision", "--check")[0], 1)
        # The next start finishes it: what the first one did is in place.
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        text = one_line("\n".join(lines))
        self.assertIn("PASS clone: in place: ", text)
        self.assertIn("code owners: in the internal state", text)
        self.assert_set_up()

    def test_a_clone_cut_short_is_made_again(self):
        cfg = config.load()
        repo = pathlib.Path(paths.get(cfg, "repo"))
        # The clone made, and its develop worktree added but not yet checked out: a start stopped between the two.
        deploy.layout(cfg)
        deploy.run(
            deploy.clone_commands(dict(cfg, clone_since=fakerepo.SINCE), self.url)
        )
        deploy.run(deploy.wtdev_commands(cfg)[:2])
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertIn(
            f"PASS clone: {repo} in place, back to ", one_line("\n".join(lines))
        )
        self.assert_set_up()

    def test_a_venv_cut_short_is_made_again(self):
        # A venv made in the working folder is in place once its install has finished, which the mark written after
        # pip says, not once its python is there; the image's, which its build makes whole or not at all, by its python.
        cfg = config.load()
        venv = pathlib.Path(paths.get(cfg, "working_dir")) / "venv-gen"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("")  # its pip cut short
        cmds = deploy.venv_commands(cfg, "gen")
        self.assertEqual(cmds[0], ["python3", "-m", "venv", str(venv)])
        self.assertEqual(cmds[-1], ["touch", f"{venv}/{deploy.VENV_MARK}"])
        (venv / deploy.VENV_MARK).write_text("")
        self.assertEqual(deploy.venv_commands(cfg, "gen"), [])
        self.assertFalse((self.venv / deploy.VENV_MARK).exists())
        self.assertEqual(deploy.venv_commands(cfg, "checks"), [])

    def test_a_clone_that_is_not_the_repositorys_is_made_again(self):
        cfg = config.load()
        repo = pathlib.Path(paths.get(cfg, "repo"))
        repo.mkdir(parents=True)
        fakerepo.git(repo, "init", "-q")  # no develop: a clone stopped before its fetch
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertIn(f"{self.url} cloned into {repo}", one_line("\n".join(lines)))
        self.assert_set_up()


class TestRepair(Base):
    """Once the work volume is set up, its record there, each start's rk provision puts back what has gone missing and
    changes nothing else; it removes nothing: a clone or a worktree that is there but not whole stays as it is, a WARN
    whose fix removes it by hand, then the stop and the start.
    """

    # How a fix that removes a piece by hand ends: the stop and the start, which make it again.
    THEN_RESTART = f"then {WRAPPER} stop --drain, then {WRAPPER} start"

    def set_up(self) -> dict:
        """The work volume set up by a first run: what the work volume and the monitor's home hold then (files)."""
        self.assertEqual(self.rk("provision")[0], 0)
        return files(self.work, self.home)

    def kept_run(self, *rows_wanted) -> str:
        """rk provision once the work volume is set up, which exits 0, each step's row as ROWS_WANTED gives it (its
        status and its step), else PASS (OFF for step 7: no local review loop), then DONE: its output in one line.
        """
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        want = {step: status for status, step in IN_PLACE + [("OFF", STEPS[-1])]}
        want.update({step: status for status, step in rows_wanted})
        self.assertEqual(rows(lines), [(s, step) for step, s in want.items()], lines)
        self.assertEqual(lines[-1], provision.DONE)
        return one_line("\n".join(lines))

    def clone_row(self):
        """The doctor's row of the monitor's clone."""
        (row,) = [
            c
            for c in doctor.configuration_checks(config.load(), "loaded")
            if c.name == "the monitor's clone"
        ]
        return row

    def test_a_start_puts_back_a_removed_piece_and_changes_nothing_else(self):
        before = self.set_up()
        cfg = config.load()
        claude = self.where()[2]
        for what, piece in (
            ("a folder of its layout", paths.get(cfg, "probes")),
            ("a worker tool", f"{paths.get(cfg, 'worker_tools')}/mkwt.py"),
            ("the installed skill", claude / "skills" / agentic.SKILL / "SKILL.md"),
            ("the review book", paths.get(cfg, "review_book")),
            ("the agentic files' render", paths.get(cfg, "agentic")),
        ):
            with self.subTest(what):
                piece = pathlib.Path(piece)
                self.assertTrue(piece.exists(), piece)
                if piece.is_dir():
                    shutil.rmtree(piece)
                else:
                    piece.unlink()
                self.kept_run()
                self.assertEqual(files(self.work, self.home), before)

    def test_a_start_puts_back_each_installed_agentic_file_that_is_gone(self):
        """While the render the install's manifest records is this configuration's, each file the install wrote that
        is gone is put back from that render, and nothing else changes: an agent type, the verifier's team member,
        which rk review-split reads, a procedure of the skill, its SKILL.md, a shared policy, and the dispatch table
        with its rows."""
        before = self.set_up()
        claude = self.where()[2]
        for rel in (
            "agents/rocke-review-verifier.md",
            "team_members/rocke_review_verifier.md",
            f"skills/{agentic.SKILL}/review_procedure.md",
            f"skills/{agentic.SKILL}/SKILL.md",
            "skills/shared/prerequisites.md",
            "team_members/dispatch_table.md",
        ):
            with self.subTest(rel):
                (claude / rel).unlink()
                text = self.kept_run()
                self.assertIn(f"agentic files: put back in {claude}: {rel}", text)
                self.assertEqual(files(self.work, self.home), before)

    def test_a_start_that_puts_back_the_skill_leaves_an_edited_file_as_it_is(self):
        """With SKILL.md gone, the render the manifest records still this configuration's, a start puts SKILL.md back
        and installs nothing again: an agent type edited by hand stays as it is, with no backup made.
        """
        self.set_up()
        claude = self.where()[2]
        edited = claude / "agents" / "rocke-code-quality-reviewer.md"
        edited.write_text(
            edited.read_text(encoding="utf-8") + "\nA line the operator added.\n",
            encoding="utf-8",
        )
        before = files(self.work, self.home)
        (claude / "skills" / agentic.SKILL / "SKILL.md").unlink()
        text = self.kept_run()
        self.assertIn(
            f"agentic files: put back in {claude}: skills/{agentic.SKILL}/SKILL.md",
            text,
        )
        self.assertEqual(files(self.work, self.home), before)

    def test_a_start_whose_install_has_no_record_of_its_render_says_so_and_installs_again(
        self,
    ):
        """With the install's manifest gone, nothing records which render the workspace holds: a start renders and
        installs again, its row says why, and the manifest records the render again."""
        self.set_up()
        claude = self.where()[2]
        (claude / "skills" / agentic.SKILL / agentic.INSTALLED).unlink()
        text = self.kept_run()
        self.assertIn(
            "agentic files: the installed skill has no record of its render "
            f"({agentic.INSTALLED} is gone, or an install was cut short): rendered again into ",
            text,
        )
        self.assertNotEqual(agentic.installed_digest(claude.parent), "")

    def test_each_start_rewrites_when_it_finished_and_with_which_package_keeping_older_install(
        self,
    ):
        """The record says when the last run finished and with which package: a start with the record there, later and
        with another package (its image's, the settings delivered for it), rewrites provisioned.at and
        provisioned.package_version, and keeps older_install."""
        self.set_up()
        state_store.set(self.home, provision.OLDER_INSTALL, True)
        later, other = "2026-03-04T05:00:00Z", "9.8.7"
        layer = self.settings_dir / RC.LAYER_FILE
        layer.write_text(
            json.dumps(
                dict(
                    json.loads(layer.read_text(encoding="utf-8")), package_version=other
                )
            ),
            encoding="utf-8",
        )
        with mock.patch.object(store, "now", return_value=later):
            with mock.patch.object(image, "package_version", return_value=other):
                self.kept_run()
        self.assertEqual(
            self.state()["provisioned"],
            {"at": later, "package_version": other, "older_install": True},
        )

    def test_a_start_makes_again_a_removed_clone_or_develop_worktree(self):
        self.set_up()
        repo, wt, _claude = self.where()

        def outside() -> dict:
            """files, but in the clone and the develop worktree, which git writes as it makes them."""
            return {
                p: d
                for p, d in files(self.work, self.home).items()
                if not any(
                    p == str(x) or p.startswith(f"{x}{os.sep}") for x in (repo, wt)
                )
            }

        before = outside()
        for what, gone in (
            ("the develop worktree", (wt,)),
            ("the clone and its develop worktree", (repo, wt)),
        ):
            with self.subTest(what):
                for p in gone:
                    shutil.rmtree(p)
                text = self.kept_run()
                self.assertIn(f"with its develop worktree {wt} made", text)
                self.assert_set_up()
                self.assertEqual(outside(), before)

    def test_the_develop_worktree_is_made_again_under_the_clone_s_lock(self):
        """The clone's worktrees forgotten and the develop worktree made again hold the clone's lock
        (gitops.clone_lock), which the monitor's fetches and stagings take: a run while the monitor uses the clone
        waits for them."""
        self.set_up()
        repo, wt, _claude = self.where()
        shutil.rmtree(wt)
        lock = gitops.clone_lock_path(str(repo))
        run = deploy.run
        held = {}

        def run_seen(cmds, env=None):
            """deploy.run, which notes for each git worktree command whether the clone's lock is held as it runs."""
            for cmd in cmds:
                if "worktree" in cmd:
                    at = cmd.index("worktree")
                    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o664)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held[" ".join(cmd[at : at + 2])] = False
                    except BlockingIOError:
                        held[" ".join(cmd[at : at + 2])] = True
                    finally:
                        os.close(fd)
            return run(cmds, env)

        with mock.patch.object(deploy, "run", run_seen):
            self.kept_run()
        self.assertEqual(held, {"worktree prune": True, "worktree add": True})
        self.assert_set_up()

    def test_a_part_made_clone_or_worktree_is_left_as_it_is_with_the_fix_that_removes_it(
        self,
    ):
        """A clone with no develop (as one a run cut short before its fetch leaves it), or a develop worktree whose
        checkout did not finish: the start leaves it as it is, nothing changed, its row a WARN whose fix removes it by
        hand, then the stop and the start, which make it again. The doctor's clone row gives the same fix.
        """
        self.set_up()
        repo, wt, _claude = self.where()
        fakerepo.git(repo, "update-ref", "-d", "refs/remotes/origin/develop")
        before = files(self.work, self.home)
        text = self.kept_run(("WARN", "clone"))
        working = paths.get(config.load(), "working_dir")
        fix = f"in {WRAPPER} shell, remove the clone {repo} and its worktrees ({working}/wt*), {self.THEN_RESTART}"
        self.assertIn(
            f"WARN clone: {repo}: it has no develop (refs/remotes/origin/develop); {provision.REMOVES_NOTHING} "
            f"fix: {fix}",
            text,
        )
        self.assertEqual(files(self.work, self.home), before)
        row = self.clone_row()
        self.assertEqual((row.status, one_line(str(row.fix))), ("WARN", fix))
        # The fix, then the start: the clone and its develop worktree made again.
        shutil.rmtree(repo)
        shutil.rmtree(wt)
        self.kept_run()
        self.assert_set_up()
        # A develop worktree whose checkout did not finish: an empty index, as git worktree add --no-checkout leaves it.
        fakerepo.git(wt, "read-tree", "--empty")
        before = files(self.work, self.home)
        text = self.kept_run(("WARN", "clone"))
        self.assertIn(
            f"WARN clone: {repo} in place, back to {fakerepo.SINCE}, but the develop worktree {wt} there is not a "
            f"checkout of it; {provision.REMOVES_NOTHING} fix: in {WRAPPER} shell, remove the worktree {wt}, "
            f"{self.THEN_RESTART}",
            text,
        )
        self.assertEqual(files(self.work, self.home), before)
        shutil.rmtree(wt)
        self.kept_run()
        self.assert_set_up()

    def test_a_missing_clone_s_doctor_fix_is_the_stop_and_the_start(self):
        """The doctor's clone row while the clone is missing: the stop and the start, which clone it."""
        self.set_up()
        repo, wt, _claude = self.where()
        shutil.rmtree(repo)
        row = self.clone_row()
        self.assertEqual(
            (row.status, str(row.fix)),
            ("WARN", f"{WRAPPER} stop --drain, then {WRAPPER} start clones it"),
        )


class TestFixes(Base):
    # A failure one of your settings causes names that setting in your deploy settings file, then confirm and apply
    # (doctor.apply_commands); any other names the start alone (TestCutShort).
    def test_a_repository_git_does_not_find_names_its_setting(self):
        nodevelop = self.dir / "nodevelop"
        nodevelop.mkdir()
        fakerepo.git(nodevelop, "init", "-q", "-b", "main")
        (nodevelop / "f.txt").write_text("f\n")
        fakerepo.git(nodevelop, "add", "-A")
        fakerepo.git(nodevelop, "commit", "-qm", "main: first")
        for url in (str(self.dir / "nothere"), nodevelop.resolve().as_uri()):
            with self.subTest(url=url):
                with mock.patch.object(deploy, "clone_url", return_value=url):
                    code, lines = self.rk("provision")
                self.assertEqual(code, 1, "\n".join(lines))
                self.assertEqual(rows(lines)[-1], ("FAIL", "clone"), lines)
                text = one_line("\n".join(lines))
                self.assertIn(
                    "git finds no such repository, or one with no develop branch: ",
                    text,
                )
                self.assertIn(
                    "fix: set repository.name in your deploy settings file to the repository the monitor watches",
                    text,
                )
                self.assertTrue(
                    text.endswith(f", then {WRAPPER} confirm and {WRAPPER} apply"),
                    text,
                )
                self.assertNotIn("provisioned", self.state())
        # Once the settings name the repository, the start sets the work volume up.
        self.assertEqual(self.rk("provision")[0], 0)
        self.assert_set_up()

    def test_agentic_files_with_no_operator_login_name_its_setting(self):
        cfg = config.load()
        cfg["operator"] = dict(cfg["operator"], github_login="")
        out = []
        self.assertFalse(provision.run(cfg, self.home, out.append))
        lines = "\n".join(out).splitlines()
        self.assertEqual(rows(lines)[-1], ("FAIL", "agentic files"), lines)
        self.assertEqual(
            one_line("\n".join(lines)).rsplit("fix: ", 1)[1],
            f"set integrations.github_cli.username in your deploy settings file to your GitHub username, then "
            f"{WRAPPER} confirm and {WRAPPER} apply",
        )


class TestOlderVolume(Base):
    # What a work volume an older package set up holds in the monitor's home: its config.json, or, once the one-time
    # move to this layout has run, config.json.migrated, the name that move gives it.
    OLDER = ("config.json", "config.json.migrated")

    def test_a_work_volume_an_older_package_set_up_counts_and_gets_its_record(self):
        """Its record says the work volume is an older install's (provision.OLDER_INSTALL): the first read decides it,
        before which the answer is what that read would record, and nothing is written.
        """
        older = dict(RECORD, older_install=True)
        for name in self.OLDER:
            with self.subTest(name=name):
                state_store.path(self.home).unlink(missing_ok=True)
                for other in self.OLDER:
                    (self.home / other).unlink(missing_ok=True)
                self.assertFalse(provision.older_install(self.home))
                (self.home / name).write_text("{}", encoding="utf-8")
                self.assertTrue(provision.older_install(self.home))
                self.assertFalse(state_store.path(self.home).exists())
                self.assertEqual(
                    self.rk("provision", "--check"), (0, [provision.words(RECORD)])
                )
                self.assertEqual(self.state()["provisioned"], older)
                # Its record, once written, is the one read: nothing is set up.
                self.assertEqual(provision.provisioned(self.home), older)
                self.assertTrue(provision.older_install(self.home))
                self.assertFalse(self.work.exists())


class TestAdoptThenStart(Base):
    def earlier(self, config_json: bool = False) -> pathlib.Path:
        """A copy of an earlier deployment's /work, with a PR in flight, staged as rkmon adopt stages it, before the
        first start: the stage. With CONFIG_JSON, its config.json too."""
        (self.settings_dir / RC.LAYER_FILE).unlink()  # before the first start
        earlier = self.dir / "earlier-work"
        (earlier / "monitor" / "state").mkdir(parents=True)
        (earlier / "monitor" / "state" / "state.json").write_text(
            json.dumps({"schema": store.SCHEMA, "prs": {"101": {"status": "open"}}}),
            encoding="utf-8",
        )
        if config_json:
            (earlier / "monitor" / RC.CONFIG_JSON).write_text(
                json.dumps({"repo": "acme/widgets"}), encoding="utf-8"
            )
        stage = self.dir / "stage"
        adopt.stage(adopt.old_folders(earlier, None), stage)
        return stage

    def test_an_adopted_config_json_is_the_moves_and_the_start_sets_the_volume_up(
        self,
    ):
        """The earlier monitor's config.json comes with its state: apply puts it in the monitor's home and
        makes the move due again, which answers that the work volume is an older install's, but the volume is not set
        up for it: an adopted work volume is set up by the next start's rk provision, whatever config.json adopt
        copied, and its record keeps the adoption and that answer."""
        stage = self.earlier(config_json=True)
        code, lines = self.rk("adopt", "apply", "--stage", str(stage))
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue(lines[-2].startswith("the one-time move is due again: "), lines)
        self.assertTrue(provision.older_install(self.home))
        self.assertEqual(
            self.state()["standby"], {"on": True, "reason": adopt.REASON, "since": AT}
        )
        self.layer()
        self.assertEqual(self.rk("provision", "--check")[0], 1)
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assert_set_up(
            record=dict(RECORD, adopted_at=AT, older_install=True), config_json=True
        )

    def test_an_apply_that_cannot_make_the_move_due_again_says_what_it_did(self):
        """rk adopt apply, once the earlier monitor's state is in place, cannot make the move due again: it says what
        it did, then ERROR, which says the state is in place, in standby, with the fix (exit 1).
        """
        stage = self.earlier(config_json=True)
        real = state_store.write_file
        internal = state_store.path(self.home)

        def write_file(file, data):
            if pathlib.Path(file) == internal:
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            code, lines = self.rk("adopt", "apply", "--stage", str(stage))
        self.assertEqual(code, 1, "\n".join(lines))
        text = one_line("\n".join(lines))
        folder = internal.parent
        again = steps.command_line("adopt --from DIR")
        self.assertIn(
            f"{self.home / RC.CONFIG_JSON}: the earlier monitor's, 0 old path(s) rewritten in it ",
            text,
        )
        self.assertIn(
            f"this deployment's own copies are in {self.work}/adopt-backup-", text
        )
        self.assertTrue(
            text.endswith(
                f"ERROR: the one-time move cannot be made due again in {folder}: Permission denied; the earlier "
                f"monitor's state is in place, in standby fix: make {folder} writable for the monitor, then run "
                f"{again} again"
            ),
            text,
        )
        self.assertNotIn("APPLIED", text)
        self.assertEqual(self.state()["standby"]["on"], True)

    def test_an_apply_an_oserror_stops_says_what_it_did_then_error_with_its_fix(self):
        """rk adopt apply that an OSError stops, never a traceback: here config.json cannot be written in the
        monitor's home, once the earlier monitor's state is in place. It says what it did, then ERROR, which says that
        the state is in place, in standby, with the fix (exit 1)."""
        stage = self.earlier(config_json=True)
        here = self.home / RC.CONFIG_JSON
        write_text = pathlib.Path.write_text

        def denied(path, *args, **kwargs):
            if path == here:
                raise PermissionError(
                    errno.EACCES, os.strerror(errno.EACCES), str(path)
                )
            return write_text(path, *args, **kwargs)

        with mock.patch.object(
            pathlib.Path, "write_text", autospec=True, side_effect=denied
        ):
            code, lines = self.rk("adopt", "apply", "--stage", str(stage))
        self.assertEqual(code, 1, "\n".join(lines))
        text = one_line("\n".join(lines))
        again = steps.command_line("adopt --from DIR")
        self.assertIn(
            f"this deployment's own copies are in {self.work}/adopt-backup-", text
        )
        self.assertTrue(
            text.endswith(
                f"ERROR: adopt stopped at {here}: {os.strerror(errno.EACCES)}; the earlier monitor's state is in "
                f"place, in standby fix: make {self.home} writable for the monitor, then run {again} again"
            ),
            text,
        )
        self.assertNotIn("APPLIED", text)
        self.assertEqual(self.state()["standby"]["on"], True)

    @unittest.skipIf(os.geteuid() == 0, "root writes in a folder that is read-only")
    def test_an_apply_whose_mark_cannot_be_written_is_refused_with_nothing_changed(
        self,
    ):
        """rk adopt apply writes the mark of the adoption in the monitor's home before it moves anything: a home that
        cannot take it refuses the apply, nothing changed and nothing said done, with the fix (exit 4).
        """
        stage = self.earlier(config_json=True)
        # the plan first, as docker/rkmon adopt runs it: its run makes the state folder in the home
        self.assertEqual(self.rk("adopt", "plan", "--stage", str(stage))[0], 0)
        mode = stat.S_IMODE(self.home.stat().st_mode)
        self.home.chmod(0o555)
        self.addCleanup(self.home.chmod, mode)
        code, lines = self.rk("adopt", "apply", "--stage", str(stage))
        self.assertEqual(code, 4, "\n".join(lines))
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            one_line("\n".join(lines)),
            f"REFUSED: adopt stopped at {adopt.mark_file(self.home)}: {os.strerror(errno.EACCES)}; nothing changed "
            f"fix: make {self.home} writable for the monitor, then run {again} again",
        )
        self.assertFalse(adopt.marked(self.home))
        self.assertEqual(list(self.work.glob("adopt-backup-*")), [])

    @unittest.skipIf(os.geteuid() == 0, "root reads every file")
    def test_a_plan_that_cannot_read_the_stage_says_so_with_nothing_changed(self):
        """rk adopt plan that cannot read a staged copy, never a traceback: REFUSED, nothing changed, with the fix,
        adopt again, which stages it afresh (exit 4)."""
        stage = self.earlier()
        staged = stage / "state" / "state.json"
        staged.chmod(0)
        self.addCleanup(staged.chmod, 0o644)
        code, lines = self.rk("adopt", "plan", "--stage", str(stage))
        self.assertEqual(code, 4, "\n".join(lines))
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            one_line("\n".join(lines)),
            f"REFUSED: the staged copy {staged} cannot be read: {os.strerror(errno.EACCES)}; nothing changed fix: "
            f"run {again} again",
        )

    def test_the_start_after_adopt_sets_the_work_volume_up_and_keeps_the_adopted_state(
        self,
    ):
        """rkmon adopt before the first start, then rkmon start: adopt runs with no configuration in the work
        volume and no settings delivered; the start then delivers them, and its rk provision sets the work volume up
        around the adopted state, which its rk init keeps, in standby with adopt's reason. The earlier work
        volume's setup record stays behind, and the adoption is recorded beside the record.
        """
        (self.settings_dir / RC.LAYER_FILE).unlink()  # before the first start
        # A copy of an earlier deployment's /work: a PR in flight, and its internal state with what it learned and
        # its own work volume's record.
        earlier = self.dir / "earlier-work"
        (earlier / "monitor" / "state").mkdir(parents=True)
        (earlier / "monitor" / "state" / "state.json").write_text(
            json.dumps(
                {
                    "schema": store.SCHEMA,
                    "prs": {"101": {"status": "open"}},
                    "review_pipes": [{"pr": 101}],
                }
            ),
            encoding="utf-8",
        )
        names = ["acme-review-bot"]
        for name, value in (
            ("reviews.automated_reviewer_names_used", names),
            (provision.PACKAGE_VERSION, VERSION),
            (provision.AT, "2025-12-01T00:00:00Z"),
        ):
            state_store.set(earlier / "monitor", name, value)
        stage = self.dir / "stage"
        adopt.stage(adopt.old_folders(earlier, None), stage)
        for action in ("plan", "apply"):
            code, lines = self.rk("adopt", action, "--stage", str(stage))
            self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue(lines[-1].startswith("APPLIED: "), lines)
        self.assertEqual(
            self.state()["reviews"]["automated_reviewer_names_used"], names
        )
        standby = {"on": True, "reason": adopt.REASON, "since": AT}
        self.assertEqual(self.state()["standby"], standby)
        self.assertEqual(self.rk("provision", "--check")[0], 1)
        # The start: the settings delivered, then rk provision and rk init --if-new --on.
        self.layer()
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assert_set_up(record=dict(RECORD, adopted_at=AT))
        self.assertEqual(self.state()["standby"], standby)
        code, lines = self.rk("init", "--if-new", "--on")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertIn(
            f"state exists: kept (`{doctor.rk_command('init', '--force')}` re-seeds it)",
            lines,
        )
        st = json.loads((config.state_dir() / "state.json").read_text(encoding="utf-8"))
        self.assertEqual((list(st["prs"]), st["review_pipes"]), (["101"], []))
        self.assertEqual(
            self.state()["reviews"]["automated_reviewer_names_used"], names
        )


class TestWorkerServer(Base):
    """Step 7, the local review loop's server: its model checked from the container, the test chat and the tool-call
    probe, on an OpenAI-compatible gateway or an Ollama server; what the checks find recorded in the internal state,
    replaced or cleared for another binding; and each start's probe again while the last probe is of another address
    or model, of another key or way of sending it, or did not pass."""

    KEY_SECRET = "example_api_key"

    def binding(self, url: str, model: str = fake_models.GOOD, **sent) -> None:
        """A layer whose local worker is MODEL on the gateway at URL, its key in the secret KEY_SECRET, sent in the
        gateway's own header; SENT, the binding's keys that say how it is sent, over those (None: the key left out).
        """
        worker = {
            "provider": "openai_compatible",
            "model_id": model,
            "base_url": url,
            "api_key_secret_name": self.KEY_SECRET,
            "api_key_sent_as": "header",
            "api_key_header": "X-Example-Key",
            **sent,
        }
        self.layer(
            model_bindings={
                "worker": {k: v for k, v in worker.items() if v is not None}
            }
        )

    def key_file(self) -> pathlib.Path:
        key = pathlib.Path(config.worker(config.load())["api_key_file"])
        self.assertTrue(str(key).startswith(str(self.secrets)), key)
        key.write_text("example-key-1\n", encoding="utf-8")
        return key

    def gateway(self, name: str = "gateway", **settings) -> fake_models.Server:
        """A fake gateway named NAME (SETTINGS: its handler's) that takes the key of KEY_SECRET, until the test ends."""
        self.binding("http://127.0.0.1:1/gw/v2")
        srv = fake_models.start(
            self.dir / "fake",
            "gateway",
            name=name,
            key_file=str(self.key_file()),
            **settings,
        )
        self.addCleanup(srv.stop)
        return srv

    def ollama(self, model: str, srv=None) -> fake_models.Server:
        """A layer whose local worker is MODEL on SRV, a fake Ollama server: without one, a server that runs until the
        test ends."""
        if srv is None:
            srv = fake_models.start(self.dir / "fake", "ollama")
            self.addCleanup(srv.stop)
        self.layer(
            model_bindings={
                "worker": {
                    "provider": "ollama",
                    "model_id": model,
                    "server_url": srv.url,
                }
            }
        )
        return srv

    def provision(self, *rows_wanted) -> str:
        """rk provision, which exits 0 with ROWS_WANTED among its rows: its output in one line."""
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        for row in rows_wanted:
            self.assertIn(row, rows(lines), lines)
        return one_line("\n".join(lines))

    def not_probed(self, url: str, model: str, fake: str = "gateway") -> None:
        """rk provision once the last probe of MODEL at URL passed both: every step in place, step 7's too, with no
        line before its checks and no request to the fake server FAKE."""
        asked = len(fake_models.read_log(self.dir / "fake", fake))
        code, lines = self.rk("provision")
        self.assertEqual(
            (code, rows(lines)), (0, IN_PLACE + [("PASS", STEPS[-1])]), lines
        )
        text = one_line("\n".join(lines))
        self.assertIn(
            f"{STEPS[-1]}: in place: {config.safe_url(url)}: {model} passed the test chat and the tool-call probe at "
            "its last probe",
            text,
        )
        self.assertNotIn(f"{STEPS[-1]}: checking", text)
        self.assertEqual(len(fake_models.read_log(self.dir / "fake", fake)), asked)

    def test_each_settings_fix_of_the_start_ends_as_doctor_apply_commands_ends_it(self):
        """A failed step's settings fix and a refused model's fix end with doctor.apply_commands, the one home of
        `<wrapper> confirm` and `<wrapper> apply`: the start after them is apply's to name.
        """
        from rkmon_lib.checks import Command

        ending = (Command(("the-ending-apply-commands-gives",)),)
        srv = self.gateway()
        self.binding(srv.url, ABSENT)
        with mock.patch.object(doctor, "apply_commands", return_value=ending):
            fix = str(provision._settings_fix("repo"))
            text = self.provision(("WARN", "local review loop's server"))
        self.assertTrue(fix.endswith(", then the-ending-apply-commands-gives"), fix)
        self.assertIn(
            "set model_bindings.worker.model_id in your deploy settings file to a model it serves, then "
            f"the-ending-apply-commands-gives {provision.DONE}",
            text,
        )

    def test_a_model_that_answers_in_text_is_recorded_for_probe_failure_to_refuse(self):
        """The start checks the model with the tool-call probe too: a model that answers it in text is a WARN that
        holds nothing up, recorded as it is, and config.probe_failure, the one judge, refuses it for its role: the row
        says the doctor fails on the record, its fix a model that can call tools in the binding, as the doctor's
        reviewer row gives it. Each start probes it again while it fails.
        """
        srv = self.gateway()
        self.binding(srv.url, fake_models.TEXT_ONLY)
        text = self.provision(("WARN", "local review loop's server"))
        self.assertIn(
            f"{fake_models.TEXT_ONLY} answers the test chat, but not the tool-call probe: it answered in text",
            text,
        )
        self.assertIn(
            f"so the reviews refuse {fake_models.TEXT_ONLY} until a probe passes, and {WRAPPER} doctor fails on "
            f"this record {TOOLS}",
            text,
        )
        with mock.patch.object(doctor, "worker_running", return_value=False):
            row = doctor.model_check(config.load())
        self.assertEqual(
            (row.status, f"fix: {one_line(str(row.fix))}"), ("FAIL", TOOLS)
        )
        self.assertEqual(self.state()["provisioned"], RECORD)
        w = config.worker(config.load())
        self.assertEqual(w["probe"]["chat"], "ok")
        self.assertTrue(
            w["probe"]["tools"].startswith("it answered in text"), w["probe"]
        )
        self.assertTrue(
            config.probe_failure(w).startswith(
                "failed the tool-call probe: it answered in text"
            ),
            w["probe"],
        )
        asked = len(fake_models.read_log(self.dir / "fake", "gateway"))
        code, lines = self.rk("provision")
        self.assertEqual(
            (code, rows(lines)),
            (0, IN_PLACE + [("WARN", "local review loop's server")]),
        )
        self.assertGreater(
            len(fake_models.read_log(self.dir / "fake", "gateway")), asked
        )

    def test_the_tool_call_probe_sends_what_the_reviews_send(self):
        """The start's checks send the settings the reviews send (config.model_for(cfg, "draft"), which every review
        gets as RKMON_WORKER_JSON): the tool-call probe asks for their max output, and an Ollama server for their
        context too, whatever the server reports: the gateway's model-info route, never read, gives a smaller max
        output, and Ollama a smaller context length."""
        for kind in ("gateway", "ollama"):
            with self.subTest(kind):
                seen = []
                fake = fake_models.Gateway if kind == "gateway" else fake_models.Ollama
                if kind == "gateway":
                    srv = self.gateway(do_POST=kept(seen, fake))
                    self.binding(srv.url)
                else:
                    srv = fake_models.start(
                        self.dir / "fake", kind, do_POST=kept(seen, fake)
                    )
                    self.addCleanup(srv.stop)
                    self.ollama(fake_models.OLLAMA_GOOD, srv)
                self.provision(("PASS", "local review loop's server"))
                draft = config.model_for(config.load(), "draft")
                (probe,) = [body for body in seen if body.get("tools")]
                if kind == "gateway":
                    self.assertEqual(probe["max_tokens"], draft["max_output_tokens"])
                    paths = [
                        r["path"] for r in fake_models.read_log(self.dir / "fake", kind)
                    ]
                    self.assertEqual([p for p in paths if "/model/info" in p], [])
                else:
                    self.assertEqual(
                        (probe["options"]["num_ctx"], probe["options"]["num_predict"]),
                        (draft["context"], draft["max_output_tokens"]),
                    )

    def test_an_address_with_a_user_and_a_password_is_shown_without_them(self):
        """Step 7 prints each address as config.safe_url shows it: a base address with a user and a password in it
        (the binding takes one) shows neither, nor does where its model list answered, which is built on it.
        """
        srv = self.gateway()
        self.binding(srv.url.replace("://", "://someone:pw-example@", 1))
        text = self.provision(("PASS", "local review loop's server"))
        self.assertIn(f"{srv.url}/v1/models", text)
        self.assertNotIn("pw-example", text)
        self.assertNotIn("someone", text)

    def test_a_line_before_the_checks_says_what_they_are_and_shows_before_they_start(
        self,
    ):
        """The checks can take about two minutes: a line before them says what they are, of which model, where, and
        stdout is flushed after it, so that it shows before they start when stdout is a pipe (rkmon start runs rk
        provision through `docker compose run -T` when it is not on a terminal)."""
        srv = self.gateway()
        self.binding(srv.url)
        flushed, shown = [], []

        class Pipe(io.StringIO):
            def flush(self):
                flushed.append(self.getvalue())

        check = models.check

        def checked(s, *args, **kwargs):
            shown.append(flushed[-1] if flushed else "")
            return check(s, *args, **kwargs)

        pipe = Pipe()
        with mock.patch.object(models, "check", checked):
            with contextlib.redirect_stdout(pipe):
                self.assertEqual(rkmon.main(["provision"]), 0)
        self.assertEqual(
            shown[0].rstrip("\n").rsplit("\n", 1)[-1],
            f"local review loop's server: checking {fake_models.GOOD} at {srv.url} from the container: its model "
            "list, a test chat and a tool-call probe, which can take about two minutes",
        )
        self.assertIn(
            ("PASS", "local review loop's server"), rows(pipe.getvalue().splitlines())
        )

    def test_a_test_chat_the_server_answers_with_an_error_refuses_the_model(self):
        """A test chat that gets an HTTP answer is judged: the start records it as it is, so config.probe_failure, the
        one judge, refuses the model for its role and rk doctor fails, while the start goes on (a WARN). Its fix puts
        that failure right, the same in step 7's row and in the doctor's reviewer row (doctor.probe_fix): a key the
        gateway refuses, sent as a bearer token while it wants its own header or in a header it does not read (HTTP
        401), or forbidden (HTTP 403): the key set again, then the stop and the start, or its header set, and how the
        key is sent while it goes as a bearer token alone, then confirm and apply; a route the gateway does not find
        (HTTP 404): the binding put right; a server error (HTTP 500): the server put right, then the stop and the start.
        The gateway's model list answers without the key.
        """
        key = (
            f"fix: set the key again: {WRAPPER} secret {self.KEY_SECRET}, and give one line of text, then {WRAPPER} "
            f"stop --drain, then {WRAPPER} start; or the header that carries it: set "
            "model_bindings.worker.api_key_header in your deploy settings file to the header the endpoint reads"
        )
        apply = f", then {WRAPPER} confirm and {WRAPPER} apply"
        unauthorized = "the test prompt got HTTP 401 (unauthorized)"
        for label, sent, post, said, fix in (
            (
                "bearer",
                {"api_key_sent_as": "bearer", "api_key_header": None},
                None,
                unauthorized,
                key + SENT_AS + apply,
            ),
            (
                "header",
                {"api_key_header": "X-Other-Key"},
                None,
                unauthorized,
                key + apply,
            ),
            (
                "forbidden",
                {},
                chat_answers(403),
                "the test prompt got HTTP 403 (forbidden)",
                key + apply,
            ),
            ("no route", {}, chat_answers(404), "the test prompt got HTTP 404", REBIND),
            ("error", {}, chat_answers(500), "the test prompt got HTTP 500", SERVER),
        ):
            with self.subTest(label):
                answers = {"do_POST": post} if post else {}
                srv = self.gateway(label, do_GET=list_open, **answers)
                self.binding(srv.url, **sent)
                text = self.provision(("WARN", "local review loop's server"))
                self.assertIn(f"{said}", text)
                self.assertIn(
                    f"; so the reviews refuse {fake_models.GOOD} until a probe passes, and {WRAPPER} doctor fails on "
                    f"this record {fix} {provision.DONE}",
                    text,
                )
                self.assertNotIn("example-key-1", text)
                cfg = config.load()
                w = config.worker(cfg)
                self.assertTrue(w["probe"]["chat"].startswith(said), w["probe"])
                self.assertTrue(
                    config.probe_failure(w).startswith(f"failed the test chat: {said}"),
                    w["probe"],
                )
                with mock.patch.object(doctor, "worker_running", return_value=False):
                    row = doctor.model_check(cfg)
                self.assertEqual(row.status, "FAIL", row)
                self.assertIn(f"failed the test chat: {said}", row.found)
                self.assertEqual(f"fix: {one_line(str(row.fix))}", fix)

    def test_each_side_of_a_refused_key_s_fix_followed_as_it_says_passes_the_next_start(
        self,
    ):
        """A key the gateway refuses (HTTP 401): each side of its fix (doctor.probe_fix), followed as it says, finds the
        next start's row PASS. The key set again by its secret's command, then the stop and the start; or, while the
        binding sends the key as a bearer token alone (its default), which goes in no other header, the header that
        carries it and how the key is sent, then confirm and apply."""
        self.binding("http://127.0.0.1:1/gw/v2")
        secret = self.key_file()
        # The key of the first gateway, not the one the secret holds.
        wanted = self.dir / "gateway_key"
        wanted.write_text("example-key-2\n", encoding="utf-8")
        # The value of each edit that only the operator knows: the header the gateway reads.
        known = {"model_bindings.worker.api_key_header": fake_models.Gateway.header}
        for label, key_file, sent, side, typed in (
            (
                "the key set again",
                wanted,
                {},
                0,
                [("secret", self.KEY_SECRET), ("stop", "--drain"), ("start",)],
            ),
            (
                "the header and how the key is sent",
                secret,
                {"api_key_sent_as": "bearer", "api_key_header": None},
                1,
                [("confirm",), ("apply",)],
            ),
        ):
            with self.subTest(label):
                srv = fake_models.start(
                    self.dir / "fake",
                    "gateway",
                    name=f"side{side}",
                    key_file=str(key_file),
                    do_GET=list_open,
                )
                self.addCleanup(srv.stop)
                self.binding(srv.url, **sent)
                text = self.provision(("WARN", "local review loop's server"))
                self.assertIn("the test prompt got HTTP 401 (unauthorized)", text)
                cfg = config.load()
                fix = doctor.probe_fix(cfg, config.worker(cfg))
                self.assertIn(f"fix: {one_line(str(fix))} {provision.DONE}", text)
                cut = fix.parts.index("; or the header that carries it:")
                done = [
                    p
                    for p in (fix.parts[:cut], fix.parts[cut + 1 :])[side]
                    if not isinstance(p, str)
                ]
                commands = [p.argv for p in done if isinstance(p, checks.Command)]
                self.assertEqual(commands, [(WRAPPER, *c) for c in typed])
                if side == 0:
                    entry = secrets_registry.entry(cfg, commands[0][2])
                    pathlib.Path(secrets_registry.path(cfg, entry)).write_text(
                        wanted.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                edits = {
                    p.key.removeprefix("model_bindings.worker."): (
                        known[p.key]
                        if isinstance(p.value, checks.Described)
                        else p.value
                    )
                    for p in done
                    if isinstance(p, checks.Edit)
                }
                self.binding(srv.url, **dict(sent, **edits))
                self.provision(("PASS", "local review loop's server"))

    def test_a_model_the_server_does_not_list_is_refused_its_fix_the_server_s_or_the_binding_s(
        self,
    ):
        """A model the gateway's model list does not name: the test chat is not sent, and the model is refused. Its fix
        gives each side its own ending (doctor.not_served_fix): the model served there, then the stop and the start,
        whose probe asks again; or a model the server serves in the binding, then confirm and apply. The doctor's
        reviewer row, which reads the server's model list itself, gives the same fix."""
        srv = self.gateway()
        self.binding(srv.url, ABSENT)
        text = self.provision(("WARN", "local review loop's server"))
        self.assertIn(f"{srv.url}: the endpoint does not list {ABSENT}", text)
        self.assertIn(f"this record {NOT_SERVED} {provision.DONE}", text)
        with mock.patch.object(doctor, "worker_running", return_value=False):
            row = doctor.model_check(config.load())
        self.assertEqual(
            (row.status, row.found, f"fix: {one_line(str(row.fix))}"),
            ("FAIL", f"{srv.url} answers but has no {ABSENT}", NOT_SERVED),
        )

    def test_the_doctor_s_row_of_a_server_that_does_not_answer_ends_each_side_its_own_way(
        self,
    ):
        """The doctor's reviewer row while the server gives no answer: the server started, then the stop and the
        start, whose probe asks it again; or its address in the binding, then confirm and apply.
        """
        base = f"http://127.0.0.1:{sandbox.closed_port(self)}/v1"
        self.binding(base)
        self.key_file()
        with mock.patch.object(doctor, "worker_running", return_value=False):
            row = doctor.model_check(config.load())
        self.assertEqual(row.status, "FAIL", row)
        self.assertIn(f"at {base}: no answer (", row.found)
        self.assertEqual(
            one_line(str(row.fix)),
            f"start the model server, then {WRAPPER} stop --drain, then {WRAPPER} start; or set "
            "model_bindings.worker.base_url in your deploy settings file to the model server's address, then "
            f"{WRAPPER} confirm and {WRAPPER} apply",
        )

    def test_a_key_refused_with_no_secret_named_is_fixed_in_the_binding(self):
        """A key the endpoint refuses (HTTP 401) while the binding names no secret for it: the fix is the binding's
        key secret and header settings, and how the key is sent while it goes as a bearer token alone, then confirm and
        apply (doctor.probe_fix)."""
        cfg = config.load()
        named = (
            "set model_bindings.worker.api_key_secret_name in your deploy settings file to the secret the key is kept "
            "in, and set model_bindings.worker.api_key_header in your deploy settings file to the header the "
            "endpoint reads"
        )
        apply = f", then {WRAPPER} confirm and {WRAPPER} apply"
        for mode, fix in (
            ("header", named + apply),
            ("bearer", named + SENT_AS + apply),
        ):
            with self.subTest(mode):
                w = dict(
                    config.worker(cfg),
                    api_key_secret="",
                    auth_mode=mode,
                    probe={
                        "chat": "the test prompt got HTTP 401 (unauthorized): refused",
                        "tools": "",
                    },
                )
                self.assertEqual(one_line(str(doctor.probe_fix(cfg, w))), fix)

    def test_a_tool_call_probe_a_rate_limit_stopped_refuses_nothing_and_the_next_start_probes_again(
        self,
    ):
        """An HTTP 429 that outlasts the tool-call probe's waits leaves it not checked: a WARN, and no refusal, whose
        fix is the stop and the start again, never a model that can call tools; the doctor's reviewer row passes. The
        next start probes the server again, and a probe that passes both is the last."""
        # The second chat request, the tool-call probe's, gets a 429 with a wait longer than the probe waits.
        srv = self.gateway(fail429=(2,), retry_after="600")
        self.binding(srv.url)
        text = self.provision(("WARN", "local review loop's server"))
        self.assertIn(
            f"that refuses no model, and {WRAPPER} doctor reads this record; the next start probes it again "
            f"{RESTART}",
            text,
        )
        self.assertNotIn(TOOLS, text)
        w = config.worker(config.load())
        self.assertEqual(w["probe"]["chat"], "ok")
        self.assertTrue(w["probe"]["tools"].startswith("not checked: "), w["probe"])
        self.assertEqual(config.probe_failure(w), "")
        with mock.patch.object(doctor, "worker_running", return_value=False):
            self.assertEqual(doctor.model_check(config.load()).status, "PASS")
        code, lines = self.rk("provision")
        self.assertEqual(
            (code, rows(lines)),
            (0, IN_PLACE + [("PASS", "local review loop's server")]),
        )
        self.assertEqual(config.worker(config.load())["probe"]["tools"], "ok")
        self.not_probed(srv.url, fake_models.GOOD)

    def test_a_tool_call_probe_with_no_answer_refuses_nothing_and_the_next_start_probes_again(
        self,
    ):
        """A tool-call probe whose request gets no HTTP answer (here the connection is closed with none) judges
        nothing, as a test chat with none does: it is recorded as not checked, which refuses no model, a WARN whose fix
        is the stop and the start again, never a model that can call tools; the doctor's reviewer row passes. The next
        start probes the server again."""
        srv = self.gateway(do_POST=drops_the_tool_call_probe)
        self.binding(srv.url)
        text = self.provision(("WARN", "local review loop's server"))
        self.assertIn(
            f"{fake_models.GOOD} answers the test chat, but not the tool-call probe: not checked: no answer: "
            "RemoteDisconnected: ",
            text,
        )
        self.assertIn(
            f"that refuses no model, and {WRAPPER} doctor reads this record; the next start probes it again "
            f"{RESTART} {provision.DONE}",
            text,
        )
        self.assertNotIn(TOOLS, text)
        w = config.worker(config.load())
        self.assertEqual(w["probe"]["chat"], "ok")
        self.assertEqual(config.probe_failure(w), "")
        with mock.patch.object(doctor, "worker_running", return_value=False):
            self.assertEqual(doctor.model_check(config.load()).status, "PASS")
        asked = len(fake_models.read_log(self.dir / "fake", "gateway"))
        self.provision(("WARN", "local review loop's server"))
        self.assertGreater(
            len(fake_models.read_log(self.dir / "fake", "gateway")), asked
        )

    def test_only_a_request_with_no_http_answer_leaves_the_tool_call_probe_not_checked(
        self,
    ):
        """The tool-call probe (reviewer_client.probe_tools) says of a request that got no HTTP answer, a connection
        error, a timeout or a dropped connection, wherever the client's requests raise it, that it got none: its error
        begins with NO_ANSWER, then the exception's type by name and its words, as the test chat's does, which the start
        records as not checked. An HTTP status, a model server's error and an answer it cannot read are the model's to
        answer for."""
        lc = models.client()
        s = dict(
            config.worker_defaults(),
            backend="openai",
            url="http://127.0.0.1:1/gw/v2",
            model=fake_models.GOOD,
        )
        for exc, unanswered in (
            (ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"), True),
            (TimeoutError("no answer or runner progress in 120 s"), True),
            (lc.RunnerHung("the runner made no progress"), True),
            (http.client.RemoteDisconnected("Remote end closed connection"), True),
            (http.client.IncompleteRead(b""), True),
            (socket.gaierror(socket.EAI_NONAME, "Name or service not known"), True),
            (ssl.SSLError(1, "[SSL] record layer failure"), True),
            (ConnectionError("HTTP 500: b'the server failed'"), False),
            (RuntimeError("model server: this model does not support tools"), False),
            (json.JSONDecodeError("Expecting value", "", 0), False),
        ):
            with self.subTest(type(exc).__name__, unanswered=unanswered):
                with mock.patch.object(lc, "chat", side_effect=exc):
                    t = lc.probe_tools(s, timeout=1)
                self.assertFalse(t["ok"])
                self.assertEqual(
                    t["error"].startswith(f"{lc.NO_ANSWER}: {type(exc).__name__}: "),
                    unanswered,
                    t["error"],
                )
                self.assertEqual(
                    t["error"].startswith(lc.NO_ANSWER), unanswered, t["error"]
                )

    def test_an_ollama_server_is_checked_and_probed_again_for_another_model(self):
        """An Ollama server's model gets the same checks: one that answers in text is recorded for probe_failure;
        once the binding names another model, the next start probes the server again, and one that passes is the
        last."""
        srv = self.ollama(fake_models.OLLAMA_TEXT)
        self.provision(("WARN", "local review loop's server"))
        w = config.worker(config.load())
        self.assertTrue(
            config.probe_failure(w).startswith("failed the tool-call probe")
        )
        self.ollama(fake_models.OLLAMA_GOOD, srv)
        code, lines = self.rk("provision")
        self.assertEqual(
            (code, rows(lines)),
            (0, IN_PLACE + [("PASS", "local review loop's server")]),
        )
        self.assertIn(
            f"{config.safe_url(srv.url)}: {fake_models.OLLAMA_GOOD} answers the test chat and the tool-call probe",
            one_line("\n".join(lines)),
        )
        w = config.worker(config.load())
        self.assertEqual(
            (w["probe"]["model"], w["probe"]["chat"], w["probe"]["tools"]),
            (fake_models.OLLAMA_GOOD, "ok", "ok"),
        )
        self.assertEqual(config.probe_failure(w), "")
        self.not_probed(srv.url, fake_models.OLLAMA_GOOD, "ollama")

    def test_another_binding_is_probed_alone_and_what_was_found_for_the_last_one_replaced(
        self,
    ):
        """What a probe found (the places of the model list and the chat route, the name the list gives the model)
        is that binding's: once the binding names another server, the next start probes it from the binding alone,
        and replaces or clears each of them."""
        route = self.gateway(
            "route",
            chat_routes=("/chat/completions",),
            models_paths=("/models",),
            deployments=True,
        )
        host = route.url.removesuffix(fake_models.Gateway.prefix)
        self.binding(route.url, fake_models.DEPLOY)
        text = self.provision(("PASS", "local review loop's server"))
        self.assertIn(RECORDED_NAME, text)
        w = config.worker(config.load())
        self.assertEqual(
            (w["models_url"], w["chat_path"], w["model_name"]),
            (f"{host}/models", "/chat/completions", fake_models.SHOWN),
        )
        self.assertEqual(config.reviewer_name(config.load()), fake_models.SHOWN)
        plain = self.gateway("plain")
        self.binding(plain.url)
        code, lines = self.rk("provision")
        self.assertEqual(
            (code, rows(lines)),
            (0, IN_PLACE + [("PASS", "local review loop's server")]),
        )
        self.assertIn(RECORDED_NAME, one_line("\n".join(lines)))
        w = config.worker(config.load())
        self.assertEqual(
            (w.get("models_url"), w.get("chat_path"), w.get("model_name")),
            ("", "", ""),
        )
        self.assertEqual(
            (w["probe"]["url"], w["probe"]["chat_path"], w["probe"]["tools"]),
            (plain.url, "/v1/chat/completions", "ok"),
        )

    def test_a_probe_again_of_the_same_binding_keeps_what_it_does_not_find(self):
        """A start that probes the same binding again keeps what the last probe found where this one finds nothing:
        while the server gives no answer, and while its model list does not answer the look-up of the model's name.
        """
        routes = dict(
            chat_routes=("/chat/completions",),
            models_paths=("/models",),
            deployments=True,
        )
        # The tool-call probe's request gets a 429 that outlasts the probe's waits, so the next start probes again.
        route = self.gateway("route", fail429=(2,), retry_after="600", **routes)
        host = route.url.removesuffix(fake_models.Gateway.prefix)
        self.binding(route.url, fake_models.DEPLOY)
        self.provision(("WARN", "local review loop's server"))

        def found() -> tuple:
            w = config.worker(config.load())
            return tuple(w.get(key) for key in provision.FOUND)

        want = (f"{host}/models", "/chat/completions", fake_models.SHOWN)
        self.assertEqual(found(), want)
        route.stop()
        text = self.provision(("WARN", "local review loop's server"))
        # What the last probe found stands in for what this one does not find, and the row names it.
        self.assertIn(
            f"the monitor takes its model list to be at {host}/models and its chat route to be /chat/completions",
            text,
        )
        probe = config.worker(config.load())["probe"]
        self.assertTrue(probe["chat"].startswith("not checked: "), probe)
        self.assertEqual(found(), want)
        again = fake_models.start(
            self.dir / "fake",
            "gateway",
            name="route",
            port=int(host.rsplit(":", 1)[1]),
            key_file=config.worker(config.load())["api_key_file"],
            **routes,
        )
        self.addCleanup(again.stop)
        with mock.patch(
            "rkmon_lib.models.openai_list",
            return_value=([], f"no answer from {again.url}", ""),
        ):
            self.provision(("PASS", "local review loop's server"))
        self.assertEqual(found(), want)

    def test_where_the_server_answers_is_recorded(self):
        self.binding("http://127.0.0.1:1/gw/v2")
        srv = fake_models.start(
            self.dir / "fake",
            "gateway",
            key_file=str(self.key_file()),
            chat_routes=("/chat/completions",),
            models_paths=("/models",),
        )
        self.addCleanup(srv.stop)
        self.binding(srv.url)
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        host = srv.url.removesuffix(fake_models.Gateway.prefix)
        self.assertIn(("PASS", "local review loop's server"), rows(lines))
        self.assertIn(RECORDED, one_line("\n".join(lines)))
        w = config.worker(config.load())
        self.assertEqual(
            (w["models_url"], w["chat_path"]), (f"{host}/models", "/chat/completions")
        )
        self.assertEqual(self.state()["provisioned"], RECORD)

    def test_a_server_that_does_not_answer_holds_nothing_up(self):
        base = f"http://127.0.0.1:{sandbox.closed_port(self)}/v1"
        self.binding(base)
        self.key_file()
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertIn(("WARN", "local review loop's server"), rows(lines))
        self.assertIn(
            f"that refuses no model, and {WRAPPER} doctor reads this record",
            one_line("\n".join(lines)),
        )
        # The default places, a base that ends in /v1 given no second /v1.
        self.assertIn(
            f"until the test chat is answered, the monitor takes its model list to be at {base}/models and its chat "
            f"route to be /v1/chat/completions {RESTART}",
            one_line("\n".join(lines)),
        )
        self.assertEqual(self.state()["provisioned"], RECORD)
        w = config.worker(config.load())
        self.assertEqual(
            (w.get("models_url") or "", w.get("chat_path") or ""), ("", "")
        )

    def test_a_server_that_answers_later_has_its_route_recorded_by_the_next_start(self):
        """A server that does not answer at the first start: a WARN that says the next start probes it again, and its
        probe recorded (worker.probe) as not checked, which never refuses the model for its role. Once it answers, the
        next start's rk provision probes it again, that step alone, and records its routes and the probe; a start after
        that probes it no more."""
        held = socket.socket()
        # bound, never listening: nothing answers at its port until it is closed
        held.bind(("127.0.0.1", 0))
        self.addCleanup(held.close)
        host = f"http://127.0.0.1:{held.getsockname()[1]}"
        self.binding(host + fake_models.Gateway.prefix)
        key = self.key_file()
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertIn(("WARN", "local review loop's server"), rows(lines))
        self.assertIn(
            f"that refuses no model, and {WRAPPER} doctor reads this record; the next start probes it again; until "
            f"the test chat is answered, the monitor takes its model list to be at {host}{fake_models.Gateway.prefix}"
            f"/v1/models and its chat route to be /v1/chat/completions {RESTART}",
            one_line("\n".join(lines)),
        )
        w = config.worker(config.load())
        probe = w["probe"]
        self.assertEqual((probe["url"], probe["model"]), (w["url"], w["model"]))
        self.assertTrue(probe["chat"].startswith("not checked: "), probe)
        self.assertEqual(config.probe_failure(w), "")
        self.assertEqual(
            (w.get("models_url") or "", w.get("chat_path") or ""), ("", "")
        )
        held.close()
        srv = fake_models.start(
            self.dir / "fake",
            "gateway",
            port=int(host.rsplit(":", 1)[1]),
            key_file=str(key),
            chat_routes=("/chat/completions",),
            models_paths=("/models",),
        )
        self.addCleanup(srv.stop)
        self.assertEqual(srv.url, w["url"])
        code, lines = self.rk("provision")
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertEqual(
            rows(lines), IN_PLACE + [("PASS", "local review loop's server")]
        )
        self.assertIn(RECORDED, one_line("\n".join(lines)))
        self.assertEqual(lines[-1], provision.DONE)
        w = config.worker(config.load())
        self.assertEqual(
            (w["models_url"], w["chat_path"]), (f"{host}/models", "/chat/completions")
        )
        self.assertEqual(
            (w["probe"]["chat"], w["probe"]["chat_path"]), ("ok", "/chat/completions")
        )
        self.not_probed(srv.url, fake_models.GOOD)

    def test_a_change_of_only_the_key_or_how_it_is_sent_is_probed_again(self):
        """The last probe's record keeps how it sent the key (models.key_sent: a digest of the headers that carried it,
        never the key): once a probe of the model passed both, the next start probes the server again when only the
        key changed, or only how it is sent, and the start after it does not."""
        srv = self.gateway()
        self.binding(srv.url)
        self.provision(("PASS", "local review loop's server"))
        self.not_probed(srv.url, fake_models.GOOD)
        # The gateway takes the key the secret has, whichever it is.
        secret = pathlib.Path(config.worker(config.load())["api_key_file"])
        for label, change in (
            (
                "another key",
                lambda: secret.write_text("example-key-2\n", encoding="utf-8"),
            ),
            (
                "the key sent another way",
                lambda: self.binding(srv.url, api_key_sent_as="header_and_bearer"),
            ),
        ):
            with self.subTest(label):
                asked = len(fake_models.read_log(self.dir / "fake", "gateway"))
                change()
                text = self.provision(("PASS", "local review loop's server"))
                self.assertIn(
                    f"{fake_models.GOOD} answers the test chat and the tool-call probe",
                    text,
                )
                self.assertGreater(
                    len(fake_models.read_log(self.dir / "fake", "gateway")), asked
                )
                probe = config.worker(config.load())["probe"]
                self.assertNotIn("example-key", json.dumps(probe))
                self.not_probed(srv.url, fake_models.GOOD)

    def test_a_passed_record_without_key_sent_is_probed_again_once(self):
        """A passed probe's record that does not say how it sent the key (no key_sent) is not taken for this key's: the
        next start probes the server again and records how it sends the key, and the start after it does not.
        """
        srv = self.gateway()
        self.binding(srv.url)
        self.provision(("PASS", "local review loop's server"))
        rec = dict(config.worker(config.load())["probe"])
        del rec["key_sent"]
        state_store.set(self.home, provision.PROBE, rec)
        asked = len(fake_models.read_log(self.dir / "fake", "gateway"))
        self.provision(("PASS", "local review loop's server"))
        self.assertGreater(
            len(fake_models.read_log(self.dir / "fake", "gateway")), asked
        )
        cfg = config.load()
        self.assertEqual(
            config.worker(cfg)["probe"]["key_sent"],
            models.key_sent(provision._worker(cfg)),
        )
        self.not_probed(srv.url, fake_models.GOOD)

    def test_no_job_gets_the_last_probe_s_record(self):
        """The last probe's record, with how it sent the key (key_sent), stays in the internal state: the reviewer
        model's settings the jobs get (config.env's RKMON_WORKER_JSON, which each job's script exports and rk env
        prints: config.export_line) leave it out."""
        srv = self.gateway()
        self.binding(srv.url)
        self.provision(("PASS", "local review loop's server"))
        cfg = config.load()
        sent = config.worker(cfg)["probe"]["key_sent"]
        worker = json.loads(config.env(cfg)["RKMON_WORKER_JSON"])
        self.assertEqual(worker["model"], fake_models.GOOD)
        self.assertNotIn("probe", worker)
        self.assertNotIn(sent, config.export_line(cfg))

    def test_the_start_names_the_one_home_of_when_it_probes_again(self):
        """The wrapper's start says when it probes the local review loop's server again by naming
        provision._probes_again, the rule's one home, which is there."""
        wrapper = (ROOT / "docker" / "rkmon").read_text(encoding="utf-8")
        start = wrapper[wrapper.index("    def cmd_start(") :]
        doc = " ".join(start[: start.index('"""', start.index('"""') + 3)].split())
        self.assertIn("server again if provision._probes_again says so", doc)
        self.assertTrue(callable(provision._probes_again))


if __name__ == "__main__":
    unittest.main()
