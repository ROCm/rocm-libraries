# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The delivered deploy settings the monitor runs (rkmon_lib/applied.py): a layer that passes the
load is recorded as applied, once, by rk start and by `rk applied --load` as soon as it runs, so that the record holds
a new layer without waiting for a tick (apply waits at most two minutes); a layer that is broken, of another package
version, of another layout of the deploy settings or of another kind is refused by both, in the same words, with its
fix (`rkmon apply` or `rkmon rollback`), and never recorded, and so is one whose configuration cannot run; `rk applied
--load --json` gives the refusal and its fix as JSON, for rkmon apply to keep with the change it puts back; a layer
missing where the monitor has run one is refused as missing, and so is one missing while the record cannot be read,
while a home that never ran one keeps config.json; a record that cannot be read is refused with its fix, and written
anew as a layer is recorded; one that cannot be written is said, with why and what to do; and `rk applied` says which
layer runs, whether a pause holds and whether the monitor is in standby, as the host's deploy side asks.
"""

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
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    applied,
    checks,
    config,
    container_layout,
    control,
    doctor,
    file_types,
    image,
    paths,
    runtime_config,
    state_store,
    store,
)

# A layer's settings whose configuration runs, and the keys of a layer's settings your file sets.
from test_runtime_config import SETTINGS, set_in  # noqa: E402

RC = runtime_config
VERSION = image.package_version(paths.PACKAGE)
FIRST, SECOND = "sha256:" + "1" * 64, "sha256:" + "2" * 64
WRAPPER = "./rkmon"  # the host's command, as the container's wrapper passes it in (RKMON_WRAPPER)
FIX = "fix: ./rkmon apply (this package's code, delivered with your settings), or ./rkmon rollback"
# rk start's lead for settings that cannot run, where `rk applied --load` says "REFUSED: ".
NOT_LOADED = "not loaded: "
# What config.json holds after a first setup: the same answers as SETTINGS.
CONFIG_JSON = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "operator": {"github_login": "<user_login>"},
    "scope": {"labels": ["team: widgets"]},
}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = sandbox.own_home(self)
        self.settings_dir = self.dir / "settings"
        self.settings_dir.mkdir()
        # store.now, which the applied record's applied_at reads.
        self.clock = "2026-01-02T03:00:00Z"
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
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.object(store, "now", side_effect=lambda: self.clock),
            mock.patch.dict(
                os.environ,
                {"RKMON_WRAPPER": WRAPPER, "XDG_CONFIG_HOME": str(self.dir / "xdg")},
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def layer(self, **over) -> dict:
        """A delivered layer, written as deliver writes it: whole, renamed into place."""
        settings = over.get("settings", SETTINGS)
        layer = {
            "kind": RC.LAYER_KIND,
            "schema_version": RC.LAYER_SCHEMA_VERSION,
            "deploy_schema_version": file_types.DEPLOY.version,
            "package_version": VERSION,
            "confirmed_at": "2026-01-02T01:00:00Z",
            "fingerprint": FIRST,
            "settings": settings,
            "set_in_file": set_in(settings),
            "container": {},
        }
        layer.update(over)
        new = self.settings_dir / "deploy.json.new"
        new.write_text(json.dumps(layer), encoding="utf-8")
        new.replace(self.settings_dir / RC.LAYER_FILE)
        return layer

    def rk(self, *argv) -> tuple[int, list[str]]:
        """rk ARGV as the container runs it: its exit and the lines it printed."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(list(argv))
        return code, out.getvalue().splitlines()

    def start(self) -> list[str]:
        """The monitor's start (rk start, which the container's start runs): the lines it printed."""
        out = []
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        control.start({}, store.Store(), run=run, out=out.append)
        return out

    def config_json(self):
        """config.json as a first setup writes it, the monitor's working folder included."""
        (self.dir / "work").mkdir(exist_ok=True)
        conf = dict(CONFIG_JSON, paths={"working_dir": str(self.dir / "work")})
        (self.home / "config.json").write_text(json.dumps(conf), encoding="utf-8")

    def recorded(self):
        rec = RC.applied(self.home)
        return rec and (rec["fingerprint"], rec["package_version"], rec["applied_at"])

    def refused_json(self) -> list[str]:
        """`rk applied --load --json`'s refusal, exit 4, as `rk applied --load` says it: its REFUSED line, then its fix
        line, when it has a fix."""
        code, said = self.rk("applied", "--load", "--json")
        self.assertEqual((code, len(said)), (4, 1), said)
        refused = json.loads(said[0])
        fix = checks.fix_from_json(refused["fix"])
        return [f"REFUSED: {refused['refused']}"] + ([f"fix: {fix}"] if fix else [])


class TestApplied(Base):
    def test_the_start_records_a_layer_that_passes_once(self):
        self.layer()
        said = self.start()
        self.assertIn(
            f"loaded: the settings you confirmed at 2026-01-02T01:00:00Z (package {VERSION}), recorded as "
            "applied",
            said,
        )
        self.assertEqual(self.recorded(), (FIRST, VERSION, "2026-01-02T03:00:00Z"))
        # A second start on the same layer, later: the record stays as it was.
        self.clock = "2026-01-02T03:05:00Z"
        said = self.start()
        self.assertIn(
            f"loaded: the settings you confirmed at 2026-01-02T01:00:00Z (package {VERSION}), applied at "
            "2026-01-02T03:00:00Z",
            said,
        )
        self.assertEqual(self.recorded(), (FIRST, VERSION, "2026-01-02T03:00:00Z"))

    def test_a_new_layer_is_taken_up_as_soon_as_rk_applied_load_runs(self):
        """A delivery that needs no restart: the next command, not the next tick, loads the new layer, and the record
        holds it once `rk applied --load` returns, whatever the tick interval."""
        self.layer()
        self.start()
        self.layer(fingerprint=SECOND, confirmed_at="2026-01-02T02:00:00Z")
        self.assertEqual(self.recorded()[0], FIRST)  # delivered, not loaded yet
        self.clock = "2026-01-02T03:01:00Z"
        code, said = self.rk("applied", "--load")
        self.assertEqual(code, 0, said)
        self.assertEqual(
            said,
            [
                f"loaded: the settings you confirmed at 2026-01-02T02:00:00Z (package {VERSION}), recorded "
                "as applied"
            ],
        )
        self.assertEqual(self.recorded(), (SECOND, VERSION, "2026-01-02T03:01:00Z"))
        self.clock = "2026-01-02T03:02:00Z"
        code, said = self.rk("applied", "--load")  # loaded again: recorded already
        self.assertEqual(
            (code, said),
            (
                0,
                [
                    f"loaded: the settings you confirmed at 2026-01-02T02:00:00Z (package {VERSION}), "
                    "applied at 2026-01-02T03:01:00Z"
                ],
            ),
        )
        code, said = self.rk("applied")
        self.assertEqual(
            (code, said),
            (
                0,
                [
                    f"applied: your confirmed settings {SECOND} (package {VERSION}), at 2026-01-02T03:01:00Z"
                ],
            ),
        )

    def test_a_record_that_cannot_be_written_is_said_and_the_last_one_stays(self):
        self.layer()
        self.start()
        unwritable = OSError(30, "Read-only file system")
        write_file = state_store.write_file

        def record_unwritable(file, data):
            """The applied record cannot be written; the state folder's other files (rk start's state) can."""
            if pathlib.Path(file).name == RC.APPLIED_FILE:
                raise unwritable
            write_file(file, data)

        state = config.state_dir(self.home)
        # A new layer, and then the same settings as the record's under another package (as after a package update).
        for over, last in (
            ({"fingerprint": SECOND}, (FIRST, VERSION)),
            ({}, (FIRST, "0.0.1")),
        ):
            with self.subTest(over=over):
                if over == {}:
                    rec = dict(RC.applied(self.home), package_version="0.0.1")
                    state_store.write_file(
                        config.state_dir(self.home) / RC.APPLIED_FILE, rec
                    )
                self.layer(confirmed_at="2026-01-02T02:00:00Z", **over)
                with mock.patch.object(
                    state_store, "write_file", side_effect=record_unwritable
                ):
                    code, said = self.rk("applied", "--load")
                    started = self.start()
                self.assertEqual(
                    (code, said),
                    (
                        1,
                        [
                            "loaded: the settings you confirmed at 2026-01-02T02:00:00Z (package "
                            f"{VERSION}); the applied record cannot be written in {state}: Read-only file system",
                            f"fix: make {state} writable, then run {WRAPPER} rk applied --load",
                        ],
                    ),
                )
                # rk start says the same, and starts.
                self.assertEqual(started[1:3], said)
                self.assertEqual(self.recorded()[:2], last)

    def test_a_record_that_cannot_be_read_back_is_said_and_rk_start_goes_on(self):
        self.layer()
        self.start()
        held = RC.applied(self.home)
        gone = RC.RecordError("the applied record cannot be read: Input/output error")
        said = [
            "loaded: the settings you confirmed at 2026-01-02T01:00:00Z (package "
            f"{VERSION}); the applied record cannot be read: Input/output error",
            f"fix: run {WRAPPER} rk applied --load",
        ]
        # The record holds the layer as record_applied reads it, and cannot be read as `record` reads it back.
        with mock.patch.object(RC, "applied", side_effect=[held, gone] * 2):
            self.assertEqual(self.rk("applied", "--load"), (1, said))
            self.assertEqual(self.start()[1:3], said)

    def test_rk_applied_json_answers_the_host_with_the_record_the_pause_and_standby(
        self,
    ):
        code, said = self.rk("applied", "--json")
        self.assertEqual(
            (code, json.loads(said[0])),
            (1, {"applied": None, "paused": False, "standby": False}),
        )
        self.layer()
        self.start()
        code, said = self.rk("applied", "--load", "--json")  # the JSON alone
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads("\n".join(said)),
            {
                "applied": {
                    "kind": RC.APPLIED_KIND,
                    "schema_version": RC.APPLIED_SCHEMA_VERSION,
                    "fingerprint": FIRST,
                    "package_version": VERSION,
                    "applied_at": "2026-01-02T03:00:00Z",
                },
                "paused": False,
                "standby": False,
            },
        )
        self.config_json()
        self.assertEqual(self.rk("pause")[0], 0)
        self.assertTrue(json.loads(self.rk("applied", "--json")[1][0])["paused"])
        self.assertEqual(self.rk("resume")[0], 0)
        self.assertFalse(json.loads(self.rk("applied", "--json")[1][0])["paused"])
        for on in (True, False):
            state_store.set(self.home, "standby.on", on)
            said = json.loads(self.rk("applied", "--json")[1][0])
            self.assertIs(said["standby"], on)

    def test_a_layer_that_cannot_run_is_refused_with_its_fix_and_never_recorded(self):
        layout = file_types.DEPLOY.version
        path = self.settings_dir / RC.LAYER_FILE
        for over, found in (
            (None, f"your confirmed settings at {path} are no JSON"),
            (
                {"package_version": "0.0.0"},
                f"it was confirmed for the package '0.0.0', and this is the package {VERSION!r}",
            ),
            (
                {"deploy_schema_version": layout - 1},
                f"it was confirmed for layout {layout - 1} of the deploy settings",
            ),
            (
                {"schema_version": RC.LAYER_SCHEMA_VERSION + 1},
                f"not 'confirmed_deploy_settings' at layout {RC.LAYER_SCHEMA_VERSION}",
            ),
        ):
            with self.subTest(over=over):
                if over is None:
                    path.write_text("{", encoding="utf-8")
                else:
                    self.layer(**over)
                code, said = self.rk("applied", "--load")
                self.assertEqual(code, 4, said)
                self.assertTrue(said[0].startswith("REFUSED: "), said)
                self.assertIn(found, said[0])
                self.assertEqual(said[-1], FIX)
                self.assertEqual(self.refused_json(), said)
                # rk start says the same, then the fix, and starts.
                self.assertEqual(
                    self.start()[1:3],
                    [said[0].replace("REFUSED: ", NOT_LOADED, 1), FIX],
                )
                self.assertIsNone(RC.applied(self.home))
        # A layer that runs, whose configuration cannot: refused, with what is wrong, and never recorded.
        self.layer(settings=dict(SETTINGS, repository={"name": "not a repository"}))
        code, said = self.rk("applied", "--load")
        self.assertEqual(code, 4)
        self.assertIn("the configuration cannot run", said[0])
        self.assertNotIn(FIX, said)
        self.assertEqual(self.refused_json(), said)
        started = self.start()
        self.assertEqual(started[1], said[0].replace("REFUSED: ", NOT_LOADED, 1))
        self.assertNotIn(FIX, started)
        self.assertIsNone(RC.applied(self.home))

    def test_a_layer_missing_where_the_monitor_ran_one_is_refused_and_config_json_never_takes_over(
        self,
    ):
        self.config_json()
        # A home that never ran a delivered layer runs config.json, as before any delivery.
        self.assertEqual(RC.load(self.home).source["repo"], RC.CONFIG_JSON)
        code, said = self.rk("applied", "--load")
        self.assertEqual(
            (code, said),
            (
                1,
                [
                    "loaded: your confirmed settings are not here: the configuration is config.json's"
                ],
            ),
        )
        self.assertFalse([s for s in self.start() if s.startswith("loaded:")])
        self.assertIsNone(RC.applied(self.home))
        self.layer()
        self.start()
        (self.settings_dir / RC.LAYER_FILE).unlink()
        with self.assertRaises(RC.LayerError) as e:
            RC.load(self.home)
        self.assertEqual(
            str(e.exception),
            f"your confirmed settings at {self.settings_dir / RC.LAYER_FILE} are missing, and this monitor has run "
            "your confirmed settings before (applied at 2026-01-02T03:00:00Z)",
        )
        code, said = self.rk("applied", "--load")
        self.assertEqual((code, said[-1]), (4, FIX))
        # rk start says the layer is missing, then the fix.
        self.assertEqual(self.start()[1:3], [NOT_LOADED + str(e.exception), FIX])
        # The record still names what the monitor last ran.
        code, said = self.rk("applied")
        self.assertEqual((code, said[0].split()[4]), (0, FIRST))

    def test_a_record_that_cannot_be_read_is_refused_then_written_anew(self):
        path = config.state_dir(self.home) / RC.APPLIED_FILE
        missing = (
            f"your confirmed settings at {self.settings_dir / RC.LAYER_FILE} are missing, and whether this "
            "monitor has run your confirmed settings before is not known: "
        )
        self.config_json()  # there, and never taking over
        cases = [
            ("{", f"the applied record {path} is no JSON: "),
            (
                "[]",
                f"{path} is no applied record: it is not of the kind {RC.APPLIED_KIND!r}",
            ),
            ('{"kind": "other"}', f"{path} is no applied record"),
        ]
        if os.geteuid() != 0:  # root reads a file whatever its mode
            cases.append(
                (0, f"the applied record {path} cannot be read: Permission denied")
            )
        for bad, why in cases:
            with self.subTest(bad=bad):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(bad) if bad == 0 else bad, encoding="utf-8")
                if bad == 0:
                    path.chmod(0)
                code, said = self.rk("applied")
                self.assertEqual(code, 4, said)
                self.assertTrue(said[0].startswith("REFUSED: " + why), said)
                self.assertEqual(said[1:], [f"fix: run {WRAPPER} rk applied --load"])
                # With no layer delivered, the load is refused: whether this monitor ran one is not known.
                with self.assertRaises(RC.LayerError) as e:
                    RC.load(self.home)
                self.assertTrue(str(e.exception).startswith(missing + why), e.exception)
                self.assertEqual(
                    self.rk("applied", "--load"), (4, [f"REFUSED: {e.exception}", FIX])
                )
                # A layer delivered: rk start records it, the record written anew.
                self.layer()
                self.assertIn(
                    f"loaded: the settings you confirmed at 2026-01-02T01:00:00Z (package {VERSION}), "
                    "recorded as applied",
                    self.start(),
                )
                self.assertEqual(
                    self.recorded(), (FIRST, VERSION, "2026-01-02T03:00:00Z")
                )
                (self.settings_dir / RC.LAYER_FILE).unlink()
                path.unlink()

    def test_rk_applied_without_a_record(self):
        code, said = self.rk("applied")
        self.assertEqual(
            (code, said),
            (
                1,
                [
                    "applied: none: the monitor has run none of your confirmed settings here"
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
