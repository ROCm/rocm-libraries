# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What the deploy path's tests share, one module of them for each part of the path (tests/test_deploy_path_<part>.py):
DeployPathCase, a test case that runs docker/rkmon on a copy of the package's docker/ and rkmon_lib/
(deploy_folder.copy) against the stand-in docker (tests/fakes/fake_docker, scripted per call), with the helpers that
tests of more than one part call; and the stand-in docker's answers and the patterns that tests of more than one part
read, each with what it is. It holds no test, so each test runs once, in its own module.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import deploy_folder
import sandbox
from rendered import state_2_next
from rkmon_lib import (
    checks,
    code_volume,
    deliver,
    image,
    outcome,
    records,
    runtime_config,
    secrets_registry,
)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# An outcome line's start (outcome.line): a word of a thing made or moved and a space, any other word and a colon, or
# the word alone; and a next or done line's. A colon is at the line's end when what follows it moved whole to the next
# line (a long temporary folder's path).
OUTCOME_LINE = re.compile(
    rf"^(?:({'|'.join(map(re.escape, outcome.OBJECT_WORDS))})(?: |$)"
    rf"|({'|'.join(re.escape(w) for w in outcome.WORDS if w not in outcome.OBJECT_WORDS)})(?::(?: |$)|$))",
    re.M,
)
NEXT_LINE = re.compile(r"^(?:next|done):(?: |$)", re.M)
# What a command run in a fresh folder needs of the stand-in docker: start waits for its container, up at once here.
FRESH_ENV = {"start": {"FAKE_DOCKER_RUNNING": "1"}}
# Another monitor on this host, as docker would answer a listing of the running containers of this package's image (its
# title label) and a read of that monitor's config.json: a deployment of the compose project elsewhere that posts
# reviews for acme/widgets. No command asks either: assert_reads_no_other_monitor.
ELSEWHERE = [
    [r"^ps .*org\.opencontainers\.image\.title=", "elsewhere elsewhere-rkmon-1", 0],
    [
        r"^exec elsewhere-rkmon-1 ",
        json.dumps(
            {
                "repo": "acme/widgets",
                "features": {"review_automation": {"mode": "live"}},
            }
        ),
        0,
    ],
]
# The docker calls, as the stand-in docker logs them, that make a Docker object that stays: a compose project's
# container (and so its volumes and network), or a volume, network or container of its own. Before the first start,
# only login, secret, adopt and start make one.
MAKES = re.compile(
    r"\t(?:compose (?:run|up|create|start)|(?:volume|network|container) create|create) "
)
LABELS = {
    "org.rocke.package.version": image.package_version(ROOT),
    "org.rocke.build.date": "2026-11-13T00:00:00Z",
    "org.rocke.base": "ubuntu:26.04",
    "org.rocke.rocm.version": "10.1.0-5",
}
GOOD = json.dumps(LABELS)
# The one-off containers that write the settings volume (deliver.LAYER_RUN) and read its layer
# (Wrapper.delivered_layer), and what the calls before an answer show once apply wrote the layer, and once it then put
# back the layer it kept (fake_docker's fifth element)
LAYER_RUN = re.escape(" ".join(deliver.LAYER_RUN))
LAYER_READ = re.escape(f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}") + "$"
LAYER_KEPT_READ = re.escape(f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_KEPT}") + "$"
LAYER_WRITTEN = LAYER_RUN
LAYER_PUT_BACK = f"(?s){LAYER_RUN}.*{LAYER_RUN}"
# the load apply waits for (rkmon_lib/applied.py), and the restart (docker compose up -d --force-recreate)
LOAD = r"rk applied --load --json$"
# why the settings volume cannot be read: docker's answer to the layer's read
UNREAD = "Error response from daemon: failed to create task for container: no space left on device"
RESTART = r"compose up -d --force-recreate$"
# the running container's start (its ID, then its State.StartedAt), and rk's last tick since it
STARTED = (
    [r"compose ps -q rkmon$", "0123456789ab", 0],
    [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
)
TICK_SINCE_START = [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0]
# rk drain's lines as it drains (rkmon_lib/drain.py), and code-install's outcomes (docker/rk_root.py): a release
# in, and nothing to do over an image whose own code is this package's
DRAINED = (
    "drain: nothing new starts; the work in flight finishes first (up to 60 minutes): 2 reviews",
    "DRAINED: nothing is in flight after 12 minutes",
)
INSTALLED = "INSTALLED: release 1 over this image, in place of the image's code"
OWN_CODE = "NOTHING TO DO: this code runs already (the image's own code)"
# The fix of a model that does not answer: its address set to one the container reaches
MODEL_FIX = checks.Edit(
    "model_bindings.fast.base_url",
    "~/rkmon-deploy.yaml",
    checks.Described("an address the container reaches"),
)
# The team dashboard published to its web host: what it needs set while it is on, beside its switch (the confirm
# fails each one blank)
PUBLISHED = {
    "enabled": True,
    "delivery": "publish",
    "public_url": "https://dashboards.example.com/widgets/",
    "publish": {
        "web_host": "web.example.com",
        "username": "widgets-web",
        "site_folder": "public_html/widgets",
    },
}


class DeployPathCase(unittest.TestCase):
    """A test of docker/rkmon (self.w) in a copy of the package (self.pkg): `rkmon` runs it in a home folder of the
    test's own (self.home), with the stand-in docker first on its PATH, which answers as `answer` scripts it and logs
    each call (`calls`); your deploy settings file there (self.settings, `write_settings`) puts the admin dashboard on a
    free port (self.port). Its other helpers are those that tests of more than one part of the path call.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.pkg = deploy_folder.copy(self.d / "pkg")
        self.w = str(self.pkg / "docker" / "rkmon")
        self.bin = sandbox.safe_path(self.d)
        shutil.copy(HERE / "fakes" / "fake_docker", pathlib.Path(self.bin) / "docker")
        (pathlib.Path(self.bin) / "docker").chmod(0o755)
        self.home = self.d / "home"
        self.home.mkdir()
        self.port = sandbox.free_port()
        self.settings = self.home / "rkmon-deploy.yaml"
        self.write_settings()
        self.log, self.stdin = self.d / "docker.log", self.d / "stdin.log"
        self.answers = self.d / "answers.json"
        self.answer([])
        self.env = {
            "PATH": self.bin,
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_DOCKER_ANSWERS": str(self.answers),
            "FAKE_DOCKER_STDIN": str(self.stdin),
            # the build path's check asks a public site: a closed port instead
            "RKMON_TLS_SITE": f"127.0.0.1:{sandbox.closed_port(self)}",
            # this host's memory and CPUs, as the tests set them, whatever the machine
            **deploy_folder.HOST_SIZE,
        }

    def write_settings(
        self,
        port=None,
        features=None,
        image_tag="rkmon-test",
        registry="",
        image_name="",
        memory_gib=12,
        ramdisk_gib=4,
        bindings=None,
        more="",
        integrations=None,
        cpus=2,
    ):
        """The deploy settings file (deploy_folder.write_settings), its admin dashboard on this test's port by
        default."""
        deploy_folder.write_settings(
            self.settings,
            port or self.port,
            features=features,
            image_tag=image_tag,
            registry=registry,
            image_name=image_name,
            memory_gib=memory_gib,
            ramdisk_gib=ramdisk_gib,
            bindings=bindings,
            more=more,
            integrations=integrations,
            cpus=cpus,
        )

    def answer(self, rows):
        """The stand-in docker's answers ROWS (tests/fakes/fake_docker), with those of an image built from this
        package where ROWS give none: first, that the image's own code is this package's (image_code); last, that
        code-install, apply's or the start's, has nothing to do over it (OWN_CODE)."""
        if not any(r[0] == deploy_folder.IMAGE_CODE for r in rows):
            rows = [*self.image_code(), *rows]
        rows = [*rows, [rf"code-install( {code_volume.LEAVE_KEPT})?$", OWN_CODE, 0]]
        self.answers.write_text(json.dumps(rows), encoding="utf-8")

    def rkmon(self, *args, stdin=None, **env):
        p = subprocess.run(
            [self.w, *args],
            cwd=self.home,
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            timeout=120,
        )
        return p.returncode, p.stdout, p.stderr

    def confirmed(self):
        """Step 3 with --yes: the settings file checked, confirmed and recorded in this folder's records (no image
        here yet)."""
        rows = self.answers.read_text()
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("confirm", "--settings", str(self.settings), "--yes")
        self.assertEqual(rc, 0, out + err)
        self.answers.write_text(rows)

    def confirm_record(self):
        """The last confirm this folder's records keep (records.Confirmed); None before the first."""
        return records.read(self.pkg).confirmed

    def running(self) -> dict:
        """The stand-in docker's environment while this folder's container runs."""
        return {
            "FAKE_DOCKER_RUNNING": "1",
            "FAKE_DOCKER_PROJECT": deploy_folder.NAME,
            "FAKE_DOCKER_WORKDIR": os.path.realpath(self.pkg / "docker"),
        }

    def delivered(self) -> list:
        """What the first start delivers for the confirmed settings, in place without a start, and the stand-in
        docker's answers that the container has them and runs them (deploy_folder.delivered).
        """
        return deploy_folder.delivered(self.pkg, self.settings)

    def login_rows(self, needed="SET", **states) -> list:
        """`rk logins --json`'s rows (rkmon_lib/logins.py rows) as an image of this package lists them in the container
        for the confirmed settings file: each entry the walk lists, in the secrets registry's order, with what needs
        it; its state as STATES names it, a state or (state, found), else NEEDED where something needs it and NOT
        NEEDED where nothing does."""
        made = deploy_folder.plan(self.pkg, self.settings)
        values = runtime_config.from_settings(made.layer["settings"]).values
        rows = []
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
            for e in secrets_registry.walk_entries(values):
                needs = secrets_registry.needed_by(values, e)
                given = states.get(e.name, needed if needs else "NOT NEEDED")
                state, found = given if isinstance(given, tuple) else (given, "")
                rows.append(
                    {
                        "name": e.name,
                        "what": e.what,
                        "needed_by": needs,
                        "state": state,
                        "found": found,
                        "command": " ".join(secrets_registry.set_command(e)),
                        "host_key": False,
                    }
                )
        return rows

    def image_code(self, changed=()) -> list:
        """The stand-in docker's answer, to put first, that the image's own code is this package's, but for the files
        CHANGED (deploy_folder.image_code): the image row of confirm, image check and start reads it.
        """
        return deploy_folder.image_code(self.pkg, changed)

    def calls(self) -> list:
        """The stand-in docker's calls, each whole, as FOLDER<tab>ARGUMENTS: each starts a line with the folder it ran
        in, and a program it was given as an argument spans several lines."""
        out = []
        lines = (
            self.log.read_text(encoding="utf-8").splitlines()
            if self.log.is_file()
            else []
        )
        for line in lines:
            folder, tab, _ = line.partition("\t")
            if (tab and folder.startswith("/")) or not out:
                out.append(line)
            else:
                out[-1] += "\n" + line
        return out

    def docker_calls(self) -> list:
        """The stand-in docker's calls (calls), each without its folder."""
        return [call.partition("\t")[2] for call in self.calls()]

    def assert_reads_no_other_monitor(self):
        """No docker call in the log lists the containers of this package's image by its title, runs a command by
        docker exec or copies by docker cp (this deployment's own run through docker compose), or names another
        deployment's container or volume (ELSEWHERE's), whatever docker would answer: no command reads another
        deployment."""
        read = [
            c
            for c in self.docker_calls()
            if c.startswith(("exec ", "cp "))
            or re.match(r"ps .*image\.title=", c)
            or "elsewhere" in c
        ]
        self.assertEqual(read, [])

    def dotenv(self) -> dict:
        from rkmon_lib import image

        return image.dotenv(self.pkg / "docker" / ".env")

    def state_2_next(self) -> str:
        """State 2's next line: a folder with no deploy settings recorded."""
        return state_2_next(self.w)

    def doctor(self, *failing) -> str:
        """`rk doctor --fresh --json`'s answer (checks.check_to_json): three checks that pass, then a FAIL for each name
        of FAILING, with the fix of a model that does not answer (MODEL_FIX).
        """
        rows = [
            checks.Check(name, checks.Status.PASS, "fine")
            for name in ("ramdisk", "config", "session login: gh")
        ]
        rows += [
            checks.Check(
                name,
                checks.Status.FAIL,
                "no answer from the binding's base_url",
                MODEL_FIX,
            )
            for name in failing
        ]
        return json.dumps([checks.check_to_json(r) for r in rows])

    def apply_answers(self, before: list, was, made) -> list:
        """The stand-in docker's answers, to put first, for apply on this folder's container: the layer in the settings
        volume and the monitor's applied record, BEFORE's (delivered) until apply writes the layer, the plan MADE's
        after (LAYER_WRITTEN), and the plan WAS's again after a put-back (LAYER_PUT_BACK); and each step of apply
        ending well."""

        def loaded(plan) -> str:
            return json.dumps(
                {"applied": {"fingerprint": plan.fingerprint}, "paused": False}
            )

        applied = before[1][0]
        return [
            [LAYER_READ, deliver.texts(was)[2], 0, {}, {"after": LAYER_PUT_BACK}],
            [LAYER_READ, deliver.texts(made)[2], 0, {}, {"after": LAYER_WRITTEN}],
            [applied, loaded(was), 0, {}, {"after": LAYER_PUT_BACK}],
            [applied, loaded(made), 0, {}, {"after": LAYER_WRITTEN}],
            [LOAD, loaded(was), 0, {}, {"after": LAYER_PUT_BACK}],
            [LAYER_KEPT_READ, deliver.texts(was)[2], 0],
            *before,
            [LOAD, loaded(made), 0],
            [LAYER_RUN, "", 0],
            [r"rk drain --timeout-minutes 60$", "\n".join(DRAINED), 0],
            [r"rk undrain$", "UNDRAINED: the drain since 13:58 is lifted", 0],
            [r"code-install$", INSTALLED, 0],
            [r"code-back$", "RESTORED: the image's code runs again", 0],
            [r"rk doctor --fresh --json$", self.doctor(), 0],
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"rk stopped$", "", 1],
            [r"rk logins --json", "[]", 0],
        ]

    def change_waits(self) -> tuple:
        """State 11: the first confirm's settings delivered and loaded (delivered), then the settings file changed
        (memory_gib 13, a change that restarts) and confirmed. (the stand-in docker's answers for apply (apply_answers),
        the plan apply delivers, the plan delivered before it)."""
        self.confirmed()
        before = self.delivered()
        was = deploy_folder.plan(self.pkg, self.settings)
        self.write_settings(memory_gib=13)
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        return self.apply_answers(before, was, made), made, was

    def called(self, *patterns, start: int = 0) -> list:
        """The index in the calls (calls), from START on, of the first call each of PATTERNS is found in (re.search); -1
        for none."""
        calls = self.calls()
        return [
            next((i for i, c in enumerate(calls) if i >= start and re.search(p, c)), -1)
            for p in patterns
        ]

    def kept_env(self):
        """docker/.env's kept copy (deliver.KEPT): its text, None when there is none."""
        path = self.pkg / (deliver.ENV_FILE + deliver.KEPT)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # the older package's own drain as its rk says it (rk stop --drain --wait)
    OLDER_DRAINED = (
        "draining: 2 reviews in flight; nothing new starts",
        "stopped: the reviews in flight finished after 12 minutes",
    )

    def older_package(self, new: str = f"(?m){RESTART}") -> tuple:
        """A deployment an older package made, its package folder updated to this package: docker/.env and the
        override as that package wrote them, no layer ever delivered, your settings file confirmed (state 11), and its
        container running that package's code, whose rk has neither `rk applied` nor `rk provision` (a usage error),
        drains with `rk stop --drain --wait`, whose stop stays until an `rk start` or the restart, as the new
        container's start runs rk start (docker/entrypoint.sh), and has a doctor without --json.
        Once the calls before an answer show NEW, a pattern (the first restart, by default), the new container runs
        this package's code: it loads the settings, says it runs in standby, its doctor passes and a tick has run since
        it started. (the stand-in docker's answers for the first apply over it, the plan apply delivers, the older
        package's docker/.env and override)."""
        self.confirmed()
        made = deploy_folder.plan(self.pkg, self.settings)
        older = (
            "RKMON_IMAGE=rocke-monitor\nRKMON_TAG=older\n",
            "services:\n  rkmon:\n    mem_limit: 12g\n",
        )
        for rel, text in zip((deliver.ENV_FILE, deliver.OVERRIDE_FILE), older):
            (self.pkg / rel).write_text(text, encoding="utf-8")
            # yours only, as the private file check's fix makes them (an older package wrote them for all to read)
            (self.pkg / rel).chmod(0o600)

        def unknown(command: str) -> str:
            return (
                "usage: rk [-h] {start,stop,stopped,doctor} ...\nrk: error: argument cmd: invalid choice: "
                f"'{command}' (choose from start, stop, stopped, doctor)"
            )

        new = {"after": new}
        loaded = json.dumps(
            {
                "applied": {"fingerprint": made.fingerprint},
                "paused": False,
                "standby": True,
            }
        )
        none = "cat: {}/{}: No such file or directory"
        older_doctor = (
            "usage: rk doctor [-h] [--fresh]\nrk: error: unrecognized arguments: --json"
        )
        report = "  PASS  ramdisk: fine\n  PASS  config: fine\n"
        rows = [
            [r"rk applied --json$", loaded, 0, {}, new],
            [r"rk applied --json$", unknown("applied"), 2],
            [r"rk provision --check$", "provisioned", 0, {}, new],
            [r"rk provision --check$", unknown("provision"), 2],
            [LOAD, loaded, 0],
            [LAYER_READ, deliver.texts(made)[2], 0, {}, {"after": LAYER_WRITTEN}],
            [LAYER_READ, none.format(deliver.LAYER_WRITE_DIR, deliver.LAYER_FILE), 1],
            [
                LAYER_KEPT_READ,
                none.format(deliver.LAYER_WRITE_DIR, deliver.LAYER_KEPT),
                1,
            ],
            [LAYER_RUN, "", 0],
            [r"rk stop --drain --wait 60$", "\n".join(self.OLDER_DRAINED), 0],
            [
                r"rk stopped$",
                "stopped by rk stop --drain",
                0,
                {},
                {"after": rf"(?ms)rk stop --drain(?!.*(?:rk start$|{RESTART}))"},
            ],
            [r"rk stopped$", "", 1],
            [r"rk doctor --fresh --json$", self.doctor(), 0, {}, new],
            [r"rk doctor --fresh --json$", older_doctor, 2],
            [r"rk doctor --fresh$", report, 0],
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            *STARTED,
            TICK_SINCE_START,
            [r"rk logins --json", "[]", 0],
        ]
        return rows, made, older

    def stopped_after_a_start(self) -> dict:
        """The first start made, the container stopped since (state 12): the stand-in docker's environment."""
        stopped = self.d / "stopped"
        stopped.touch()
        return {**self.running(), "FAKE_DOCKER_STOPPED": str(stopped)}

    def stray(self):
        """A private file in the package folder that is not its own (deployment.stray_rows)."""
        (self.pkg / "docker" / "notes.yaml").write_text(
            "kind: deploy_settings\n", encoding="utf-8"
        )

    def flat(self, text: str) -> str:
        return " ".join(text.split())
