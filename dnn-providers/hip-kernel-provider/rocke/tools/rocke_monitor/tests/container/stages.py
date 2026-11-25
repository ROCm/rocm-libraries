#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container tests' stages that need Python, and the deployment the image's stages share (set_up). Every check
prints PASS, FAIL or SKIP; the last line is `STAGE <name>: PASS|FAIL`, and the exit code is 0 only when every check
passed.

The image's own stages run inside the image built from docker/Dockerfile; anywhere else each prints its SKIP line and
passes. basics, misconfigured, deploy (stage_deploy.py), reviewer, parallel-reviews and parallel-reviews-local
(stage_models.py) and stop (stage_stop.py) run there as root (RKMON_AS=root), in a home under /tmp, which the
monitor's user can read, and play the deploy machine and its container at once: set_up deploys the monitor from
tests/container/deploy.yaml as the deploy path does (the settings confirmed, the secrets set from the host, the first
start), against the fake servers, and the stage runs the monitor's commands as docker/rkmon runs them in the running
container.

  python3 -B stages.py basics         rk features, rk doctor and two ticks in prototype, which only read the fake
                                      GitHub; the image's ROCm found; with no login made (rkmon login), the logins
                                      rk doctor's only failures and the Claude session incomplete, said once by the
                                      tick, while the rest runs; no secret the deployment set in any output, log or
                                      file outside the secrets folders; the image's code as the image has it
  python3 -B stages.py misconfigured  the same deployment without the alerts' webhook: rk features shows the alerts on
                                      but incomplete, rk doctor fails on them, the tick skips them and says so once,
                                      rk teams-test refuses
  python3 -B stages.py deploy|reviewer|parallel-reviews|parallel-reviews-local|stop|rocm|atlassian|isolation|canary
                                      each in its own file (stage_deploy.py, stage_models.py, stage_stop.py,
                                      stage_rocm.py, stage_atlassian.py, stage_isolation.py, stage_canary.py), which
                                      says how it runs

The released tarball's stage changes the home it is given, so outside a container it changes nothing: it prints its
SKIP line and passes. tests/container/tarball.py runs it in a throwaway container on each base:

  python3 -B stages.py tarball --tarball FILE [--dist DIR] [--unpacked DIR]   the tarball, unpacked by this (non-root)
                                      user: every member a plain file or folder with no owner of the build machine,
                                      exactly the built folder's files, and the privacy check on what it unpacked

  --home DIR    the scratch home (default $HOME)
  --port N      the first of the four fake servers' ports (default 18701, as deploy.yaml has them);
                the stages of the other files put their model fakes at N+10 and up
"""

import argparse
import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TESTS = HERE.parent
sys.dont_write_bytecode = True
for p in (TESTS.parent, TESTS, TESTS / "fakes"):
    sys.path.insert(0, str(p))

import fake_servers  # noqa: E402
import fakerepo  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    container_layout,
    deliver,
    deploy_plan,
    deployment,
    derived,
    doctor,
    file_types,
    image,
    privacy,
    records,
    secrets_registry,
    settings_file,
)

DEPLOY_FILE = HERE / "deploy.yaml"
COMPOSE_FILE = TESTS.parent / "docker" / "compose.yaml"
ENTRY = pathlib.Path(container_layout.IMAGE_CODE) / "docker" / "entrypoint.sh"
FIRST_PORT = 18701
FAKES = 4  # the fake servers, at the first port and up: GitHub, the release, the model, JIRA (fake_servers.start)
PLANTED = {
    "first_review_pass_api_key": "sk-fake-" + "smoke-worker-key-424242",
    "teams_webhook": "https://hooks.example.invalid/" + "smoke-webhook-171717",
}
ACCESS_TOKEN = "admin_access_token"  # the admin dashboard's access token (webserve.ACCESS_SECRET), made by the host
WEBHOOK = "teams_webhook"  # the alerts' secret, which misconfigured leaves out
# rk doctor's rows of the image's ROCm
GPU_ROWS = ("GPU runtime (rocminfo)", "ROCm LLVM tools (clang)")
WRAPPER = "rkmon"  # the program a deployment's commands name, as docker/rkmon tells each (RKMON_WRAPPER)
# a value docker compose fills in: ${NAME}, ${NAME:-X}, ${NAME:?X}
FILLED = re.compile(r"\$\{(\w+)(?::([-?])(.*))?\}")
# a line of a compose service's `environment` block: NAME: VALUE
ENV_LINE = re.compile(r"(\w+): *(.*)")


class Stage:
    def __init__(self, name: str):
        self.name, self.failed = name, 0

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(
            f"{'PASS' if ok else 'FAIL'}  {what}"
            + ("" if ok or not detail else f"\n      {detail[-1500:]}"),
            flush=True,
        )
        self.failed += not ok
        return ok

    def skip(self, what: str, why: str) -> None:
        print(f"SKIP  {what} ({why})", flush=True)

    def note(self, what: str) -> None:
        """Something a reader should know (a step that works, but not as written); never a failure."""
        print(f"NOTE  {what}", flush=True)

    def end(self) -> int:
        print(f"STAGE {self.name}: {'FAIL' if self.failed else 'PASS'}", flush=True)
        return 1 if self.failed else 0


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def jsonl(path: pathlib.Path) -> list:
    return [json.loads(x) for x in read(path).splitlines() if x.strip()]


def report_row(report: str, name: str) -> tuple:
    """Row NAME of rk doctor's report or rk provision's, read by deployment.doctor_rows: (its status, its lines as one
    text, status first), or ("", "") when the report has no such row."""
    for status, row_name, lines in deployment.doctor_rows(report):
        if row_name == name:
            return status, " ".join(x.strip() for x in lines)
    return "", ""


def run(cmd, env, timeout=900, cwd=None, input=None):
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=cwd,
        input=input,
    )
    return p.returncode, p.stdout + p.stderr


def _moved(port: int, first: int) -> int:
    """PORT, a fake server's port as deploy.yaml has it (FIRST_PORT and up), moved to FIRST and up; any other port as
    it is."""
    return (
        first + port - FIRST_PORT if FIRST_PORT <= port < FIRST_PORT + FAKES else port
    )


def deploy_values(port: int = FIRST_PORT, over: dict | None = None) -> dict:
    """The stage deployment's deploy settings, as a deploy settings file in this package's layout holds them:
    deploy.yaml's keys, each value typed as the settings reader types it (settings_file.loads), with the fake servers'
    ports moved to PORT and up; the image this package builds (image.tag, the package's version); and the monitor's
    ids in the image as the container users' (host.users), as an operator who runs rkmon as root sets them, root's
    being no id they take. Then OVER's values, each one whole at its dotted key (a model binding, a feature's section).
    """
    text = re.sub(
        r"127\.0\.0\.1:(\d+)",
        lambda m: f"127.0.0.1:{_moved(int(m.group(1)), port)}",
        DEPLOY_FILE.read_text(encoding="utf-8"),
    )
    kind = file_types.DEPLOY
    header = f"kind: {kind.kind}\nschema_version: {kind.version}\n"
    values = settings_file.loads(header + text, kind).values
    values["image"] = {"tag": image.package_version()}
    monitor = container_layout.MONITOR
    values["host"] = {"users": {"monitor_uid": monitor.uid, "monitor_gid": monitor.gid}}
    for key, value in (over or {}).items():
        *path, last = key.split(".")
        node = values
        for part in path:
            node = node.setdefault(part, {})
        node[last] = value
    return values


def _filled(value: str, variables: dict) -> str:
    """VALUE as docker compose fills it in from VARIABLES: see service_environment."""
    m = FILLED.fullmatch(value)
    if not m:
        if "$" in value:
            raise ValueError(
                f"{value}: not a form this reader fills in as compose does"
            )
        return value
    name, how, rest = m.groups()
    given = str(variables.get(name, ""))
    if given:
        return given
    if how == "?":
        raise ValueError(f"{name} is not set: {rest}")
    return rest if how == "-" else ""


def service_environment(variables: dict) -> dict:
    """The environment docker compose gives the monitor's service (docker/compose.yaml: the deploy_plan.SERVICE
    service's `environment`), each value filled in from VARIABLES, a plan's (deploy_plan.DeployPlan.variables), as
    compose fills it in: ${NAME} NAME's value or none; ${NAME:-DEFAULT} NAME's value, else DEFAULT; ${NAME:?WHY}
    NAME's value, a ValueError (WHY) without one. A line of the block other than NAME: VALUE, or a value with another
    form, is a ValueError too: this reader does not read it as compose does."""
    out, top, service, inside = {}, "", "", False
    for line in COMPOSE_FILE.read_text(encoding="utf-8").splitlines():
        text = line.split(" #", 1)[0].rstrip()
        if not text.strip() or text.lstrip().startswith("#"):
            continue
        indent = len(text) - len(text.lstrip())
        if indent == 0:
            top, service, inside = text.split(":", 1)[0], "", False
        elif indent == 2:
            service = text.strip().rstrip(":") if top == "services" else ""
            inside = False
        elif indent == 4:
            inside = service == deploy_plan.SERVICE and text.strip() == "environment:"
        elif inside:
            m = ENV_LINE.fullmatch(text.strip())
            if indent != 6 or not m:
                raise ValueError(
                    f"{COMPOSE_FILE.name}: {text.strip()}: not a NAME: VALUE line of the service's environment"
                )
            out[m.group(1)] = _filled(m.group(2), variables)
    return out


class SetUpFailed(Exception):
    """A step of set_up that did not end as it ends in a deployment: WHAT, the step, and SAID, what it said."""

    def __init__(self, what: str, said: str = ""):
        super().__init__(what)
        self.what, self.said = what, said


class Deployment:
    """This container as a deployment's deploy machine and its container at once (set_up). HOME: the stage's home;
    ENVIRON: what the stage adds to every command's environment (the fake gh first on PATH after the gh counter, the
    fake servers' addresses, git's way to the fake upstream). Each step runs as docker/rkmon runs it: `confirm`,
    `secret` and `deliver` as on the deploy machine, `one_off` in a one-off container of the monitor's service, `rk`
    (and any other command line: `in_container`) in the running container. `plan`: what the confirmed settings make
    (deploy_plan.plan); `secrets`: each secret set, by name, which no output or file outside the secrets folders may
    hold; `out`: each command's words -> (its exit code, its output), in the order they ran.
    """

    def __init__(self, home: pathlib.Path, environ: dict):
        self.home, self.environ = home, environ
        self.plan, self.secrets, self.out = None, {}, {}

    def environment(self, **more: str) -> dict:
        """What a command in the monitor's service gets: the image's environment (this container's, without what the
        stage's own start adds: RKMON_AS, and the read login's folder that the entry point's shell gives a command
        typed there), the service's (service_environment, from the plan), the stage's (ENVIRON), then MORE.
        """
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("RKMON_AS", "GH_CONFIG_DIR")
        }
        env.update(service_environment(self.plan.variables))
        env.update(self.environ)
        env.update(more)
        return env

    def kept(self, words: str, rc: int, text: str) -> tuple:
        """A command's exit code RC and output TEXT, kept in `out` under WORDS (WORDS (2) for a second run): (RC,
        TEXT)."""
        name, n = words, 1
        while name in self.out:
            n += 1
            name = f"{words} ({n})"
        self.out[name] = (rc, text)
        return rc, text

    def step(self, what: str, words: str, argv: list, env: dict, input=None) -> str:
        """A step of the set-up: ARGV run, its output kept under WORDS; SetUpFailed (WHAT) unless it exits 0. Its
        output."""
        rc, text = self.kept(
            words,
            *run(argv, env, cwd=container_layout.WORK_DIR, input=input),
        )
        if rc != 0:
            raise SetUpFailed(f"{what} (exit {rc})", text)
        return text

    def confirm(self, values: dict) -> None:
        """VALUES as the deploy settings file (rkmon-deploy.yaml in the home, a private file), read and checked as
        rkmon confirm reads and checks it (deployment.read, then the rows confirm adds, deployment.confirm_rows, against
        this package; a FAIL row is SetUpFailed), then what it makes with this package (its version and its code's
        SHA-256) on this machine, as rkmon confirm makes it (`plan`).
        """
        path = self.home / "rkmon-deploy.yaml"
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, values
        )
        path.write_text(settings_file.dump(doc), encoding="utf-8")
        path.chmod(0o600)
        try:
            cfg, rows = deployment.read(str(path))
        except settings_file.Refused as exc:
            raise SetUpFailed("rkmon confirm reads the deploy settings", str(exc))
        confirmed = settings_file.read(path, file_types.DEPLOY)
        rows = [*rows, *deployment.confirm_rows(cfg, str(path), image.ROOT)]
        if checks.failed(rows):
            raise SetUpFailed(
                "rkmon confirm takes the deploy settings",
                checks.render(rows, "problems"),
            )
        host = deploy_plan.Host(os.getuid(), os.getgid(), records.timestamp())
        self.plan = deploy_plan.plan(
            confirmed, image.package_version(), image.code_sha256(), host
        )

    def secret(self, name: str, value: str) -> None:
        """NAME set to VALUE as `rkmon secret NAME` sets it from the host before the first start (docker/rkmon
        secret_set, walk_inside): rk secret-set NAME in a one-off container of the logins' walk, handed the layer the
        confirmed settings make, as the secret's keeper (the writer for a secret it keeps, else the monitor), VALUE on
        its input."""
        entry = secrets_registry.entry({}, name)
        if entry is not None and entry.keeper == secrets_registry.WRITER:
            who = {"RKMON_AS": container_layout.WRITER.name}
        else:
            who = {"RKMON_WRAPPER": WRAPPER}
        handed = {container_layout.HANDED_LAYER_VARIABLE: deliver.texts(self.plan)[2]}
        try:
            self.step(
                f"rkmon secret {name}",
                f"secret {name}",
                ["sh", str(ENTRY), "rk", "secret-set", name],
                self.environment(**who, **handed),
                input=value + "\n",
            )
        finally:  # the one-off container's own copy of the layer, which goes with it
            shutil.rmtree(container_layout.HANDED_LAYER_DIR, ignore_errors=True)
        self.secrets[name] = value

    def compose(self, args, data: bytes, variables: dict) -> int:
        """docker compose as deliver runs it (deliver.COMPOSE), here: its one-off container (deliver.LAYER_RUN, in the
        plan's compose project) runs its program as root in this container, on the settings volume's folder where the
        monitor's service mounts it (container_layout.SETTINGS_DIR) rather than where the one-off does
        (LAYER_WRITE_DIR); docker makes that folder for the volume. Its exit code."""
        if list(args) != ["-p", variables["COMPOSE_PROJECT_NAME"], *deliver.LAYER_RUN]:
            raise SetUpFailed(
                "the first start's delivery runs deliver's one-off container",
                f"docker compose {' '.join(args)}",
            )
        given = deliver.LAYER_RUN
        program = [
            given[given.index("--entrypoint") + 1],
            *given[given.index(deliver.LAYER_SERVICE) + 1 :],
        ]
        folder = container_layout.SETTINGS_DIR
        os.makedirs(folder, exist_ok=True)
        argv = [folder if a == deliver.LAYER_WRITE_DIR else a for a in program]
        done = subprocess.run(argv, input=data, capture_output=True, timeout=120)
        said = (done.stdout + done.stderr).decode("utf-8", "replace")
        return self.kept("deliver", done.returncode, said)[0]

    def deliver(self) -> None:
        """The first start's delivery (docker/rkmon first_delivery): the plan into place by deliver, its one writer
        (docker/.env, the override and the records in the home's `package` folder, the layer through `compose`), what
        it replaces kept as deliver decides (deliver.keep_previous), the layer in place read from the settings folder.
        """
        pkg = self.home / "package"
        (pkg / "docker").mkdir(parents=True, exist_ok=True)
        in_place = pathlib.Path(container_layout.SETTINGS_DIR) / deliver.LAYER_FILE
        try:
            keep = deliver.keep_previous(pkg, self.plan, lambda: (read(in_place), ""))
            deliver.deliver(self.plan, keep, package_dir=pkg, compose=self.compose)
        except (deliver.Failed, OSError, ValueError) as exc:
            raise SetUpFailed("the first start's delivery", str(exc))

    def one_off(self, *args: str) -> str:
        """The entry point's ARGS in a one-off container of the monitor's service, as docker/rkmon's first start runs
        rk there (docker compose run --rm with RKMON_WRAPPER): the root phase, then the monitor's user. Its output;
        SetUpFailed unless it exits 0."""
        words = " ".join(args)
        return self.step(
            f"{words}, in a one-off container",
            words,
            ["sh", str(ENTRY), *args],
            self.environment(RKMON_WRAPPER=WRAPPER),
        )

    def start(self) -> None:
        """The first start after its delivery (docker/rkmon cmd_start): rk provision sets the work volume up from the
        delivered settings, then rk init --if-new --on seeds the monitor's state, the monitor on, each in a one-off
        container; then rk provision --check, in the running container, finds it set up.
        """
        self.one_off("rk", "provision")
        self.one_off("rk", "init", "--if-new", "--on")
        rc, text = self.rk("provision", "--check")
        if rc != 0:
            raise SetUpFailed(
                f"rk provision --check finds the work volume set up (exit {rc})", text
            )

    def in_container(self, *argv: str) -> tuple:
        """ARGV as docker/rkmon runs a command in the running container (docker compose exec --user rkmon, with
        RKMON_WRAPPER): (its command line, as the monitor's user, and its environment).
        """
        user = container_layout.MONITOR
        line = ["setpriv", f"--reuid={user.name}", f"--regid={user.gid}"]
        line += ["--init-groups", "--", *argv]
        return line, self.environment(RKMON_WRAPPER=WRAPPER)

    def rk(self, *args: str, timeout: int = 900) -> tuple:
        """rk ARGS in the running container (in_container): (its exit code, its output), kept in `out` as `rk ARGS`."""
        line, env = self.in_container("rk", *args)
        rc, text = run(line, env, timeout=timeout, cwd=container_layout.WORK_DIR)
        return self.kept(" ".join(["rk", *args]), rc, text)


def secret_values(*left_out: str) -> dict:
    """The secrets a stage's deployment sets, by name: the planted ones (PLANTED) and the admin dashboard's access
    token, made as `rkmon secret admin_access_token --generate` makes it (secrets_registry.generate); but those named
    in LEFT_OUT."""
    token = secrets_registry.generate(secrets_registry.entry({}, ACCESS_TOKEN))
    values = dict(PLANTED, **{ACCESS_TOKEN: token})
    return {k: v for k, v in values.items() if k not in left_out}


def _give(root: pathlib.Path, user) -> None:
    """ROOT and all in it to USER (a container_layout.User) and their group."""
    for p in [root, *root.rglob("*")]:
        os.chown(p, user.uid, user.gid, follow_symlinks=False)


@contextlib.contextmanager
def set_up(
    home: pathlib.Path,
    port: int = FIRST_PORT,
    over: dict | None = None,
    secrets: dict | None = None,
    upstream=None,
):
    """A deployment of deploy.yaml in this container, as the deploy path makes one, against the fake servers at PORT
    and up, as root in the image: yields its Deployment, and stops the fakes after it.

    In the home (mode 755, under /tmp: the monitor's user reads it): the fake upstream (fakerepo.make, with a commit
    made now: fakerepo.today), the monitor's user's, as git wants a repository's owner to be; `upstream(origin)` adds
    to it and returns the fake GitHub's open PRs. The fake servers; the fake gh first on PATH after the gh counter,
    which runs it, with its log the monitor's user writes (gh.log); git's way to the fake upstream from GitHub's
    address of the repository (derived.github_address), where the monitor clones it from. Then the deploy path:
    deploy_values(PORT, OVER) confirmed, SECRETS (secret_values() by default) set from the host, the first start (its
    delivery, rk provision, rk init). A step that ends otherwise than in a deployment is SetUpFailed.
    """
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o755)
    origin = home / "upstream" / "origin"
    address = fakerepo.make(home / "upstream")
    fakerepo.today(origin)
    prs = upstream(origin) if upstream else None
    _give(home / "upstream", container_layout.MONITOR)
    urls, stop = fake_servers.start(
        home / "fakes", ports=tuple(range(port, port + FAKES)), prs=prs
    )
    try:
        tools = home / "bin"
        tools.mkdir(exist_ok=True)
        # the image's umask (027) keeps others out of a folder made here
        tools.chmod(0o755)
        shutil.copy(TESTS / "fakes" / "fake_gh", tools / "gh")
        (tools / "gh").chmod(0o755)
        gh_log = home / "gh.log"
        gh_log.touch()
        monitor = container_layout.MONITOR
        os.chown(gh_log, monitor.uid, monitor.gid)
        gh_log.chmod(0o660)
        values = deploy_values(port, over)
        counter = container_layout.GH_COUNTER_DIR
        rest = [p for p in os.environ.get("PATH", "").split(":") if p != counter]
        environ = {
            "PATH": ":".join([counter, str(tools), *rest]),
            "FAKE_GITHUB_URL": urls["github"],
            "FAKE_GH_LOGIN": values["integrations"]["github_cli"]["username"],
            "FAKE_GH_LOG": str(gh_log),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.{address}.insteadOf",
            "GIT_CONFIG_VALUE_0": derived.github_address(values["repository"]["name"]),
        }
        d = Deployment(home, environ)
        d.confirm(values)
        for name, value in (secret_values() if secrets is None else secrets).items():
            d.secret(name, value)
        d.deliver()
        d.start()
        yield d
    finally:
        stop()


def in_the_image() -> bool:
    """Whether this runs inside the image built from docker/Dockerfile: its mark (RKMON_CONTAINER=1) and its entry
    point are there."""
    return os.environ.get("RKMON_CONTAINER") == "1" and ENTRY.is_file()


def image_stage(name: str):
    """A stage of the image, BODY(stage, home, port), as a stage function (home, port) -> exit code: as root in the
    image; anywhere else its SKIP line, and it passes. A set-up step that fails is the stage's FAIL row.
    """

    def wrap(body):
        def stage(home: pathlib.Path, port: int) -> int:
            st = Stage(name)
            if not in_the_image():
                st.skip(f"stage {name}", "not inside the image (docker/Dockerfile)")
            elif os.geteuid() != 0:
                st.check(
                    False,
                    f"stage {name} runs as root in the image (RKMON_AS=root): the deploy machine and the container",
                )
            else:
                try:
                    body(st, home, port)
                except SetUpFailed as exc:
                    st.check(False, f"the deployment's set-up: {exc.what}", exc.said)
            return st.end()

        stage.__name__ = stage.__qualname__ = name
        stage.__doc__ = body.__doc__
        return stage

    return wrap


def code_state() -> tuple:
    """The image's code (container_layout.IMAGE_CODE): the path of each file and folder in it, and its SHA-256 as
    rkmon confirm takes it (image.code_sha256), which a file rewritten in place changes too.
    """
    root = pathlib.Path(container_layout.IMAGE_CODE)
    return {str(p.relative_to(root)) for p in root.rglob("*")}, image.code_sha256(root)


def code_unchanged(st: Stage, before: tuple) -> None:
    """The row that nothing was written into the image's code since BEFORE, its code_state then: no file or folder
    made or gone, and none rewritten."""
    names, sha = code_state()
    st.check(
        (names, sha) == before,
        f"nothing was written into the image's code ({container_layout.IMAGE_CODE})",
        f"made: {sorted(names - before[0])[:8]}; gone: {sorted(before[0] - names)[:8]}; SHA-256 before: "
        f"{before[1]}, after: {sha}",
    )


def common(st: Stage, d: Deployment, code: tuple) -> None:
    """What basics and misconfigured both check, after the set-up: rk features, rk doctor and two ticks, in the running
    container; the ticks read the fake GitHub and only read it; no gh call wrote or failed; and the image's code is as
    it was before the set-up (code_unchanged, CODE its code_state then)."""
    for args in (("features",), ("doctor",), ("tick",), ("tick",)):
        d.rk(*args)
    rc, text = d.out["rk features"]
    st.check(
        rc == 0
        and "**Features**" in text
        and "| Capability | State | Runs while | Configured with | Missing |" in text,
        "rk features prints the features' summary and their table",
        text,
    )
    for name in ("rk tick", "rk tick (2)"):
        st.check(d.out[name][0] == 0, f"{name} completes", d.out[name][1])
    st.check(
        "reviews=prototype merge-queue=prototype" in d.out["rk tick"][1],
        "the tick ran with both features in prototype",
        d.out["rk tick"][1],
    )
    gh_log = jsonl(d.home / "fakes" / "github.log")
    st.check(
        bool(gh_log)
        and all(r["method"] == "GET" or r["path"] == "/graphql" for r in gh_log),
        "the ticks read the fake GitHub, and only read it",
    )
    calls = jsonl(d.home / "gh.log")
    st.check(
        bool(calls) and all(c["outcome"] == "ok" for c in calls),
        "no gh call wrote or failed",
        json.dumps([c for c in calls if c["outcome"] != "ok"])[:600],
    )
    code_unchanged(st, code)


def secrets_absent(st: Stage, d: Deployment) -> None:
    """No secret the deployment set (d.secrets) in any output (d.out), nor in any file outside the secrets folders:
    the work volume, the delivered settings and the stage's home (the deploy settings file, the package folder's
    files, the fakes' logs and the fake gh's)."""
    texts = {f"the output of {k}": v[1] for k, v in d.out.items()}
    roots = (container_layout.WORK_DIR, container_layout.SETTINGS_DIR, d.home)
    for root in roots:
        for f in pathlib.Path(root).rglob("*"):
            if f.is_file() and ".git" not in f.parts and f.stat().st_size < 4_000_000:
                texts[str(f)] = f.read_bytes().decode("utf-8", "replace")
    leaks = [w for w, t in texts.items() if any(s in t for s in d.secrets.values())]
    st.check(
        not leaks,
        "no secret or login the deployment set appears in any output, log or file outside the secrets folders",
        ", ".join(leaks),
    )


def doctor_fails(d: Deployment) -> dict:
    """rk doctor's FAIL rows in d.out (deployment.doctor_rows): name -> its lines as one text."""
    return {
        name: " ".join(x.strip() for x in lines)
        for status, name, lines in deployment.doctor_rows(d.out["rk doctor"][1])
        if status == "FAIL"
    }


def skipped_once(st: Stage, d: Deployment, feature: str) -> None:
    """The first tick reports FEATURE, which is on but blocked, as skipped; the second, which goes on as the first
    did, does not say it again."""
    line = f"FEATURE_SKIPPED {feature}"
    first, second = d.out["rk tick"][1], d.out["rk tick (2)"][1]
    st.check(
        line in first and line not in second,
        f"the tick skips the {feature}, says so once and goes on",
        first[-1500:],
    )


@image_stage("basics")
def basics(st: Stage, home: pathlib.Path, port: int) -> None:
    """The deployment of deploy.yaml, every secret set and no login made: what `common` checks; the image's GPU runtime
    and ROCm tools found, so the develop guard is on; the logins (rkmon login) rk doctor's only failures; and the
    Claude session, which needs a login, on but incomplete while the rest runs."""
    code = code_state()
    with set_up(home, port) as d:
        common(st, d, code)
        report, feats = d.out["rk doctor"][1], d.out["rk features"][1]
        found = [report_row(report, name) for name in GPU_ROWS]
        st.check(
            all(status == "PASS" for status, _ in found)
            and "| Develop guard | on |" in feats,
            "rk doctor finds the image's GPU runtime and ROCm LLVM tools, and the develop guard is on",
            " / ".join(text for _, text in found),
        )
        fails = doctor_fails(d)
        st.check(
            all("rkmon login" in text for text in fails.values()),
            "rk doctor fails only on the logins no stage makes: each failure's fix is rkmon login",
            " / ".join(fails.values()),
        )
        st.check(
            "| Claude session | on (incomplete) |" in feats
            and "| Reviews | on |" in feats
            and "| Merge plan | on |" in feats,
            "the missing Claude login blocks the Claude session only: rk features shows it on but incomplete, the "
            "reviews and the merge plan on",
            feats[:1500],
        )
        skipped_once(st, d, "Claude session")
        secrets_absent(st, d)


@image_stage("misconfigured")
def misconfigured(st: Stage, home: pathlib.Path, port: int) -> None:
    """The same deployment without the alerts' webhook (teams_webhook): the alerts on but incomplete in rk features,
    with what they need; rk doctor fails on them, else only on the logins; the tick skips them and says so once; rk
    teams-test refuses and names what is missing."""
    code = code_state()
    with set_up(home, port, secrets=secret_values(WEBHOOK)) as d:
        common(st, d, code)
        feats = d.out["rk features"][1]
        st.check(
            "| Develop break alerts | on (incomplete) | features.alerts.develop_breaks.enabled is true |"
            in feats
            and f"needs the channel's incoming-webhook URL (the secret {WEBHOOK}"
            in feats,
            "rk features shows the alerts on but incomplete, with what is missing",
            feats,
        )
        fails = doctor_fails(d)
        st.check(
            any(f"rkmon secret {WEBHOOK}" in text for text in fails.values())
            and all(
                f"rkmon secret {WEBHOOK}" in text or "rkmon login" in text
                for text in fails.values()
            ),
            f"rk doctor fails on the alerts, whose fix is rkmon secret {WEBHOOK}, and else only on the logins",
            " / ".join(fails.values()),
        )
        skipped_once(st, d, "develop break alerts")
        rc, text = d.rk("teams-test")
        st.check(
            rc == 5 and "NOT SET UP: develop break alerts is on but needs" in text,
            "rk teams-test refuses with exit 5 and names what is missing",
            text,
        )
        secrets_absent(st, d)


STAGES = {"basics": basics, "misconfigured": misconfigured}

TAR_ROOT = "rocke_monitor"


def tarball_members(
    st: Stage, tarball: pathlib.Path, dist: pathlib.Path | None
) -> None:
    """What the tarball has: every member a plain file or folder under rocke_monitor/, with no owner or group
    name and no uid or gid but 0 (nothing of the machine that built it), no write bit for the group or others; and, with
    the built folder at hand, exactly its files, byte for byte."""
    import hashlib
    import tarfile

    with tarfile.open(tarball) as t:
        members = t.getmembers()
        files = {
            m.name.split("/", 1)[1]: hashlib.sha256(t.extractfile(m).read()).hexdigest()
            for m in members
            if m.isfile() and "/" in m.name
        }
    odd = [
        m.name
        for m in members
        if not (m.isfile() or m.isdir())
        or m.name.startswith("/")
        or ".." in m.name.split("/")
        or not (m.name == TAR_ROOT or m.name.startswith(TAR_ROOT + "/"))
    ]
    st.check(
        not odd,
        "every member is a plain file or folder under rocke_monitor/",
        ", ".join(odd[:10]),
    )
    owners = sorted(
        {f"{m.uname or '-'}/{m.gname or '-'} {m.uid}:{m.gid}" for m in members}
    )
    st.check(
        all(
            m.uname in ("", "root")
            and m.gname in ("", "root")
            and m.uid == 0
            and m.gid == 0
            for m in members
        ),
        "no member carries the build machine's owner: no user or group name, uid and gid 0",
        (
            f"owners in the tarball: {len(owners)} distinct (names not shown)"
            if owners
            else ""
        ),
    )
    loose = [m.name for m in members if m.mode & 0o022]
    st.check(
        not loose, "no member is writable by the group or others", ", ".join(loose[:10])
    )
    if dist is None:
        st.skip("the tarball has exactly the built folder's files", "no --dist given")
        return
    built = {
        str(p.relative_to(dist))
        .replace(os.sep, "/"): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in dist.rglob("*")
        if p.is_file()
    }
    extra, gone = sorted(set(files) - set(built)), sorted(set(built) - set(files))
    differ = sorted(k for k in set(files) & set(built) if files[k] != built[k])
    st.check(
        not extra and not gone and not differ,
        f"the tarball has exactly the built folder's {len(built)} files, byte for byte",
        f"only in the tarball: {extra[:8]}; only in the folder: {gone[:8]}; different: {differ[:8]}",
    )


def tarball(
    home: pathlib.Path,
    tarball_path: pathlib.Path,
    dist: pathlib.Path | None,
    unpacked: pathlib.Path | None,
) -> int:
    """The released tarball: its members (tarball_members), then the privacy check on the unpacked folder, with an
    invented deny list outside it, whose value no file contains. `unpacked`: where this user already unpacked it
    (tar -xzf), else here."""
    st = Stage("tarball")
    tarball_members(st, tarball_path, dist)
    if unpacked is None:
        (home / "install").mkdir(parents=True, exist_ok=True)
        code, text = run(
            ["tar", "-xzf", str(tarball_path), "-C", str(home / "install")],
            dict(os.environ),
        )
        st.check(code == 0, "the tarball unpacks as this user", text[-800:])
        unpacked = home / "install" / TAR_ROOT
    pkg = pathlib.Path(unpacked)
    deny = (
        home / "invented_deny.txt"
    )  # the check requires a deny list; this value is joined here, so no file contains it
    deny.write_text("tarball" + "-stage-canary-x\n", encoding="utf-8")
    code, text = run(
        privacy.command("--deny", deny, "."),
        dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        cwd=pkg,
    )
    st.check(
        code == 0 and "Result: PASS" in text,
        "the privacy check passes on the unpacked folder",
        text[-2500:],
    )
    return st.end()


IMAGE_FILES = (
    "stage_deploy",
    "stage_models",
    "stage_stop",
    "stage_rocm",
    "stage_atlassian",
    "stage_isolation",
    "stage_canary",
)  # each says where it runs


def skipped_here(stage: str) -> int:
    """A stage that runs in a throwaway container, outside one: it would change the home it is given, so it changes
    nothing and passes as skipped."""
    st = Stage(stage)
    st.skip(
        f"stage {stage}",
        "not inside a container: tarball.py runs it in a throwaway one",
    )
    return st.end()


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    image_stages = dict(STAGES)
    for mod in IMAGE_FILES:
        image_stages.update(__import__(mod).STAGES)
    ap.add_argument("stage", choices=sorted([*image_stages, "tarball"]))
    ap.add_argument("--home", default=os.path.expanduser("~"))
    ap.add_argument("--port", type=int, default=FIRST_PORT)
    ap.add_argument("--tarball", help="tarball: the released tarball")
    ap.add_argument("--dist", help="tarball: the built folder it was made from")
    ap.add_argument("--unpacked", help="tarball: where this user unpacked it already")
    a = ap.parse_args(argv)
    if a.stage not in image_stages and not doctor.in_container():
        return skipped_here(a.stage)
    home = pathlib.Path(a.home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    if a.stage == "tarball":
        if not a.tarball:
            ap.error("tarball needs --tarball FILE")
        return tarball(
            home,
            pathlib.Path(a.tarball),
            pathlib.Path(a.dist) if a.dist else None,
            pathlib.Path(a.unpacked) if a.unpacked else None,
        )
    return image_stages[a.stage](home, a.port)


if __name__ == "__main__":
    sys.exit(main())
