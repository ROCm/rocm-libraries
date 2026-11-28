# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's several-deployments stage (tests/container/image.py): several deployments on one host (check 15).
No command of one reads another, nothing refuses a second deployment of a repository,
and a start refuses another deployment's container name, which is its compose project, another deployment's volume,
and a busy port.

Each deployment is the deploy path's (tests/container/stage_deploy_path.py, DeployPath: the guide's steps on a fresh
machine, each checked, its results under this stage's name and "deployment X"), with a package folder, a settings
file and an admin dashboard port of its own, and but for c a container name, stand-ins for GitHub and the Claude
gateway and an image that trusts them of its own, all under one account: a's GitHub login, with a's token, and a's
key. Its objects are named after the scope <names>-several-deployments-X, X its letter:
- a and b, two deployments of one repository, each walk the whole path, b's while a runs: nothing refuses the second,
  and a's container is the same one, still running, after it;
- c, whose container.name is a's, takes a's compose project with it, and with the project the objects docker names
  after it, its network and the stand-ins on it among them, as two folders of one container name do; so it takes a's
  image too, which trusts those stand-ins, as a host's image trusts what its network reaches (Deployment.prepare). c
  is confirmed before a starts; once a runs, its start refuses, as the container name is used by another deployment on
  this host (NAME_REFUSED), with the fix that sets container.name and the next line that gives confirm, and a's
  container runs on, the same one;
- d walks to its logins; then its rkmon-code volume, by the name docker compose gives it in d's compose project, is
  made again as another deployment's (a's compose project in its label), and d's start refuses with that volume's
  FAIL row; that volume removed, the stage itself listens on d's admin dashboard port, and d's start refuses with the
  port's FAIL row. After each refusal, d has no container and no docker/.env: nothing changed.

No command of one deployment reads another (one_reads_none). Docker shows each deployment the host as it is, with the
others on it: its docker guard (tests/fakes/docker_guard) takes the stage's scope for its own, so that each listing
shows every deployment's objects. Then no docker command of a deployment names another's object (its container,
compose project, volume or network, by its name, by a container's ID, or in a listing's filter), runs in another's
folder or names one. An image is the host's, not a deployment's: each one's guard names the images it may use. c's
compose project is a's: until a's container starts, the objects named after it are c's as much as a's; from then on,
c may name them only in a listing (LISTINGS): docker ps filtered exactly on the name or on its compose project, as
check 15 reads it, or compose ps, ls or config (others_findings). No Python process of a deployment's commands opens
a file or lists a folder in another's folder (tests/fakes/folder_reads, which logs it, first on each command's
PYTHONPATH), and no output of its commands shows another's folder. Besides, as in the deploy path, no docker command
was refused, and before confirm none acted on or read an object but through listings filtered exactly on the
deployment's own name; and a and b read GitHub with the read login, cloned the repository, reached the gateway with
the key, and wrote nothing.

Then the end, each deployment by its own `rkmon remove --delete-data --yes` (DeployPath.clean): b, a, then c, whose
container name is a's, then d.
"""

import json
import math
import os
import re
import shlex
import socket

import hostrun
import stage_deploy_path
from stage_deploy_path import (
    STEP_TIMEOUT_S,
    command_label,
    during,
    next_command,
    next_line,
    says,
)

STAGE = "several-deployments"
# The volume of d's that the stage makes again as another deployment's, by its key in docker/compose.yaml
VOLUME = "rkmon-code"
# The outcome lines of the start's refusals, their beginnings, and their exit code
NAME_REFUSED = "REFUSED: the container name {} is used by another deployment on this host; nothing changed"
VOLUME_REFUSED = "REFUSED: a volume this deployment would use is another deployment's;"
PORT_REFUSED = "REFUSED: a port the dashboards need is in use;"
REFUSED_EXIT = 4
# A container's ID, as docker gives it in full or shortened
CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")
# The listings, as the docker guard's log names the command, in which a deployment may name the compose project it
# shares with another once that one's container runs (others_findings)
LISTINGS = ("ps", "compose ps", "compose ls", "compose config")


def others_findings(
    calls: list,
    marks: list,
    own: tuple,
    others: dict,
    ids: dict,
    shared_until: float = math.inf,
) -> list:
    """Each docker call of CALLS, the guard's log lines of one deployment, that reads another (the module's docstring),
    with the rkmon command it ran in (MARKS). OWN: (the deployment's scope, its container name, which is its compose
    project); OTHERS: each other deployment's scope -> its folder; IDS: each container's full ID -> its name, by which
    an ID is read. A container name that is another deployment's names that one's objects as its own compose project's
    until SHARED_UNTIL, when that one's container starts; from then on, only exactly, as itself or as its compose
    project, and only in a listing (LISTINGS): any other command on it is a finding. An image is the host's, not a
    deployment's: the guard names the images it may use.
    """
    from rkmon_lib import identity  # noqa: E402

    scope, name = own
    exact = (["project", name], ["name", identity.exact_name(name)])
    bad = []
    for r in calls:
        said = f"during {during(r['t'], marks)}: docker {shlex.join(r['argv'])}"
        mine = (scope, name) if r["t"] < shared_until else (scope,)
        named = []
        for kind, word in r.get("names") or []:
            if CONTAINER_ID.fullmatch(word):
                word = next((n for i, n in ids.items() if i.startswith(word)), word)
            if kind == "image" or ([kind, word] in exact and r["command"] in LISTINGS):
                continue
            if not any(m in word for m in mine):
                named += [f"the {kind} {word}" for o in others if o in word][:1]
        folders = [
            f for f in others.values() if any(f in a for a in [r["cwd"], *r["argv"]])
        ]
        named += [f"runs in or names the folder {f}" for f in folders]
        bad += [f"{said}: {x}" for x in named[:1]]
    return bad


def read_findings(lines: list, marks: list, folders: dict) -> list:
    """Each file opened or folder listed of LINES, a deployment's folder reads (tests/fakes/folder_reads), in a folder
    of FOLDERS (path -> whose), with the rkmon command it ran in (MARKS); and, when its log has fewer starts of the
    program than MARKS has rkmon commands, that it did not run in each of them."""
    bad = []
    for r in lines:
        if "path" in r:
            whose = next(
                (w for f, w in folders.items() if r["path"].startswith(f)), "?"
            )
            bad.append(
                f"during {during(r['t'], marks)}: {r['event']} {r['path']}, in deployment {whose}'s folder"
            )
    started = [
        r
        for r in lines
        if any(str(a).endswith("rkmon") for a in r.get("started") or [])
    ]
    if len(started) < len(marks):
        bad.append(
            f"the folder reads' log saw {len(started)} starts of ./rkmon, for {len(marks)} commands: it did not run "
            "in each"
        )
    return bad


class Deployment(stage_deploy_path.DeployPath):
    """One deployment of the stage (the module's docstring): LETTER its own; ACCOUNT the deployment whose GitHub token
    and key it takes (one account), none for the first; NAME the deployment whose container name it takes, none for
    its own. Its results, the deploy path's and its own, go under the stage's name and its letter (result).
    """

    def __init__(
        self,
        engine: str,
        image: str,
        pkg: str,
        names: str,
        run_opts: list,
        res,
        letter: str,
        account=None,
        name=None,
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, f"{STAGE}-{letter}")
        self.letter = letter
        # the stage's scope, which each deployment's begins with
        self.shared = self.scope[: -len(letter)]
        self.owner = name  # the deployment whose container name it takes, if any
        self.name = name.scope if name is not None else self.scope
        if account is not None:
            self.token, self.key = account.token, account.key
        self.others = []  # the stage's other deployments (watch)
        self.outputs = []  # (each rkmon command, as command_label names it, its output)

    def result(self, ok: bool, text: str) -> bool:
        """The result TEXT, under the stage's name and the deployment's letter: PASS when OK, else FAIL. Returns OK."""
        self.res.result(
            "PASS" if ok else "FAIL", f"{STAGE}: deployment {self.letter}, {text}"
        )
        return ok

    def prepare(self) -> None:
        """The deploy path's fresh machine (DeployPath.prepare), whose docker guard then takes the stage's scope for its
        own, with the folder reads first on each command's PYTHONPATH, and their log (watch names the folders). A
        deployment that takes another's container name (owner) has that one's network, stand-ins and image in place of
        the ones made for it here, which are removed (the module's docstring): the owner is prepared first.
        """
        super().prepare()
        guard = json.loads((self.top / "guard.json").read_text(encoding="utf-8"))
        guard["scope"] = self.shared
        if self.owner is not None:
            self.must("rm", "-f", self.fakes)
            self.must("network", "rm", self.network)
            self.must("rmi", self.derived)
            self.made_image = False
            guard["network"] = self.owner.network
            guard["images"] = [self.owner.derived]
        (self.top / "guard.json").write_text(json.dumps(guard), encoding="utf-8")
        hook = os.path.join(self.pkg, "tests", "fakes", "folder_reads")
        self.env["PYTHONPATH"] = os.pathsep.join(
            [hook, *filter(None, [self.env.get("PYTHONPATH")])]
        )
        self.env["FOLDER_READS_LOG"] = str(self.top / "folder_reads.log")

    def watch(self, others: list) -> None:
        """OTHERS, the stage's other deployments: the folder reads watch their folders."""
        self.others = list(others)
        self.env["FOLDER_READS_WATCH"] = os.pathsep.join(str(o.top) for o in others)

    def rkmon(
        self, *args: str, stdin: bytes = b"", timeout: float = STEP_TIMEOUT_S
    ) -> tuple:
        """DeployPath.rkmon, its output kept (outputs)."""
        rc, out = super().rkmon(*args, stdin=stdin, timeout=timeout)
        self.outputs.append((command_label(args), out))
        return rc, out

    def fill_in(self) -> None:
        """The deploy path's values (DeployPath.fill_in); for a deployment that takes another's container name, that
        name and that one's image (prepare)."""
        super().fill_in()
        if self.owner is None:
            return
        from rkmon_lib import file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        v = doc.values
        v["container"]["name"] = self.name
        v["image"]["name"], _, v["image"]["tag"] = self.owner.derived.rpartition(":")
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    def started_at(self) -> float:
        """When its container first started: its first `./rkmon start` began (its marks); never (math.inf) before."""
        return next((t for t, what in self.marks if what == "./rkmon start"), math.inf)

    def admin_port(self) -> int:
        from rkmon_lib import file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        return int(doc.values["features"]["dashboards"]["admin"]["port"])

    def container_id(self) -> str:
        """The full ID of the deployment's container (DeployPath.monitor), while it runs."""
        return self.must("inspect", "--format", "{{.Id}}", self.monitor()).strip()

    def changed(self) -> list:
        """What a refused start of this deployment changed: a container of its compose project, or a docker/.env."""
        from rkmon_lib import identity  # noqa: E402

        found = []
        listed = self.must(
            "ps", "--all", "-q", "--filter", f"label={identity.PROJECT}={self.scope}"
        ).split()
        if listed:
            found.append(f"{len(listed)} containers of its compose project")
        if (self.copy / "docker" / ".env").exists():
            found.append("a docker/.env")
        return found

    # --- c: another deployment's container name ----------------------------------------------------------------------
    def name_refused(self, a) -> bool:
        """`rkmon start` in this folder, whose container name is A's: refused, as the module's docstring says."""
        before = a.container_id()
        rc, out = self.rkmon("start")
        found = []
        if rc != REFUSED_EXIT or not says(out, NAME_REFUSED.format(a.scope)):
            found.append(f"start: exit {rc}, no line {NAME_REFUSED.format(a.scope)!r}")
        if not says(out, "fix: set container.name in"):
            found.append("start: no fix that sets container.name")
        if next_command(next_line(out))[0] != 3:
            found.append(f"start: next line {next_line(out)!r}")
        if a.container_id() != before:
            found.append("a's container is not the one that ran before")
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "whose container name is a's, its compose project: its start refuses, as the container name is used by "
            "another deployment on this host, with the fix that sets container.name and the next line that gives "
            "confirm, and a's container runs on, the same one",
        )

    # --- d: another deployment's volume, then a busy port ------------------------------------------------------------
    def volume_refused(self, a) -> bool:
        """`rkmon start` in this folder, one of whose volumes is made again as A's (the module's docstring): refused,
        nothing changed; then that volume removed, and before it, when the start went through, the containers of the
        deployment's compose project, one of which uses it."""
        from rkmon_lib import identity  # noqa: E402

        volume = f"{self.scope}_{VOLUME}"
        self.docker("volume", "rm", "--force", volume)
        self.must(
            "volume",
            "create",
            "--label",
            f"{identity.PROJECT}={a.scope}",
            "--label",
            f"com.docker.compose.volume={VOLUME}",
            volume,
        )
        try:
            rc, out = self.rkmon("start")
            left = self.changed()
        finally:
            if self.docker("volume", "rm", volume)[0] != 0:
                listed = self.must(
                    "ps",
                    "--all",
                    "-q",
                    "--filter",
                    f"label={identity.PROJECT}={self.scope}",
                )
                self.must("rm", "--force", *listed.split())
                self.must("volume", "rm", volume)
        row = f"{volume} is used by another deployment on this host"
        found = [] if says(out, row) else [f"start: no FAIL row {row!r}"]
        if rc != REFUSED_EXIT or not says(out, VOLUME_REFUSED):
            found.append(f"start: exit {rc}, no line {VOLUME_REFUSED!r}")
        found += [f"after the start: {x}" for x in left]
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "its start refuses a volume of its compose project that is another deployment's, with its FAIL row, and "
            "changes nothing",
        )

    def port_refused(self) -> bool:
        """`rkmon start` in this folder while the stage itself listens on its admin dashboard port: refused, nothing
        changed."""
        from rkmon_lib import container_layout  # noqa: E402

        port = self.admin_port()
        with socket.socket() as s:
            s.bind((container_layout.LOOPBACK, port))
            s.listen()
            rc, out = self.rkmon("start")
        left = self.changed()
        row = f"{container_layout.LOOPBACK}:{port} is in use by another program or container"
        found = [] if says(out, row) else [f"start: no FAIL row {row!r}"]
        if rc != REFUSED_EXIT or not says(out, PORT_REFUSED):
            found.append(f"start: exit {rc}, no line {PORT_REFUSED!r}")
        found += [f"after the start: {x}" for x in left]
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "its start refuses an admin dashboard port another program listens on, with its FAIL row, and changes "
            "nothing",
        )

    # --- what each deployment did ------------------------------------------------------------------------------------
    def calls(self) -> list:
        """The guard's log: a line for each docker command of the deployment's."""
        log = self.top / "guard.log"
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        return [json.loads(x) for x in text.splitlines()]

    def docker_calls(self) -> None:
        """The deploy path's checks of the deployment's docker commands, on the stage's scope: none refused; before
        confirm, none that acted on or read an object but through listings filtered exactly on its own name, its
        container name (stage_deploy_path.before_confirm_findings)."""
        calls = self.calls()
        refused, counts = stage_deploy_path.guard_findings(calls)
        early = stage_deploy_path.before_confirm_findings(calls, self.marks, self.name)
        said = [
            *refused,
            *early,
            f"docker commands: {json.dumps(counts, sort_keys=True)}",
        ]
        hostrun.indented("\n".join(said))
        self.result(
            not refused and not early,
            "its docker commands: none refused, and before confirm none acted on or read an object but through "
            "listings filtered exactly on its own name",
        )

    def one_reads_none(self, ids: dict) -> bool:
        """No command of the deployment read another (the module's docstring): others_findings on its guard's log, IDS
        the containers' full IDs, its compose project shared with the deployment whose container name it takes until
        that one's container starts; read_findings on its folder reads; and its outputs.
        """
        log = self.top / "folder_reads.log"
        lines = (
            [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
            if log.exists()
            else []
        )
        others = {o.scope: str(o.top) for o in self.others}
        until = self.owner.started_at() if self.owner is not None else math.inf
        found = others_findings(
            self.calls(), self.marks, (self.scope, self.name), others, ids, until
        )
        found += read_findings(
            lines,
            self.marks,
            {os.path.join(str(o.top), ""): o.letter for o in self.others},
        )
        found += [
            f"{label} shows deployment {o.letter}'s folder"
            for label, out in self.outputs
            for o in self.others
            if str(o.top) in out
        ]
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "none of its commands read another deployment: no docker command named another's object or folder, no "
            "file or folder of another's was opened or listed, and no output showed its folder",
        )


def container_ids(t: Deployment) -> dict:
    """Each container of the stage's scope, its full ID -> its name."""
    out = t.must(
        "ps",
        "--all",
        "--no-trunc",
        "--format",
        "{{.ID}} {{.Names}}",
        "--filter",
        f"name=^{t.shared}",
    )
    return dict(x.split(" ", 1) for x in out.splitlines() if " " in x)


def side_by_side(a: Deployment, b: Deployment) -> bool:
    """B's whole path while A runs: nothing refuses it, and A's container runs on, the same one."""
    before = a.container_id()
    b.path()
    ok = b.done and a.container_id() == before
    return b.result(
        ok,
        "of the same repository as a, walks the whole path while a runs, and a's container runs on, the same one",
    )


def several_deployments_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    args = (engine, image, pkg, names, run_opts, res)
    a = Deployment(*args, "a")
    b = Deployment(*args, "b", account=a)
    c = Deployment(*args, "c", account=a, name=a)
    d = Deployment(*args, "d", account=a)
    every = (a, b, c, d)
    try:
        for t in every:
            t.prepare()
        for t in every:
            t.watch([o for o in every if o is not t])
        named = c.path(last=3) == 3
        walked = d.path(last=5) == 5
        if a.path() >= 6 and a.done:
            side_by_side(a, b)
            if named:
                c.name_refused(a)
            if walked and d.volume_refused(a):
                d.port_refused()
        for t in (a, b):
            t.after()
        for t in (c, d):
            t.docker_calls()
        ids = container_ids(a)
        for t in every:
            t.one_reads_none(ids)
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        for t in (b, a, c, d):
            t.clean()
