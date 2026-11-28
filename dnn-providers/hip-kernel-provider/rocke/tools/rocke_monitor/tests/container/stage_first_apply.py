# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's first-apply stage (tests/container/image.py): the first apply over an older package,
a failed one, and the way back, on a deployment the earlier package made, moved in place. It needs an image of the
package before this one (image.py --earlier-image IMAGE), and is skipped without it.

The deployment is first the earlier package's, as a person has it before the move: its package folder, copied out of the
earlier image from the folder its entry point names (code_folder) and writable by you, with the docker/.env its settings
step writes (older_env); its read login set by its own `rkmon login gh-read --with-token`; config.json and the monitor's
clone, which its own `rkmon setup --settings FILE` writes in the work volume, its sections EARLIER_SECTIONS, from a
deploy settings file in its layout (earlier_settings); and its monitor started by its own `rkmon start`. Each image runs
as a build on top of it that trusts the stand-ins for GitHub and the Claude gateway, on an internal network
(DeployPath.prepare), and every docker command of either package goes through the guard (tests/fakes/docker_guard),
which lets both images run.

In order:
- The move in place: this package's `rkmon package update --from NEW --folder FOLDER` puts it in the earlier
  package's folder, which it keeps beside it as FOLDER.prev; then template, check, confirm and login in the folder, the
  earlier monitor running, with the deploy path's values (DeployPath.fill_in) and the earlier deployment's admin port,
  which the move keeps. Confirm's next line is shown; login's is apply's (CHANGE_WAITS), as state 7 reads this
  package's logins while the earlier monitor runs.
- The first apply, whose first load fails: the stage plants an internal state that is not JSON (PLANTED: an invented
  fault, as an older package writes none), which the new monitor's load refuses, moving nothing, and a stand-in for the
  earlier package's GitHub write login (WRITE_LOGIN), as that package keeps one. ERROR, exit 1, with both ways forward
  and state 10's next line; the new container stopped; `rkmon next` gives state 10's line; `rkmon rollback` says NOTHING
  TO DO with the way back to the earlier package. The earlier package's files are kept (kept_findings, tree_findings):
  its folder, FOLDER.prev, as the update left it; in the volumes, each file of the monitor's home and of the two secrets
  volumes, with its owner and mode, config.json, the secrets, the write login and the planted file byte for byte
  (kept_exactly), and any other file, the earlier monitor's state, last written before the new container was made, since
  that monitor's own drain and stop write it. Three changes are allowed, each shown when it is found: a mode in the work
  volume that the container's layout gives it (layout_mode), which this package's start gives an older volume once; the
  state's stop mark and journal (STOP_WRITES), which the new container's own stop writes; and the older dashboard's pid
  file gone (DASHBOARD_PID), which that dashboard removes when it stops.
- The way back: `rkmon stop`, `rkmon package update --from FOLDER.prev`, and the earlier package's `./rkmon
  start`: its folder is as it was, and its monitor runs again (earlier_runs).
- The way forward again: the planted file moved aside, as its fix says, and the earlier package's doctor, for the
  checks it fails; then `package update --from FOLDER.prev --folder FOLDER` by the rkmon of the package's copy the
  move began from, which puts this package back with its records (the earlier package's rkmon, FOLDER's own now, has
  no package update); `rkmon next` gives state 10's line, and `rkmon apply` makes the first apply over the running
  earlier monitor: that package's doctor, its failing rows shown as it gives them, its drain and the load; APPLIED,
  in standby, its next line first-tick's or, once that tick has run, standby off's (applied_next), and the new image
  runs. Then the new monitor's `rk stopped`, in its container, finds no stop mark left in its state, the earlier
  drain's cleared (stop_left), and `rkmon first-tick` waits for the new monitor's first tick, in standby, its next
  line standby off's.
- The way back from standby, as above: the earlier monitor runs again on the state the new monitor wrote in standby.
  What it makes of that state is shown, not judged: each file the two monitors wrote
  since the earlier one stopped; then its `rkmon status` and a tick (`rkmon rk tick`), each with its exit code, and
  the modes of what it wrote meanwhile, beside each file's before the first apply and at the way back
  (written_modes); its doctor's exit code and the checks it fails, beside those it failed before that first apply;
  and its log's lines that name an error.
Each check that the next ones need ends the stage when it fails. Then what both packages did (after): the stand-ins'
log, as the deploy path reads it, and the docker commands, each the deployment's own.

Its objects are named after the scope <names>-first-apply, the earlier image's build <name>:<scope>-earlier among them,
and removed at the end by name (DeployPath.clean): by then the folder holds the earlier package, and the deployment's
own remove is the deploy-path stage's to check.
"""

import contextlib
import datetime
import hashlib
import io
import json
import os
import pathlib
import re
import runpy
import shlex
import stat
import subprocess
import tarfile

import hostrun
import stage_deploy_path
from stage_deploy_path import (
    LABEL,
    LOGIN,
    OUTCOMES,
    PROFILE,
    REPOSITORY,
    free_port,
    next_line,
    outcome,
    step_timeout,
)

STAGE = "first-apply"
# The deployment's package folder, and where package update keeps the one it replaces
FOLDER = "monitor"
PREV = f"{FOLDER}.prev"
# The volumes of the earlier monitor's files, by the names compose gives them in the deployment's project, and where
# the container mounts each: the monitor's home and config.json in the first, the secrets in the second, the GitHub
# write login alone in the third; and the folders of them the stage reads (volume_files)
VOLUMES = {
    "rkmon-work": "/work",
    "rkmon-secrets": "/secrets",
    "rkmon-secrets-write": "/secrets-write",
}
HOME = "/work/monitor"
CONFIG_JSON = f"{HOME}/config.json"
SECRETS = VOLUMES["rkmon-secrets"]
SECRETS_WRITE = VOLUMES["rkmon-secrets-write"]
READ = (HOME, SECRETS, SECRETS_WRITE)
# A stand-in for the earlier package's GitHub write login, which the stage plants as that package keeps one (its
# docker/entrypoint.sh): gh's hosts.yml in a folder of the write user's (rkwrite), the folder 700 and the file 600; its
# bytes invented, with no login in them, so that nothing reads it as one
WRITE_LOGIN = f"{SECRETS_WRITE}/gh/hosts.yml"
WRITE_LOGIN_TEXT = "# a stand-in for the GitHub write login of the earlier package: invented bytes, no login in them\n"
# What a failed first apply may change of the older monitor's files, each shown when it does (kept_findings): in the
# work volume (WORK), a mode this package's start gives it once (layout_mode); the state's stop mark and
# journal (STOP_WRITES), which the new container's own stop writes (docker/entrypoint.sh's stop runs rk stop); and the
# older dashboard's pid file (DASHBOARD_PID), which that dashboard removes when it stops
WORK = VOLUMES["rkmon-work"]
STOP_WRITES = (f"{HOME}/state/state.json", f"{HOME}/state/journal.md")
DASHBOARD_PID = f"{HOME}/state/web.pid"
# The fault the stage plants for the first apply's load: the new monitor's internal state (rkmon_lib/state_store.py),
# not JSON; and where its fix moves it
PLANTED = f"{HOME}/state/internal_state.json"
ASIDE = f"{PLANTED}.old"
# The deploy settings file the earlier package's setup reads, beside the deployment's folder, and the sections of
# that setup its monitor needs to run: config.json (settings) and the monitor's clone (deploy)
EARLIER_SETTINGS = "earlier-settings.json"
EARLIER_SECTIONS = ("settings", "deploy")
# The earlier package's docker/.env, in its folder, and the mode its settings step writes it with: its writer's default
ENV = "docker/.env"
OLDER_ENV_MODE = 0o644
# The entry point's script in an image of the package, at this path in the folder of the image's own code (code_folder)
ENTRY_SCRIPT = "docker/entrypoint.sh"
# The outcome and next lines the stage reads, each a pattern of the whole unwrapped line; the way back's
# pointer, as docker/rkmon gives it (EARLIER_PACKAGE)
WAY_BACK = 'QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor", step 7'
FAILED = (
    r"ERROR: the new monitor failed its first load: .+; its container is stopped, and the older monitor's files are "
    rf"kept, so it can run again fix: .+; or go back to the earlier package: {re.escape(WAY_BACK)}"
)
STATE_10 = r"next: \S+ apply, once what failed is fixed \(a fix in your file: \S+ confirm first\)"
CHANGE_WAITS = r"next: \S+ apply"
NOTHING_KEPT = (
    r"NOTHING TO DO: no earlier version of this package is kept; to go back to the earlier package, see "
    + re.escape(WAY_BACK)
)
APPLIED = (
    r"APPLIED: the deployment runs your confirmed settings( \(\d+ changes?\))?, in standby: it sends nothing until "
    r"you run \S+ standby off"
)
STANDBY_NEXT = r"next: \S+ standby off, once \S+ status and the dashboards read right"
# The next line in standby while the new monitor's first tick has not run since its start (states.next_line: the walk
# of deploy steps 7 and 8), with step 8's words while the records keep no first tick (applied_next)
FIRST_TICK_NEXT = r"next: (step 8, first tick: )?\S+ first-tick"
UPDATED = r"UPDATED: .+"
# apply's lines of its steps over the earlier monitor: that package's doctor's row, its drain, and the load; and
# below that doctor's row, each of its failing rows as it gives them (doctor_shown)
OVER_EARLIER = (
    "doctor before",
    "drain: the older package's own",
    "drain: done",
    "loaded: the monitor runs the settings you confirmed at",
)
# The earlier package's commands that show what its monitor makes of the state the new one wrote, besides its doctor,
# and how long each may take, in seconds; and how long its doctor may take
ON_THE_STATE = ((("status",), 300), (("rk", "tick"), 900))
DOCTOR_S = 900
# Each regular file under the folders named after the code, as root reads it in a container: its owner's and group's
# ids, its mode, the sha256 of its bytes and when it was last written, in nanoseconds since the epoch; as JSON. No
# value is ever printed.
VOLUME_FILES = (
    "import hashlib, json, os, stat, sys\n"
    "out = {}\n"
    "for top in sys.argv[1:]:\n"
    "    for folder, _dirs, files in os.walk(top):\n"
    "        for name in files:\n"
    "            p = os.path.join(folder, name)\n"
    "            s = os.lstat(p)\n"
    "            if not stat.S_ISREG(s.st_mode):\n"
    "                continue\n"
    "            with open(p, 'rb') as f:\n"
    "                sha = hashlib.sha256(f.read()).hexdigest()\n"
    "            out[p] = {'uid': s.st_uid, 'gid': s.st_gid, 'mode': stat.S_IMODE(s.st_mode), 'sha256': sha,\n"
    "                      'mtime_ns': s.st_mtime_ns}\n"
    "print(json.dumps(out))\n"
)


def older_env(scope: str, image: str, port: int, uid: int, gid: int) -> str:
    """docker/.env as the earlier package's settings step writes it for a deployment of its own: the compose project
    SCOPE, without which its compose file names the default deployment's; the image IMAGE, <name>:<tag>; the admin
    dashboard's host port PORT, on this host's loopback; the monitor's user's ids UID and GID.
    """
    name, _, tag = image.rpartition(":")
    values = {
        "COMPOSE_PROJECT_NAME": scope,
        "RKMON_IMAGE": name,
        "RKMON_TAG": tag,
        "RKMON_ADMIN_HOST": "127.0.0.1",
        "RKMON_PORT": port,
        "RKMON_UID": uid,
        "RKMON_GID": gid,
    }
    return "".join(f"{k}={v}\n" for k, v in values.items())


def code_folder(entrypoint: str) -> str:
    """The folder of an image's own code, from ENTRYPOINT, its entry point as `docker image inspect --format
    '{{json .Config.Entrypoint}}'` gives it: a JSON list, one of whose items is the absolute path of ENTRY_SCRIPT in
    that folder. An earlier package's image may keep its code in another folder than this package's
    (container_layout.IMAGE_CODE). ValueError for an entry point that names no such script below a folder.
    """
    try:
        words = json.loads(entrypoint)
    except ValueError:
        words = None
    tail = tuple(ENTRY_SCRIPT.split("/"))
    for word in words if isinstance(words, list) else []:
        path = pathlib.PurePosixPath(word if isinstance(word, str) else "")
        if (
            path.is_absolute()
            and len(path.parts) > len(tail) + 1
            and path.parts[-len(tail) :] == tail
        ):
            return str(path.parents[len(tail) - 1])
    raise ValueError(
        f"the image's entry point names no {ENTRY_SCRIPT} below a folder: {entrypoint.strip()[:200]!r}"
    )


def earlier_settings(rocke_path: str) -> str:
    """A deploy settings file in the earlier package's layout (its schema/deploy_settings.schema.json), as JSON: the
    stand-ins' repository, rocKE's folder in it ROCKE_PATH, the scope's label and the read login's user (the deploy
    path's: REPOSITORY, LABEL, LOGIN), the values its settings section needs; the rest is that package's defaults.
    """
    settings = {
        "kind": "deploy_settings",
        "schema_version": 1,
        "repository": {
            "name": REPOSITORY,
            "path": rocke_path,
            "scope_labels": [LABEL],
        },
        "integrations": {"github_cli": {"account": LOGIN}},
    }
    return json.dumps(settings, indent=1) + "\n"


def writable(member: tarfile.TarInfo, dest: str):
    """tarfile's data filter, then each file and folder writable by you, as a package folder of yours is."""
    member = tarfile.data_filter(member, dest)
    if member.mode is not None:
        member = member.replace(mode=member.mode | stat.S_IWUSR, deep=False)
    return member


def tree(folder) -> dict:
    """What FOLDER holds, by each entry's path in it: a folder's mode, a file's mode and the sha256 of its bytes, or a
    link's target."""
    top, out = pathlib.Path(folder), {}
    for p in sorted(top.rglob("*")):
        s = p.lstat()
        mode = stat.S_IMODE(s.st_mode)
        if stat.S_ISLNK(s.st_mode):
            out[str(p.relative_to(top))] = ["link", os.readlink(p)]
        elif stat.S_ISDIR(s.st_mode):
            out[str(p.relative_to(top))] = ["folder", mode]
        else:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(top))] = ["file", mode, sha]
    return out


def tree_findings(before: dict, after: dict) -> list:
    """How AFTER, what a folder holds (tree), differs from BEFORE: each entry gone, added or changed. [] when none."""
    return (
        [f"{p} is gone" for p in before if p not in after]
        + [f"{p} was added" for p in after if p not in before]
        + [f"{p} changed" for p in before if p in after and before[p] != after[p]]
    )


def layout_mode(pkg):
    """The mode this package's start gives a file of the work volume that an older package made, by the
    rule of docker/rk_layout.py in the package folder PKG: Layout.mode, for the kind of the top folder the file is in
    (Layout.kind_of; a file of the working folder itself is a shared one). A function of the file's path, under WORK,
    and its mode before."""
    ns = runpy.run_path(str(pathlib.Path(pkg) / "docker" / "rk_layout.py"))
    layout = ns["Layout"]

    def mode(path: str, have: int) -> int:
        top, sep, _rest = path[len(WORK) + 1 :].partition("/")
        return layout.mode(have, layout.kind_of(top) if sep else ns["SHARED"], False)

    return mode


def kept_exactly(path: str) -> bool:
    """Whether a failed first apply keeps the file PATH byte for byte (kept_findings' EXACT): config.json, the planted
    file, and each secret, the write login among them."""
    return path in (CONFIG_JSON, PLANTED) or path.startswith(
        (SECRETS + "/", SECRETS_WRITE + "/")
    )


def kept_findings(before: dict, after: dict, made_ns: int, exact, layout) -> tuple:
    """How AFTER differs from BEFORE, each the files of the volumes (VOLUME_FILES) before a first apply over an older
    package and after it failed, against its claim that the older monitor's files are kept: (what breaks the claim,
    the changes it allows), each a list of lines. It breaks: each file gone; each owner or mode changed; each file
    EXACT is true of whose bytes changed; and each other file whose bytes changed after MADE_NS, when the new container
    was made, as only the older monitor's own drain and stop write its files before. It allows: the mode of a file
    under WORK that LAYOUT gives it (layout_mode: a function of its path and its mode before); STOP_WRITES written
    after MADE_NS; and DASHBOARD_PID gone. A file added is the new package's own.
    """
    found, allowed = [], []
    for path, was in sorted(before.items()):
        now = after.get(path)
        if now is None:
            (allowed if path == DASHBOARD_PID else found).append(f"{path} is gone")
            continue
        for what in ("uid", "gid"):
            if now[what] != was[what]:
                found.append(f"{path}: its {what} was {was[what]}, now {now[what]}")
        if now["mode"] != was["mode"]:
            line = f"{path}: its mode was {was['mode']:o}, now {now['mode']:o}"
            if path.startswith(WORK + "/") and now["mode"] == layout(path, was["mode"]):
                allowed.append(f"{line}, as the container's layout gives it")
            else:
                found.append(line)
        if now["sha256"] == was["sha256"]:
            continue
        if exact(path):
            found.append(f"{path}: its bytes changed")
        elif now["mtime_ns"] >= made_ns:
            line = f"{path}: written after the new container was made"
            (allowed if path in STOP_WRITES else found).append(line)
    return found, allowed


def doctor_shown(report: str, out: str) -> list:
    """Each FAIL row of REPORT, the earlier package's doctor's report, that OUT, what apply printed, does not show as
    that doctor gives it: a FAIL row of the same name whose words begin with those of the report's row's first line
    (deployment.doctor_rows; apply shows no fix); and, when REPORT has no FAIL row, that it has none, as the stage's
    earlier deployment has no Claude key. [] when apply shows each."""
    from rkmon_lib import deployment  # noqa: E402

    def words(lines: list) -> str:
        return " ".join(" ".join(lines).split())

    shown = [words(r.lines) for r in deployment.doctor_rows(out) if r.status == "FAIL"]
    failing = [r for r in deployment.doctor_rows(report) if r.status == "FAIL"]
    if not failing:
        return ["the earlier package's doctor fails no check"]
    return [
        f"{words(r.lines[:1])} (not shown as the earlier doctor gives it)"
        for r in failing
        if not any(s.startswith(words(r.lines[:1])) for s in shown)
    ]


def applied_next(line: str, ticked: bool = False) -> bool:
    """Whether LINE is the state's next line in standby after the first apply (states.next_line): standby off's
    once the new monitor's first tick has run since its start (TICKED: first-tick's outcome), and until then
    first-tick's too (FIRST_TICK_NEXT)."""
    lines = (STANDBY_NEXT,) if ticked else (FIRST_TICK_NEXT, STANDBY_NEXT)
    return any(re.fullmatch(p, line) for p in lines)


def stop_left(code: int, said: str) -> str:
    """What the new monitor's `rk stopped` says of a stop mark in its state, from its exit CODE and its output SAID: ""
    when none is left (exit 1, its last line "not stopped"), else what it said, with its exit code.
    """
    lines = said.strip().splitlines()
    if code == 1 and lines[-1:] == ["not stopped"]:
        return ""
    return f"rk stopped: exit {code}: {' / '.join(lines[-3:]) or 'no answer'}"


def written(before: dict, after: dict) -> list:
    """Each file of AFTER that BEFORE does not have, or whose bytes differ, and each file of BEFORE that AFTER does not
    have, each saying which: what was written between the two (VOLUME_FILES)."""
    return [
        f"{p} ({'added' if p not in before else 'changed'})"
        for p in sorted(after)
        if p not in before or before[p]["sha256"] != after[p]["sha256"]
    ] + [f"{p} (gone)" for p in sorted(before) if p not in after]


def written_modes(before: dict, after: dict, first: dict) -> list:
    """Each file of AFTER written since BEFORE (added, or last written later) with three modes: in FIRST, in BEFORE and
    in AFTER, "-" where a reading does not have the file; each reading as VOLUME_FILES makes it.
    """

    def mode(files: dict, path: str) -> str:
        return f"{files[path]['mode']:o}" if path in files else "-"

    return [
        f"{p}: {mode(first, p)}, {mode(before, p)}, {mode(after, p)}"
        for p in sorted(after)
        if p not in before or before[p]["mtime_ns"] != after[p]["mtime_ns"]
    ]


def epoch_ns(stamp: str) -> int:
    """STAMP, a time as docker inspect gives one (RFC 3339 in UTC, to the nanosecond at most), in nanoseconds since the
    epoch. ValueError for any other text."""
    m = re.fullmatch(
        r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z", stamp.strip()
    )
    if not m:
        raise ValueError(f"no time as docker gives one: {stamp!r}")
    at = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    seconds = int(at.replace(tzinfo=datetime.timezone.utc).timestamp())
    return seconds * 10**9 + int((m.group(2) or "").ljust(9, "0"))


class FirstApply(stage_deploy_path.DeployPath):
    """One run of the stage: the deploy path's machine (DeployPath), with the earlier package's deployment on it."""

    def __init__(
        self,
        engine: str,
        image: str,
        pkg: str,
        names: str,
        run_opts: list,
        res,
        earlier: str,
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        self.earlier = earlier
        self.earlier_derived = f"{self.derived}-earlier"
        self.made_earlier = False
        # the package's copy, which the update takes, and the deployment's folder, in which the commands run
        self.new = self.copy
        self.folder = self.top / FOLDER
        self.prev = self.top / PREV
        self.port = free_port()
        # what the earlier package's folder holds once the update has moved it (moved_in)
        self.earlier_tree = {}
        # the volumes' files before the first apply, which fails (failed_first)
        self.original = {}
        # the volumes' files before the first apply over the earlier monitor that ends well, and the checks the
        # earlier package's doctor fails then, and its report (applied_over_earlier)
        self.before = {}
        self.failing = []
        self.report = ""

    # --- the machine, and the earlier package's deployment -----------------------------------------------------------
    def prepare(self) -> None:
        """The deploy path's fresh machine (DeployPath.prepare); the earlier image's build that trusts the stand-ins, as
        that of the image under test does, which the guard lets the packages run too; the earlier package's deployment
        (earlier_deployment), in whose folder the stage's commands run from then on."""
        super().prepare()
        ctx = self.top / "trust"
        lines = (ctx / "Dockerfile").read_text(encoding="utf-8").splitlines(True)
        lines[0] = f"FROM {self.earlier}\n"
        (ctx / "Dockerfile.earlier").write_text("".join(lines), encoding="utf-8")
        self.must(
            "build",
            "--network",
            "none",
            "-f",
            str(ctx / "Dockerfile.earlier"),
            "-t",
            self.earlier_derived,
            str(ctx),
            timeout=900,
        )
        self.made_earlier = True
        guard = self.top / "guard.json"
        config = json.loads(guard.read_text(encoding="utf-8"))
        config["images"].append(self.earlier_derived)
        guard.write_text(json.dumps(config), encoding="utf-8")
        self.earlier_deployment()
        self.copy = self.folder

    def earlier_deployment(self) -> None:
        """The earlier package's deployment, before its setup, its login and its start: its package folder, copied out
        of the earlier image from the folder of its own code, which its entry point names (code_folder), each file and
        folder writable by you; and its docker/.env (older_env), at its writer's mode.
        """
        said = self.must(
            "image",
            "inspect",
            "--format",
            "{{json .Config.Entrypoint}}",
            self.earlier_derived,
        )
        code = code_folder((said.splitlines() or [""])[-1])
        holder = f"{self.scope}-earlier-code"
        self.must("create", "--name", holder, "--network", "none", self.earlier_derived)
        try:
            p = subprocess.run(
                [self.engine, "cp", f"{holder}:{code}", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
        finally:
            self.docker("rm", "-f", holder)
        if p.returncode != 0:
            raise RuntimeError(f"docker cp exited {p.returncode}: {p.stderr[-500:]!r}")
        unpacked = self.top / "unpacked"
        with tarfile.open(fileobj=io.BytesIO(p.stdout)) as archive:
            archive.extractall(unpacked, filter=writable)
        (unpacked / pathlib.PurePosixPath(code).name).rename(self.folder)
        unpacked.rmdir()
        env = self.folder / ENV
        env.write_text(
            older_env(
                self.scope, self.earlier_derived, self.port, os.getuid(), os.getgid()
            ),
            encoding="utf-8",
        )
        env.chmod(OLDER_ENV_MODE)

    # --- what the stage reads ----------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def at(self, folder):
        """The person's commands (DeployPath.rkmon) run in FOLDER, a package folder, while in the block."""
        was, self.copy = self.copy, folder
        try:
            yield
        finally:
            self.copy = was

    def update(self, source, by=None) -> tuple:
        """`rkmon package update --from SOURCE` for the deployment's folder: by its own rkmon, or by BY's, a package
        folder, with --folder. Its exit code and its output."""
        if by is None:
            return self.rkmon("package", "update", "--from", str(source))
        with self.at(by):
            return self.rkmon(
                "package", "update", "--from", str(source), "--folder", str(self.folder)
            )

    def compose_filter(self) -> tuple:
        """docker's filters on the deployment's own container: its compose project's, not a one-off."""
        from rkmon_lib import identity  # noqa: E402

        return (
            "--filter",
            f"label={identity.PROJECT}={self.scope}",
            "--filter",
            "label=com.docker.compose.oneoff=False",
        )

    def container(self) -> str:
        """The deployment's container, running or not; "" when there is none."""
        rc, out = self.docker(
            "ps", "-a", *self.compose_filter(), "--format", "{{.Names}}"
        )
        names = out.split() if rc == 0 else []
        return names[0] if len(names) == 1 else ""

    def running_image(self) -> str:
        """The image of the deployment's running container; "" when none runs."""
        rc, out = self.docker("ps", *self.compose_filter(), "--format", "{{.Image}}")
        return out.strip() if rc == 0 else ""

    def volume_files(self) -> dict:
        """Each file of the folders READ, the monitor's home and the two secrets volumes, as VOLUME_FILES reads them,
        as root, in a one-off container of the image under test with no network and the volumes read-only.
        """
        mounts = [
            w
            for v, where in VOLUMES.items()
            for w in ("-v", f"{self.scope}_{v}:{where}:ro")
        ]
        out = self.must(
            "run",
            "--rm",
            "--name",
            f"{self.scope}-files",
            "--network",
            "none",
            "--user",
            "0:0",
            *mounts,
            "--entrypoint",
            "python3",
            self.derived,
            "-I",
            "-c",
            VOLUME_FILES,
            *READ,
        )
        return json.loads(out.splitlines()[-1])

    # --- the checks --------------------------------------------------------------------------------------------------
    def earlier_runs(self) -> bool:
        """The earlier package's own `./rkmon start` in the folder, then its `./rkmon status`: each exits 0, and the
        deployment's running container is the earlier image's."""
        rc, _out = self.rkmon("start")
        image = self.running_image()
        code = self.rkmon("status")[0] if rc == 0 else None
        ok = rc == 0 and image == self.earlier_derived and code == 0
        if not ok:
            hostrun.indented(
                f"the earlier package's start: exit {rc}; running: {image or 'nothing'}; its status: exit {code}"
            )
        return ok

    def earlier_started(self) -> bool:
        """The earlier package's read login, by its own `rkmon login gh-read --with-token`; its own setup of the
        sections EARLIER_SECTIONS from a deploy settings file (earlier_settings), which writes config.json and clones
        the repository with that login; then its start: its monitor runs (earlier_runs). Each exits 0.
        """
        file = self.top / EARLIER_SETTINGS
        file.write_text(earlier_settings(self.rocke_path()), encoding="utf-8")
        file.chmod(0o600)
        rc, _out = self.rkmon(
            "login", "gh-read", "--with-token", stdin=(self.token + "\n").encode()
        )
        if rc == 0:
            sections = [w for s in EARLIER_SECTIONS for w in ("--section", s)]
            rc, _out = self.rkmon("setup", "--settings", str(file), *sections)
        return self.result(
            rc == 0 and self.earlier_runs(),
            "the earlier package's own login, setup and start run its monitor on the deployment's volumes",
        )

    def fill_in(self) -> None:
        """The deploy path's values (DeployPath.fill_in), with the earlier deployment's admin port, which a move in
        place keeps."""
        from rkmon_lib import file_types, settings_file  # noqa: E402

        super().fill_in()
        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        doc.values["features"]["dashboards"]["admin"]["port"] = self.port
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    def moved_in(self) -> bool:
        """The move in place, to the login: package update by this package's rkmon, which keeps the earlier folder
        as FOLDER.prev; then, in the folder, template (and fill_in), the check and confirm its next lines give, and the
        login walk (DeployPath.logins), each exiting 0 with its outcome line, login's next line apply's (CHANGE_WAITS).
        Confirm's next line is shown: the earlier monitor runs meanwhile."""
        from rkmon_lib import steps  # noqa: E402

        earlier = tree(self.folder)
        # the update sets the earlier package's docker/.env, which its writer left looser, to 600, as its row says
        earlier[ENV][1] = 0o600
        rc, out = self.update(self.new, by=self.new)
        self.earlier_tree = tree(self.prev) if self.prev.is_dir() else {}
        moved = [f"{PREV}: {x}" for x in tree_findings(earlier, self.earlier_tree)]
        new = (self.new / "docker" / "rkmon").read_bytes()
        here = self.folder / "docker" / "rkmon"
        ok = (
            rc == 0
            and bool(outcome(out, UPDATED))
            and not moved
            and here.is_file()
            and here.read_bytes() == new
        )
        said = [f"package update: exit {rc}, outcome {outcome(out, UPDATED)!r}", *moved]
        if ok:
            rc, out = self.rkmon(
                "template", "deploy", "--profile", PROFILE, str(self.file)
            )
            ok = rc == 0 and bool(outcome(out, OUTCOMES[1]))
            said.append(f"template: exit {rc}")
        if ok:
            self.fill_in()
            for n, more in ((2, ()), (3, ("--yes",))):
                rc, out = self.follow(out, *more)
                ok = rc == 0 and bool(outcome(out, OUTCOMES[n]))
                said.append(f"step {n}: exit {rc}, next line {next_line(out)!r}")
                if not ok:
                    break
        if ok:
            login = steps.command_name(steps.DEPLOY_STEPS[4].command)
            rc, out = self.logins(f"next: ./rkmon {login}")
            ok = (
                rc == 0
                and bool(outcome(out, OUTCOMES[5]))
                and bool(re.fullmatch(CHANGE_WAITS, next_line(out)))
            )
            said.append(f"login: exit {rc}, next line {next_line(out)!r}")
        hostrun.indented("\n".join(said))
        return self.result(
            ok,
            f"package update puts this package in the earlier package's folder, keeping that one as {PREV}, and "
            "template, check, confirm and login pass there, the earlier monitor running, login's next line apply",
        )

    def failed_first(self) -> bool:
        """The first apply over the earlier monitor, its first load failing on the planted file (PLANTED), with a
        stand-in write login planted too (WRITE_LOGIN): ERROR, exit 1, with both ways forward and state 10's next
        line; then the new container stopped, `rkmon next` gives that line, and rollback NOTHING TO DO with the way
        back; and the earlier package's files kept (kept_findings, tree_findings), each change it allows shown. Whether
        the apply ended in ERROR, exit 1, which the way back needs.
        """
        folder = os.path.dirname(WRITE_LOGIN)
        self.must(
            "exec",
            "-u",
            "0",
            self.container(),
            "sh",
            "-c",
            f"printf '{{' > {PLANTED} && chown rkmon:rkmon {PLANTED} && chmod 600 {PLANTED} && mkdir -p {folder} "
            f"&& printf %s {shlex.quote(WRITE_LOGIN_TEXT)} > {WRITE_LOGIN} && chown -R rkwrite:rkmon {folder} "
            f"&& chmod 700 {folder} && chmod 600 {WRITE_LOGIN}",
        )
        before = self.original = self.volume_files()
        login = before.get(WRITE_LOGIN)
        if login:
            hostrun.indented(
                f"planted: {WRITE_LOGIN}, uid {login['uid']}, gid {login['gid']}, mode {login['mode']:o}"
            )
        rc, out = self.rkmon("apply")
        ended = rc == 1 and bool(outcome(out, FAILED))
        ok = ended and bool(re.fullmatch(STATE_10, next_line(out)))
        if not ok:
            hostrun.indented(f"apply: exit {rc}, next line {next_line(out)!r}")
        self.result(
            ok,
            "the first apply over the earlier monitor, whose first load fails, ends in ERROR, exit 1, with both ways "
            "forward and state 10's next line",
        )
        name = self.container()
        made = self.must("inspect", "--format", "{{.Created}} {{.Config.Image}}", name)
        stamp, _, image = made.partition(" ")
        rc, said = self.rkmon("next")
        code, back = self.rkmon("rollback")
        stopped = (
            image == self.derived
            and not self.running_image()
            and rc == 0
            and bool(re.fullmatch(STATE_10, next_line(said)))
            and code == 0
            and bool(outcome(back, NOTHING_KEPT))
        )
        if not stopped:
            hostrun.indented(
                f"the container's image {image!r}, running {self.running_image()!r}; next: exit {rc}, "
                f"{next_line(said)!r}; rollback: exit {code}, outcome {outcome(back, NOTHING_KEPT)!r}"
            )
        self.result(
            stopped,
            "after it the new container is stopped, rkmon next gives state 10's line, and rollback says NOTHING TO DO "
            "with the way back to the earlier package",
        )
        found, allowed = kept_findings(
            before,
            self.volume_files(),
            epoch_ns(stamp),
            kept_exactly,
            layout_mode(self.new),
        )
        if WRITE_LOGIN not in before:
            found.append(f"{WRITE_LOGIN}: not planted")
        found += [
            f"{PREV}: {x}" for x in tree_findings(self.earlier_tree, tree(self.prev))
        ]
        if found or allowed:
            hostrun.indented("\n".join([*(f"allowed: {x}" for x in allowed), *found]))
        self.result(
            not found,
            "the failed first apply keeps the earlier package's files: its folder, and each file of the monitor's home "
            "and the two secrets volumes with its owner and its mode (or the container layout's, in the work volume), "
            "config.json, the secrets, the write login and the planted file byte for byte, the others last written "
            "before the new container was made, but for the stop mark and journal its stop writes",
        )
        return ended

    def way_back(self, when: str) -> bool:
        """The way back, WHEN: `rkmon stop`, `rkmon package update --from FOLDER.prev`, then the earlier package's
        start in the folder, which holds it as it was: its monitor runs (earlier_runs).
        """
        rc, _out = self.rkmon("stop")
        code, out = self.update(self.prev)
        found = tree_findings(self.earlier_tree, tree(self.folder))
        if found:
            hostrun.indented("\n".join(found))
        ok = (
            rc == 0
            and code == 0
            and bool(outcome(out, UPDATED))
            and not found
            and self.earlier_runs()
        )
        return self.result(
            ok,
            f"the way back {when}: rkmon stop, package update --from {PREV} and the earlier package's start run the "
            "earlier monitor again, from its folder as it was",
        )

    def applied_over_earlier(self) -> bool:
        """The way forward again: the planted file moved aside, as its fix says; the checks the earlier package's
        doctor fails (older_doctor); package update --from FOLDER.prev by the package's copy the move began from, which
        puts this package back with its records and keeps the earlier package's folder as FOLDER.prev again, as it was;
        `rkmon next`, state 10's line; then `rkmon apply`, the first apply over the running earlier monitor
        (OVER_EARLIER), the earlier doctor's failing rows as it gives them (doctor_shown): APPLIED, in standby, its next
        line the state's (applied_next), and the new image runs.
        """
        self.must("exec", "-u", "0", self.container(), "mv", PLANTED, ASIDE)
        _rc, self.failing, self.report = self.older_doctor()
        earlier = tree(self.folder)
        rc, out = self.update(self.prev, by=self.new)
        self.earlier_tree = tree(self.prev) if self.prev.is_dir() else {}
        moved = [f"{PREV}: {x}" for x in tree_findings(earlier, self.earlier_tree)]
        code, said = self.rkmon("next")
        ok = (
            rc == 0
            and bool(outcome(out, UPDATED))
            and not moved
            and code == 0
            and bool(re.fullmatch(STATE_10, next_line(said)))
        )
        if not ok:
            hostrun.indented(
                "\n".join(
                    [
                        f"package update: exit {rc}; next: exit {code}, {next_line(said)!r}",
                        *moved,
                    ]
                )
            )
        if ok:
            self.before = self.volume_files()
            rc, out = self.rkmon("apply")
            missing = [x for x in OVER_EARLIER if x not in out]
            missing += doctor_shown(self.report, out)
            image = self.running_image()
            ok = (
                rc == 0
                and bool(outcome(out, APPLIED))
                and applied_next(next_line(out))
                and not missing
                and image == self.derived
            )
            if not ok:
                hostrun.indented(
                    f"apply: exit {rc}, next line {next_line(out)!r}, lines missing {missing}, running {image!r}"
                )
        return self.result(
            ok,
            "with the fault moved aside, package update --from the kept folder and rkmon apply make the first apply "
            "over the running earlier monitor, its doctor, with its failing rows as it gives them, its drain and the "
            "load, APPLIED in standby, its next line first-tick's or, once that tick has run, standby off's",
        )

    def no_stop_left(self) -> bool:
        """After APPLIED, the new monitor's own `rk stopped`, in its container as its user: no stop mark is left in its
        state (stop_left), though the earlier package's drain wrote one, which the new container's start clears
        (docker/entrypoint.sh runs rk start), or else apply after the load."""
        from rkmon_lib import container_layout  # noqa: E402

        user = container_layout.MONITOR.name
        rc, out = self.docker("exec", "-u", user, self.container(), "rk", "stopped")
        found = stop_left(rc, out)
        if found:
            hostrun.indented(found)
        return self.result(
            not found,
            "after APPLIED, no stop mark is left in the new monitor's state: the earlier package's drain's was cleared",
        )

    def older_doctor(self) -> tuple:
        """The earlier package's `./rkmon doctor` in the folder: its exit code, the names of the checks it fails, as
        deployment.doctor_rows reads its report, and its report."""
        from rkmon_lib import deployment  # noqa: E402

        rc, out = self.rkmon("doctor", timeout=DOCTOR_S)
        names = [r.name for r in deployment.doctor_rows(out) if r.status == "FAIL"]
        return rc, names, out

    def first_tick(self) -> None:
        """`rkmon first-tick`: the new monitor's first tick, in standby, as deploy step 8 waits for it; its next line
        standby off's."""
        rc, out = self.rkmon("first-tick", timeout=step_timeout(8))
        self.result(
            rc == 0
            and bool(outcome(out, OUTCOMES[8]))
            and applied_next(next_line(out), ticked=True),
            "the new monitor's first tick runs in standby, and its next line is standby off's",
        )

    def on_the_state(self) -> None:
        """What the earlier monitor makes of the state the new monitor wrote in standby,
        shown, not judged: each file the two monitors wrote since the first apply that ended well began (written); the
        earlier package's commands of ON_THE_STATE, each with its exit code, and the modes of what the earlier monitor
        wrote meanwhile, beside each file's before the first apply and at the way back (written_modes); its doctor's
        exit code and the checks it fails, beside those it failed before that apply and those it fails only now
        (deployment.new_failures); and its container's log's lines since its start that name an error.
        """
        from rkmon_lib import deployment  # noqa: E402

        start = self.volume_files()
        lines = [
            "what the monitors wrote since the earlier one last ran alone:",
            *(f"  {x}" for x in written(self.before, start)),
        ]
        for words, timeout in ON_THE_STATE:
            rc, _out = self.rkmon(*words, timeout=timeout)
            lines.append(f"./rkmon {' '.join(words)}: exit {rc}")
        meanwhile = written_modes(start, self.volume_files(), self.original)
        lines += [
            "the modes of what it wrote meanwhile: before the first apply, at the way back, now:",
            *(f"  {x}" for x in meanwhile),
        ]
        rc, failing, _report = self.older_doctor()
        new = deployment.new_failures(self.failing, failing)
        lines.append(
            f"./rkmon doctor: exit {rc}, failing: {', '.join(failing) or 'none'}; before the first apply: "
            f"{', '.join(self.failing) or 'none'}; failing only now: {', '.join(new) or 'none'}"
        )
        _rc, logs = self.docker("logs", self.container())
        errors = [
            x for x in logs.splitlines() if re.search(r"error|traceback", x, re.I)
        ]
        lines += ["the earlier monitor's log, lines that name an error:", *errors[-20:]]
        hostrun.indented(
            "the earlier monitor on the state the new monitor wrote in standby:\n"
            + "\n".join(lines)
        )

    # --- what the packages did, and the end --------------------------------------------------------------------------
    def after(self) -> None:
        """What both packages did: the stand-ins' log as the deploy path reads it (requests_findings), and each docker
        command the deployment's own (guard_findings). A move in place begins with the earlier package's commands, so
        the deploy path's check of the docker commands before confirm does not apply."""
        bad, good = stage_deploy_path.requests_findings(self.requests(), self.marks)
        hostrun.indented("\n".join(bad + good))
        self.result(
            not bad,
            "the packages read GitHub with the read login, cloned the repository, reached the Claude gateway with its "
            "key, and wrote nothing",
        )
        log = self.top / "guard.log"
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        refused, counts = stage_deploy_path.guard_findings(
            [json.loads(x) for x in lines]
        )
        hostrun.indented(
            "\n".join(
                refused + [f"docker commands: {json.dumps(counts, sort_keys=True)}"]
            )
        )
        self.result(
            not refused,
            "every docker command of both packages was the deployment's own, and every listing showed only its objects",
        )

    def clean(self) -> None:
        """DeployPath.clean from the package's copy, where no settings were confirmed: every object of the scope removed
        by name, as the folder's package changes during the stage; then the earlier image's build.
        """
        self.copy = self.new
        super().clean()
        if self.made_earlier:
            self.docker("rmi", self.earlier_derived)


def first_apply_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res, earlier: str
) -> None:
    """The stage (the module's docstring), on IMAGE and EARLIER, an image of the package before this one: its results
    through RES. Skipped without EARLIER."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    if not earlier:
        print(
            f"SKIP  {STAGE}: it needs an image of the package before this one: tests/container/image.py --no-build "
            f"--image IMAGE --earlier-image EARLIER_IMAGE --stages {STAGE}"
        )
        return
    t = FirstApply(engine, image, pkg, names, run_opts, res, earlier)
    try:
        t.prepare()
        if (
            t.earlier_started()
            and t.moved_in()
            and t.failed_first()
            and t.way_back("after the failed apply")
            and t.applied_over_earlier()
        ):
            if t.no_stop_left():
                t.first_tick()
            if t.way_back("from standby"):
                t.on_the_state()
        t.after()
    except (RuntimeError, OSError, ValueError, tarfile.TarError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
