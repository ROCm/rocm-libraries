# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's deploy-path stage (tests/container/image.py): the deploy path, its eight steps, as
QUICK_DEPLOY_GUIDE.md takes a person through them on a host with another deployment ("Several deployments": a package
folder, a container name and a port of its own), from a copy of the package folder, on the image under test, every
feature in prototype. Step 1 is the guide's command; each step after it runs the command the next line before it
gives. The stage checks each step's outcome and next line, and after confirm that `rkmon next` gives the same line;
once the monitor has started, that `rk config show --source` shows deploy for exactly the keys the settings file sets,
at their values, and that nothing in the container writes the settings volume, as the monitor's user or as root. Once
the path is done, it changes a setting as a person changes one: an edit of the settings file's line,
`rkmon confirm`, then the command its next line gives, `rkmon apply`, and each next line after it to the done line, as
after a start; after which config show shows the file's keys again, the changed one at its new value. Once the monitor
has started, the stage stops it as a person does: `rkmon stop --drain` lets the work in flight
finish, then stops the monitor and its container, its next line gives `rkmon start`, and docker lists no running
container of the deployment. Then it checks what the deployment did: it read GitHub with the read
login's token, cloned the repository, reached the Claude gateway with the key sent as the settings say, and wrote
nothing; before confirm, docker was asked about no object but in listings filtered exactly on the deployment's own
name; and at the end, that the deployment's own `rkmon remove --delete-data --yes` leaves nothing of it on this host.

Where the guide's person uses a terminal, the stage does without one:
- step 1's four `<...>` values are filled in by code, with the other values the guide has the person change: the
  container's name and the admin dashboard's port of their own, the image on this host, and the Claude gateway;
- confirm runs with `--yes`, which answers its question;
- step 5 sets each login and secret the walk lists as missing: a GitHub login with `rkmon login NAME --with-token` and
  the claude login with `rkmon login claude`, each with its value on stdin, and a secret with `rkmon secret NAME
  FILE`.

Each next line is the state's (rkmon_lib/states.py): it names the step whose command it gives, with the step's number
and name until a tick has run since the start. The image is on this host already, so confirm's next line skips step 4
and names step 5, and `rkmon image check`, step 4's check of an image here, runs as a check of the stage's own,
outside the guide's steps. From the start on, the container's own tick may run at any time; after it, the next line
names its command alone, and doctor's is the done line, a tick being step 8's work: `rkmon first-tick` then runs as a
check of the stage's own too, whose rk status of the first tick has no alarm line.

The fresh machine:
- docker as a machine with no other deployment shows it (tests/fakes/docker_guard, first on each command's PATH):
  every listing narrowed to the stage's objects, a one-off container that names no network placed on the stage's
  network, and any command on another object refused, which fails the stage.
- GitHub and the Claude gateway are stand-ins (tests/fakes/fake_net.py) in a container on the deployment's network,
  made before the first start, with no way out of this host (an internal network). Docker's DNS gives that container
  the names github.com, api.github.com and the gateway's host (network aliases), and the image the deployment runs
  trusts the stand-ins' root certificate, added by a build on top of the image under test. Neither is a setting, and
  no deploy file can set either: the deployment runs the settings a person writes, and calls GitHub and the gateway at
  the addresses they name.

Its objects are named after its scope, <names>-deploy-path: the compose project (container.name), its container,
volumes and network, the stand-ins' container, and the image <name>:<scope>. Each is removed at the end: the
stand-ins' container, then the deployment by its own remove, then whatever is left of the scope, the image, and the
temporary folder that holds the package's copy, the settings file and the secrets' files.
"""

import json
import os
import pathlib
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import hostrun

STAGE = "deploy-path"
PROFILE = "full"  # the guide's step 1
LOGIN = "op-login"  # the GitHub user the stand-in's tokens belong to
REPOSITORY = "acme/widgets"
LABEL = "rocke"
GATEWAY = "llm-gateway.example.com"
KEY_HEADER = "X-Gateway-Key"  # the gateway's own header, which carries the key beside the bearer token
FAKES_USER = "65534:65534"  # the stand-ins' user, who owns only their state
FAKES_DIR = "/tmp/fakes"  # in the stand-ins' container: their code and their config
FAKES_STATE = "/fakes-state"  # and what they make and log, on a tmpfs of their own: the image's /tmp is root's
STEP_TIMEOUT_S = 900
# The steps a step's next line may name, when not only the one after it: confirm's, the state's, as `rkmon next` gives
# it, skips step 4 when the image is on this host, as the stage's is, and names step 5.
SKIPS = {3: (5,)}
# What each step the stage runs does, as its result says: confirm's next line skips step 4 (SKIPS).
WHAT = {
    1: f"template deploy --profile {PROFILE} writes the settings file, yours only",
    2: "check passes the filled-in settings on this machine",
    3: "confirm records the settings file",
    5: "login passes once each login and secret the settings need is set",
    6: "start starts the monitor on the confirmed settings",
    7: "doctor passes the running monitor and what it reaches",
    8: "first-tick sees the monitor's first tick",
}
# The checks of the stage's own, outside the guide's steps (the module's docstring), by the step whose work a next line
# skips as done: what each sees, beside its exit code and the step's outcome line.
OWN = {
    4: "the image here suits this package and trusts the network's TLS",
    8: "the monitor's first tick has run since the start with no alarm, and the path is done",
}
# How an alarm line of a tick or of rk status starts: in prototype nothing is wrong, so the first tick raises none.
ALARM = "!!"
# The two ways a step's last line begins: the next step's command, or that the path is done.
NEXT = ("next: ", "done: ")
# The done line's beginning: after step 8, and after doctor once a tick has run since the start.
DONE = "done: the monitor runs"
# Each step's outcome line.
OUTCOMES = {
    1: r"WROTE \S+ \(profile \S+; yours only, permissions 600\): fill in the \d+ values? marked <\.\.\.>",
    2: r"PASSED: \d+ checks?( \(\d+ warnings?\))?",
    3: r"CONFIRMED: \S+ is this deployment's settings file",
    4: r"PASSED: \S+ suits this package",
    5: r"PASSED: every login and secret your settings need is set( \(\d+ warnings?\))?",
    6: r"STARTED: the monitor runs your confirmed settings",
    7: r"PASSED: \d+ checks?( \(\d+ warnings?\))?",
    8: r"PASSED: the first tick ran at \S+",
}
# The setting the stage changes once the path is done, and its new value: the template sets it to another
CHANGE = ("monitor.tick_interval_minutes", 15)
# The outcome lines of that change's confirm and apply
CHANGE_OUTCOMES = {
    "confirm": r"CONFIRMED: 1 change waits for apply; nothing runs differently yet",
    "apply": r"APPLIED: the deployment runs your confirmed settings \(1 change\)",
}
# The most the stage's `rkmon stop --drain` waits for the work in flight, in minutes: in prototype, against the
# stand-ins, no review is in flight
STOP_DRAIN_MINUTES = 5
# `rkmon stop --drain`'s lines once its drain has ended well: rk drain's own first line, passed on
# indented (rkmon_lib/drain.py wait), so that its drain ran; the drain's last line, said indented, with how long it
# took once that is a minute or more; then the outcome line
DRAIN_STARTED = "drain: nothing new starts"
DRAIN_DONE = "drain: done"
STOP_OUTCOME = r"STOPPED: the monitor's container"
# The wrapper as the guide's person types it, which the lines name; the outcome lines of a start in standby, as the
# start after a one-time move is, and of `rkmon standby off`, which ends it
PROGRAM = "./rkmon"
STANDBY_WORDS = (
    rf"in standby: it sends nothing until you run {re.escape(PROGRAM)} standby off"
)
STARTED_IN_STANDBY = (
    r"STARTED: the monitor runs your confirmed settings, " + STANDBY_WORDS
)
RELEASED = r"RELEASED: the monitor sends again, as your settings allow"
# How long a stage waits for the one-time move once the start has said STARTED: the container's own start makes it
MOVE_WAIT_S = 300
# The deployment's work volume, by its name in docker/compose.yaml: docker compose puts the compose project's name and
# "_" before it
WORK_VOLUME = "rkmon-work"
# A write to the settings volume, tried in the running container as one of its users (DeployPath.settings_read_only):
# python3 -c, given the settings folder and its layer file. A new file made in the folder, removed again when it is
# made, and the layer file opened to add to it, nothing written: each try's path and what it found, "written" or the
# error's words, as JSON.
WRITE_TRIES = (
    "import json, os, sys\n"
    "folder, layer = sys.argv[1:3]\n"
    "out = {}\n"
    "for path, flags in ((os.path.join(folder, 'write-try'), os.O_CREAT | os.O_EXCL), (layer, os.O_APPEND)):\n"
    "    try:\n"
    "        os.close(os.open(path, os.O_WRONLY | flags, 0o600))\n"
    "    except OSError as e:\n"
    "        out[path] = e.strerror\n"
    "        continue\n"
    "    out[path] = 'written'\n"
    "    if flags & os.O_CREAT:\n"
    "        os.unlink(path)\n"
    "print(json.dumps(out))\n"
)
# The error's words for a write on a read-only mount (EROFS)
READ_ONLY = "Read-only file system"


def free_port() -> int:
    """A port nothing on this host listens on now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def step_timeout(n: int) -> float:
    """The most the stage waits for deploy step N's command, in seconds: first-tick (step 8) waits its default time at
    most, and the stage waits for it a little longer; any other step STEP_TIMEOUT_S."""
    from rkmon_lib import commands  # noqa: E402

    return (commands.FIRST_TICK_TIMEOUT_MINUTES + 5) * 60 if n == 8 else STEP_TIMEOUT_S


def github_token() -> str:
    """An invented fine-grained GitHub token, in the form GitHub gives one (its prefix in two parts, so that no scanner
    reads this file as holding a token)."""
    return "github" + "_pat_" + secrets.token_hex(11) + "_" + secrets.token_hex(29)


def next_command(line: str) -> tuple:
    """(the deploy step a next line names, the words of its command after the program's name), from
    `next: step N, NAME: PROGRAM WORDS`, or from `next: PROGRAM WORDS`, whose step is the one with that command
    (rkmon_lib/steps.py); (0, []) for any other line."""
    from rkmon_lib import steps  # noqa: E402

    m = re.fullmatch(r"next: (?:step (\d+), [^:]+: )?(.+)", line)
    try:
        words = shlex.split(m.group(2))[1:] if m else []
    except ValueError:
        words = []
    if not words:
        return 0, []
    if m.group(1):
        return int(m.group(1)), words
    for step in steps.DEPLOY_STEPS:
        for command in (step.command, *step.others):
            name = steps.command_name(command).split()
            if words[: len(name)] == name:
                return step.n, words
    return 0, words


def lines_from(lines: list, i: int, stop) -> str:
    """LINES[I] and the lines after it up to the first one STOP is true of, or the end, as one line: rkmon wraps a long
    line at a space, and its next lines begin where it began."""
    end = next((j for j in range(i + 1, len(lines)) if stop(lines[j])), len(lines))
    return " ".join(x.strip() for x in lines[i:end])


def outcome(out: str, pattern: str) -> str:
    """The outcome line in OUT that PATTERN matches whole, unwrapped, or "": it comes right before the next line."""
    want, lines = re.compile(pattern), out.splitlines()
    for i in range(len(lines)):
        text = lines_from(lines, i, lambda x: not x.strip() or x.startswith(NEXT))
        if want.fullmatch(text):
            return text
    return ""


def next_line(out: str) -> str:
    """The last next or done line of OUT, unwrapped, or "": the last thing a step shows."""
    lines = out.splitlines()
    at = [i for i, x in enumerate(lines) if x.startswith(NEXT)]
    return lines_from(lines, at[-1], lambda x: not x.strip()) if at else ""


def says(out: str, text: str) -> bool:
    """Whether OUT holds TEXT, however either is wrapped: each run of white space as one space."""
    return " ".join(text.split()) in " ".join(out.split())


def polled(read, ready, wait: float) -> dict:
    """What READ returns once READY is true of it, or once WAIT seconds have passed, read every 3 seconds: the last
    answer. An answer READ cannot give (RuntimeError, ValueError, IndexError) is {"unread": why}, never ready.
    """
    end = time.time() + wait
    while True:
        try:
            answer = read()
        except (RuntimeError, ValueError, IndexError) as exc:
            answer = {"unread": str(exc)[-300:]}
        if ("unread" not in answer and ready(answer)) or time.time() > end:
            return answer
        time.sleep(3)


def stopped_drained(out: str) -> bool:
    """Whether OUT, `rkmon stop --drain`'s, has rk drain's own first line (DRAIN_STARTED), the drain's done line
    (DRAIN_DONE, alone or with how long it took), the outcome line (STOP_OUTCOME), and a next line that gives `rkmon
    start`."""
    lines = [x.strip() for x in out.splitlines()]
    started = any(x.startswith(DRAIN_STARTED) for x in lines)
    drained = any(
        x == DRAIN_DONE or x.startswith(f"{DRAIN_DONE} after ") for x in lines
    )
    return (
        started
        and drained
        and bool(outcome(out, STOP_OUTCOME))
        and next_command(next_line(out))[1] == ["start"]
    )


def command_label(args) -> str:
    """An rkmon command as the stand-ins' findings name it: its words up to the first option or path."""
    words = []
    for a in args:
        if a.startswith(("-", "/", ".")):
            break
        words.append(a)
    return shlex.join(["./rkmon", *words])


def during(t: float, marks: list) -> str:
    """The rkmon command that ran at time T: the last of MARKS, (start, name) pairs, to start by then."""
    names = [name for start, name in marks if start <= t]
    return names[-1] if names else "the stage's own setup"


def walk_names(out: str) -> list:
    """The logins and secrets the login walk's OUT says are missing or wrong: the name of each of its FAIL rows, which
    come before the doctor's part."""
    names = []
    for line in out.splitlines():
        if line.strip().startswith("The doctor"):
            break
        m = re.match(r"\s*FAIL\s+([\w.-]+):", line)
        if m:
            names.append(m.group(1))
    return names


def edited(text: str, dotted: str, value) -> str:
    """TEXT, a settings file, with the key DOTTED set to VALUE as a person edits it: on the key's own line, found under
    its sections by their indent (two spaces a level), what follows the colon replaced. ValueError unless exactly one
    line sets the key."""
    want, path, found = dotted.split("."), [], []
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = re.match(r"( *)([\w.-]+):", line)
        if m:
            path = path[: len(m.group(1)) // 2] + [m.group(2)]
            if path == want:
                found.append(i)
    if len(found) != 1:
        raise ValueError(f"{dotted}: {len(found)} lines of the settings file set it")
    key, _, rest = lines[found[0]].partition(":")
    lines[found[0]] = f"{key}: {json.dumps(value)}" + (
        "\n" if rest.endswith("\n") else ""
    )
    return "".join(lines)


def file_keys(values: dict, maps, path: str = "") -> dict:
    """The keys a settings file's VALUES set, dotted, each at its value: each key with a value that is not blank (null
    or ""), a map (one of MAPS, dotted names) one key. A section is no key, so an empty one sets none.
    """
    out = {}
    for k, v in values.items():
        name = f"{path}.{k}" if path else k
        if isinstance(v, dict) and name not in maps:
            out.update(file_keys(v, maps, name))
        elif v is not None and v != "":
            out[name] = v
    return out


def deploy_differences(show: str, values: dict) -> list:
    """How SHOW, the output of `rk config show --source`, differs from VALUES, the keys the settings file sets at their
    values: each key of VALUES it shows at another value, from another layer or not at all, and each
    deploy row of a key VALUES does not hold."""
    from rkmon_lib import runtime_config  # noqa: E402

    rows = {}
    for line in show.splitlines():
        m = re.fullmatch(r"(\S+) +(\S+) = (.+)", line)
        if m:
            try:
                rows[m.group(2)] = (m.group(1), json.loads(m.group(3)))
            except ValueError:
                # no value as config show writes one: kept as its text, which no value of the file equals
                rows[m.group(2)] = (m.group(1), m.group(3))
    deploy = runtime_config.DEPLOY
    bad = [
        f"{name}: the file sets {json.dumps(value)}, config show "
        + (
            f"has {rows[name][0]} {json.dumps(rows[name][1])}"
            if name in rows
            else "has no row for it"
        )
        for name, value in values.items()
        if rows.get(name) != (deploy, value)
    ]
    return bad + [
        f"{name}: a {deploy} row, {json.dumps(value)}, of a key the file does not set"
        for name, (source, value) in rows.items()
        if source == deploy and name not in values
    ]


def read_only_findings(tries: dict) -> list:
    """How TRIES, {a user: what WRITE_TRIES found as that user in the running container, or the command's output when
    it gave nothing to read}, differ from a settings volume nothing in the container writes: each try refused, and
    root's as a read-only file system (READ_ONLY), since the container's CAP_DAC_OVERRIDE lets root past every file's
    permissions and only the read-only mount stops it. [] when they do not."""
    found = []
    for user, tried in tries.items():
        if not isinstance(tried, dict) or len(tried) != 2:
            found.append(f"as {user}, the two tries did not run: {str(tried)[-500:]}")
            continue
        for path, said in tried.items():
            if said == "written":
                found.append(f"as {user}, {path} was opened to write")
            elif user == "root" and said != READ_ONLY:
                found.append(f"as root, {path}: {said}, not {READ_ONLY.lower()}")
    return found


def requests_findings(
    log: list, marks: list, sends: str = "", clone: bool = True, gateway: bool = True
) -> tuple:
    """(what the stand-ins' log shows went wrong, what it shows went right) of LOG, its lines, MARKS the rkmon commands'
    starts: each GitHub API call with a login's token, each read with the read login's; no write but the login checks'
    probes, which make nothing, before SENDS, the command from whose first start on the deployment sends (MARKS' name
    of it: command_label; "" for none), and from then on at least one, each the write login's, answered; a GitHub read;
    the clone, unless CLONE is false; the gateway reached with its key, unless GATEWAY is false; each call that sent the
    key sending it as the settings say. Each call answered otherwise than 200 is named with the command it came during.
    """
    bad, good = [], []
    unsigned = [
        r
        for r in log
        if r["host"] == "api.github.com" and r.get("token") not in ("login", "write")
    ]
    bad += [
        f"a GitHub call without a login's token ({r.get('token')}), during {during(r['t'], marks)}: "
        f"{r['method']} {r['path']}"
        for r in unsigned
    ]
    writes = [
        r for r in log if r.get("write") and not r.get("probe") and r not in unsigned
    ]
    probes = [r for r in log if r.get("probe")]
    start = next((t for t, name in marks if sends and name == sends), None)
    sent = [
        r
        for r in writes
        if start is not None
        and r["t"] >= start
        and r.get("token") == "write"
        and r["status"] == 200
    ]
    bad += [
        f"a write, during {during(r['t'], marks)}: {r['method']} {r['host']}{r['path']}"
        for r in writes
        if r not in sent
    ]
    if sends and not sent:
        bad.append(f"no write with the write login's token went out after {sends}")
    reads = [
        r
        for r in log
        if r["host"] == "api.github.com"
        and r["status"] == 200
        and r.get("token") == "login"
    ]
    clones = [
        r
        for r in log
        if r["host"] == "github.com"
        and r["path"].endswith("/git-upload-pack")
        and r["status"] == 200
    ]
    keyed = [r for r in log if r["host"] == GATEWAY and r.get("key_headers")]
    wrong = [r for r in keyed if r.get("key") != "right"]
    bad += [
        f"the gateway's key sent otherwise: {r['method']} {r['path']} in {', '.join(r['key_headers'])}"
        for r in wrong
    ]
    needed = [("no GitHub read answered", reads)]
    if clone:
        needed.append(("no clone of the repository", clones))
    if gateway:
        right = [r for r in keyed if r.get("key") == "right"]
        needed.append(("no call reached the gateway with its key", right))
    bad += [what for what, found in needed if not found]
    good.append(
        f"{len(reads)} GitHub reads with the read login's token, {len(clones)} fetches of the repository, "
        f"{len(keyed) - len(wrong)} gateway calls with the key in the {KEY_HEADER} header and as a bearer token, no "
        f"write but {len(probes)} login probes"
        + (
            f"; after {sends}, {len(sent)} sent with the write login's token"
            if sends
            else ""
        )
    )
    others = {}
    for r in log:
        if r["status"] != 200 and not r.get("probe") and r not in unsigned:
            said = (
                f"answered {r['status']}: {r['method']} {r['host']}{r['path'].split('?')[0]}, during "
                f"{during(r['t'], marks)}"
            )
            others[said] = others.get(said, 0) + 1
    good += [f"{said} ({n}x)" for said, n in sorted(others.items())]
    return bad, good


def before_confirm_findings(log: list, marks: list, own: str) -> list:
    """Each docker call made before confirm (the guard's lines of LOG from before confirm's start in MARKS) that acts
    on, or reads, a named object other than through a listing filtered on OWN, the settings file's container.name,
    exactly: by its compose project's label, or by that one name (identity.exact_name), as docker reads a name filter
    as a pattern that may match a part of another deployment's names. That listing is check 15's minimal read, where a
    listing with no filter would read the whole host.
    """
    from rkmon_lib import identity  # noqa: E402

    start = next((t for t, what in marks if what.startswith("./rkmon confirm")), None)
    exact = (["project", own], ["name", identity.exact_name(own)])
    bad = []
    for r in log:
        if start is not None and r["t"] >= start:
            continue
        names = r.get("names") or []
        if r["verdict"] == "narrowed":
            ok = bool(names) and all(n in exact for n in names)
        else:
            ok = r["verdict"] == "pass" and not names  # the engine's own questions
        if not ok:
            bad.append(
                f"during {during(r['t'], marks)}: docker {shlex.join(r['argv'])}"
            )
    return bad


def guard_findings(log: list) -> tuple:
    """(each docker command the guard refused, the count of each verdict) of LOG, the guard's lines."""
    refused = [
        f"refused: docker {shlex.join(r['argv'])} ({r.get('why', '')})"
        for r in log
        if r["verdict"] == "refused"
    ]
    counts = {}
    for r in log:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return refused, counts


class DeployPath:
    """One run of the deploy path by the stage named STAGE: this one, or one that walks the path to make the deployment
    it checks (the web-host stage's WebHost, the no-restart stage's NoRestart). Its scope, <names>-<stage>, names its
    objects and its temporary folder, and its results go under the stage's name (result).
    """

    def __init__(
        self,
        engine: str,
        image: str,
        pkg: str,
        names: str,
        run_opts: list,
        res,
        stage: str = STAGE,
    ):
        self.engine = engine
        self.image = image
        self.pkg = pkg
        self.run_opts = run_opts
        self.res = res
        self.stage = stage
        self.scope = re.sub(r"[^a-z0-9_-]", "-", f"{names}-{stage}".lower())
        # docker's name filter that lists only the names that begin with the scope: docker reads the filter as a
        # pattern, in which each character of the scope stands for itself
        self.begins = f"name=^{self.scope}"
        self.network = f"{self.scope}_default"
        self.fakes = f"{self.scope}-fakes"
        self.derived = (
            f"{image.rpartition(':')[0].rpartition('/')[2] or image}:{self.scope}"
        )
        self.top = pathlib.Path(tempfile.mkdtemp(prefix=f"{self.scope}-"))
        # the person's own files: the settings file and the secrets' files
        self.home = self.top / "home"
        self.copy = self.top / "package"
        self.file = self.home / "rkmon-deploy.yaml"
        self.key = "sk-fake-" + secrets.token_hex(16)
        self.token = github_token()  # the read login's
        self.made_image = False
        self.marks = []  # (when, what) of each rkmon command, as they start
        # a next line has said that a tick has run since the start (told)
        self.ticked = False
        # the last step's next line was the done line: the path is done (path)
        self.done = False

    def result(self, ok: bool, text: str) -> bool:
        """The stage's result TEXT, under its name: PASS when OK, else FAIL. Returns OK."""
        self.res.result("PASS" if ok else "FAIL", f"{self.stage}: {text}")
        return ok

    # --- docker, as the stage itself runs it -------------------------------------------------------------------------
    def docker(self, *args: str, stdin=None, timeout: float = 300) -> tuple:
        """`ENGINE ARGS` on the real engine, for the stage's own objects, with STDIN on its stdin when one is given: its
        exit code and its output."""
        try:
            p = subprocess.run(
                [self.engine, *args],
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124, f"{self.engine} {args[0]}: no answer in {timeout:g} s"
        return p.returncode, p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")

    def must(self, *args: str, timeout: float = 300) -> str:
        rc, out = self.docker(*args, timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"{self.engine} {' '.join(args[:2])} exited {rc}: {out[-1500:]}"
            )
        return out

    # --- the fresh machine -------------------------------------------------------------------------------------------
    def fakes_config(self) -> dict:
        """The stand-ins' configuration (tests/fakes/fake_net.py CONFIG): the read login's user and token, the
        repository, with no open PR, and the gateway, with its key sent in its own header and as a bearer token.
        """
        from rkmon_lib import deploy_tree  # noqa: E402

        return {
            "github": {"login": LOGIN, "token": self.token, "repository": REPOSITORY},
            "gateway": {
                "host": GATEWAY,
                "key": self.key,
                "header": KEY_HEADER,
                "scheme": "",
                "sent_as": "header_and_bearer",
                "models": list(deploy_tree.BUILTIN.values()),
            },
        }

    def prepare(self) -> None:
        """The package's copy, the deployment's network, the stand-ins, the image that trusts them, and the guard."""
        sys.path.insert(0, self.pkg)
        from rkmon_lib import identity, package_files  # noqa: E402

        self.home.mkdir(mode=0o700)
        package_files.copy(self.pkg, self.copy)
        self.must(
            "network",
            "create",
            "--internal",
            "--label",
            f"{identity.PROJECT}={self.scope}",
            "--label",
            "com.docker.compose.network=default",
            self.network,
        )
        code = self.top / "fakes"
        code.mkdir()
        tests = pathlib.Path(self.pkg) / "tests"
        for f in (
            tests / "fakes" / "fake_net.py",
            tests / "fakes" / "fake_servers.py",
            tests / "fakerepo.py",
        ):
            shutil.copy(f, code / f.name)
        (code / "config.json").write_text(
            json.dumps(self.fakes_config()), encoding="utf-8"
        )
        aliases = [
            w
            for n in ("github.com", "api.github.com", "codeload.github.com", GATEWAY)
            for w in ("--network-alias", n)
        ]
        uid, gid = FAKES_USER.split(":")
        self.must(
            "create",
            "--name",
            self.fakes,
            "--network",
            self.network,
            *aliases,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            FAKES_USER,
            "--tmpfs",
            f"{FAKES_STATE}:uid={uid},gid={gid},mode=0700,size=64m",
            *self.run_opts,
            "--entrypoint",
            "python3",
            self.image,
            "-I",
            "-B",
            f"{FAKES_DIR}/fake_net.py",
            "--config",
            f"{FAKES_DIR}/config.json",
            "--state",
            FAKES_STATE,
        )
        self.must("cp", f"{code}/.", f"{self.fakes}:{FAKES_DIR}")
        self.must("start", self.fakes)
        ready = ("exec", self.fakes, "test", "-f", f"{FAKES_STATE}/ready")
        for _ in range(60):
            if self.docker(*ready)[0] == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"the stand-ins did not start: {self.docker('logs', self.fakes)[1][-1500:]}"
            )
        ctx = self.top / "trust"
        (ctx / "ca").mkdir(parents=True)
        # docker cp reads no tmpfs: the stand-ins' user reads it out
        (ctx / "ca" / "image-test-root.crt").write_text(
            self.must("exec", self.fakes, "cat", f"{FAKES_STATE}/ca.crt") + "\n",
            encoding="utf-8",
        )
        (ctx / "Dockerfile").write_text(
            f"FROM {self.image}\n"
            "COPY ca/ /tmp/image-test-ca/\n"
            "RUN sh /opt/rkmon/build/pkgs.sh trust /tmp/image-test-ca && rm -rf /tmp/image-test-ca\n",
            encoding="utf-8",
        )
        self.must(
            "build", "--network", "none", "-t", self.derived, str(ctx), timeout=900
        )
        self.made_image = True
        bin_dir = self.top / "bin"
        bin_dir.mkdir()
        guard = {
            "engine": shutil.which(self.engine) or self.engine,
            "scope": self.scope,
            "images": [self.derived],
            "network": self.network,
            "log": str(self.top / "guard.log"),
        }
        (self.top / "guard.json").write_text(json.dumps(guard), encoding="utf-8")
        shim = bin_dir / os.path.basename(self.engine)
        program = tests / "fakes" / "docker_guard"
        shim.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -B {shlex.quote(str(program))} "
            f"{shlex.quote(str(self.top / 'guard.json'))} \"$@\"\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("RKMON_", "COMPOSE_"))
        }
        self.env["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        self.env["CONTAINER_ENGINE"] = os.path.basename(self.engine)
        # ./rkmon runs on its first line's python3, which then writes nothing into the package copy
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    # --- the person's commands ---------------------------------------------------------------------------------------
    def rkmon(
        self, *args: str, stdin: bytes = b"", timeout: float = STEP_TIMEOUT_S
    ) -> tuple:
        """`./rkmon ARGS` in the package copy's docker folder, as the guide types it, so that its exec bit and its
        first line are the ones that run it, with no terminal: its exit code and its output, shown.
        """
        argv = ["./rkmon", *args]
        self.marks.append((time.time(), command_label(args)))
        try:
            p = subprocess.run(
                argv,
                cwd=self.copy / "docker",
                env=self.env,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            rc = p.returncode
            out = p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")
        except subprocess.TimeoutExpired as exc:
            rc = 124
            out = (exc.stdout or b"").decode(
                "utf-8", "surrogateescape"
            ) + f"\n(no end in {timeout:g} s)"
        hostrun.indented(f"$ ./rkmon {shlex.join(args)}\n{out}")
        return rc, out

    def follow(self, out: str, *more: str, timeout: float = STEP_TIMEOUT_S) -> tuple:
        """The command OUT's next line gives (OUT: the output of a step that passed), with MORE after its words: its
        exit code and its output."""
        return self.rkmon(*next_command(next_line(out))[1], *more, timeout=timeout)

    def step(self, n: int, rc: int, out: str) -> bool:
        """Step N's result: exit 0, its outcome line, and its next line, the state's (told)."""
        said = next_line(out)
        ok = rc == 0 and bool(outcome(out, OUTCOMES[n])) and self.told(n, said)
        if not ok:
            hostrun.indented(
                f"step {n}: exit {rc}, outcome {outcome(out, OUTCOMES[n])!r}, next line {said!r}"
            )
        self.result(ok, f"step {n}, {WHAT[n]}")
        return ok

    def told(self, n: int, said: str) -> bool:
        """Whether SAID, step N's next line, is the state's (rkmon_lib/states.py next_line). After step 8, the done
        line. Before it, the command of the step after N, or of one SKIPS gives, after that step's number and name
        until a tick has run since the start. The container's own tick may run at any time from the start on (step
        6): after it, the line after step 6 names step 7's command alone, and the line after step 7 is the done line,
        a tick being step 8's work; once a line has said that a tick has run (self.ticked), no later one names a
        step's number and name."""
        from rkmon_lib import steps  # noqa: E402

        if said.startswith(DONE):
            return n >= 7
        k = next_command(said)[0]
        if n == 8 or k not in SKIPS.get(n, (n + 1,)):
            return False
        if said.startswith(f"next: {steps.step_words(steps.DEPLOY_STEPS[k - 1])} "):
            return not self.ticked
        self.ticked = n == 6
        return self.ticked

    def fill_in(self) -> None:
        """The values a person writes in the template, through the package's one writer of settings files: the stage's
        names, the stand-ins' repository and login, a free port, and the gateway with its key sent in its own header and
        as a bearer token."""
        from rkmon_lib import file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        v = doc.values
        v["container"]["name"] = self.scope
        v["image"]["name"], _, v["image"]["tag"] = self.derived.rpartition(":")
        v["repository"]["name"] = REPOSITORY
        v["repository"]["folder"] = self.rocke_path()
        v["repository"]["scope_labels"] = [LABEL]
        v["features"]["dashboards"]["admin"]["port"] = free_port()
        v["integrations"]["github_cli"]["username"] = LOGIN
        v["integrations"]["claude_code"].update(
            base_url=f"https://{GATEWAY}",
            api_key_sent_as="header_and_bearer",
            api_key_header=KEY_HEADER,
        )
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    def path(self, last: int = 8) -> int:
        """The guide's steps, each checked: step 1's command, then each next line's, with what the stage does without a
        terminal, and the stage's own checks (the module's docstring): after confirm, rkmon next (next_agrees) and
        image check; after doctor, when its next line is the done line, first-tick (own, first_tick_done). The path
        stops at the first step that fails, and is done when the last step's next line is the done line (self.done).
        A stage that runs a command of its own before a step, as `rkmon adopt` runs before the first start, walks the
        steps up to LAST alone. The last step that passed, 0 for none.
        """
        rc, out = self.rkmon("template", "deploy", "--profile", PROFILE, str(self.file))
        private = self.file.exists() and self.file.stat().st_mode & 0o777 == 0o600
        if not self.step(1, rc if private else 1, out):
            return 0
        self.fill_in()
        passed = 1
        for n in range(2, last + 1):
            if next_command(next_line(out))[0] != n:
                continue  # a step the next line skips, which step() allowed
            if n == 5:
                rc, out = self.logins(out)
            else:
                # confirm asks its question with no terminal to answer it on
                more = ("--yes",) if n == 3 else ()
                rc, out = self.follow(out, *more, timeout=step_timeout(n))
            if not self.step(n, rc, out):
                return passed
            passed, self.done = n, next_line(out).startswith(DONE)
            if n == 3:
                self.next_agrees(out)
                self.own(4, self.trusted)
            elif n == 7 and next_line(out).startswith(DONE):
                self.own(8, self.first_tick_done, timeout=step_timeout(8))
        return passed

    @staticmethod
    def first_tick_done(out: str) -> bool:
        """OUT, first-tick's, ends with the done line, and the rk status it shows of the first tick has no alarm line
        (ALARM), each of which it prints."""
        alarms = [x for x in out.splitlines() if x.lstrip().startswith(ALARM)]
        for line in alarms:
            hostrun.indented(f"an alarm after the first tick: {line.strip()}")
        return next_line(out).startswith(DONE) and not alarms

    def next_agrees(self, out: str) -> None:
        """`rkmon next` after confirm, whose output OUT is: exit 0, and confirm's next line, each being the state's."""
        rc, said = self.rkmon("next")
        ok = rc == 0 and next_line(said) == next_line(out)
        if not ok:
            hostrun.indented(
                f"rkmon next: exit {rc}, next line {next_line(said)!r}; confirm's {next_line(out)!r}"
            )
        self.result(ok, "rkmon next after confirm gives confirm's next line")

    def trusted(self, out: str) -> bool:
        """OUT, image check's, has the row of an image that trusts this network's TLS: here, the stand-ins' root."""
        from rkmon_lib import checks, deployment  # noqa: E402

        return checks.render(deployment.image_tls_checks("TLS ok")) in out

    def own(self, n: int, also, timeout: float = STEP_TIMEOUT_S) -> None:
        """A check of the stage's own, outside the guide's steps, of step N, whose work a next line skipped as done
        (the module's docstring): the step's check command, else its command (rkmon_lib/steps.py), with no options;
        exit 0, the step's outcome line, and ALSO, true of its output. Step 4's, image check: confirm checked that the
        image was built from this package's image recipe; image check checks it against this package's version, and
        its trust of this network's TLS (trusted)."""
        from rkmon_lib import steps  # noqa: E402

        step = steps.DEPLOY_STEPS[n - 1]
        words = steps.command_name(step.check or step.command)
        rc, out = self.rkmon(*words.split(), timeout=timeout)
        ok = rc == 0 and bool(outcome(out, OUTCOMES[n])) and also(out)
        self.result(ok, f"{words}, a check outside the guide's steps: {OWN[n]}")

    def rocke_path(self) -> str:
        sys.path.insert(0, str(pathlib.Path(self.pkg) / "tests"))
        import fakerepo  # noqa: E402

        return fakerepo.ROCKE_PATH

    def values(self) -> dict:
        """What the stage gives each login and secret the walk may ask for, by its name: (how it is set, the value).
        "token": a GitHub login, `login NAME --with-token`; "key": the claude login, `login claude`, the key alone,
        which goes where the settings say, and as they say (fill_in); each with the value on stdin. "file": a secret,
        `secret NAME FILE`, the value in a file of the person's.
        """
        return {
            "github_read": ("token", self.token + "\n"),
            "claude": ("key", self.key + "\n"),
            "admin_access_token": ("file", secrets.token_urlsafe(32) + "\n"),
        }

    def logins(self, out: str) -> tuple:
        """Step 5, from OUT, the output whose next line gives its command: the walk, with no terminal to ask on, lists
        what is missing or wrong (walk_names); each of them is set as `values` says, a secret from a file of the
        person's; then the walk again: its exit code and its output."""
        _rc, walk = self.follow(out)
        values, failed = self.values(), []
        for name in walk_names(walk):
            if name not in values:
                failed.append(
                    f"{name}: the walk asks for it, and the stage has no value for it"
                )
                continue
            how, value = values[name]
            if how == "file":
                path = self.home / name
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as w:
                    w.write(value)
                words = ("secret", name, str(path))
                rc, _out = self.rkmon(*words)
            else:
                words = ("login", name, *(("--with-token",) if how == "token" else ()))
                rc, _out = self.rkmon(*words, stdin=value.encode())
            failed += [f"{shlex.join(words)}: exit {rc}"] if rc else []
        rc, walk = self.follow(out)
        if failed:
            hostrun.indented("failed: " + "; ".join(failed))
        return (1 if failed else rc), walk

    # --- what the deployment runs ------------------------------------------------------------------------------------
    def differences(self) -> tuple:
        """`rk config show --source` against the settings file: (its exit code, the keys the file sets at
        their values, updates left out, which only the host reads; how config show differs from them:
        deploy_differences)."""
        from rkmon_lib import deploy_plan, file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        maps = settings_file.map_keys(file_types.DEPLOY.load())
        values = {
            name: value
            for name, value in file_keys(doc.values, maps).items()
            if name.split(".")[0] not in deploy_plan.LEFT_OUT
        }
        rc, out = self.rkmon("rk", "config", "show", "--source", timeout=300)
        bad = deploy_differences(out, values)
        said = f"config show: exit {rc}, the file sets {len(values)} keys, {len(bad)} shown otherwise"
        hostrun.indented("\n".join([said, *bad]))
        return rc, values, bad

    def sources(self) -> None:
        """Once the monitor has started: `rk config show --source` shows deploy for exactly the keys the settings file
        sets, each at the file's value (differences)."""
        rc, _values, bad = self.differences()
        self.result(
            rc == 0 and not bad,
            "rk config show --source shows deploy for exactly the keys the settings file sets, at their values",
        )

    def settings_read_only(self) -> None:
        """Once the monitor has started: nothing in its container writes the settings volume, which docker/compose.yaml
        mounts read-only at container_layout.SETTINGS_DIR. As the monitor's user and as root, a new file in the folder
        and the layer file opened to add to it are refused, root's by the read-only mount (WRITE_TRIES,
        read_only_findings)."""
        from rkmon_lib import container_layout as L  # noqa: E402

        layer, tries = f"{L.SETTINGS_DIR}/{L.LAYER_FILE}", {}
        for user in (L.MONITOR.name, "root"):
            rc, out = self.inside(
                "python3", "-I", "-c", WRITE_TRIES, L.SETTINGS_DIR, layer, user=user
            )
            try:
                tries[user] = json.loads(out.splitlines()[-1]) if rc == 0 else out
            except (ValueError, IndexError):
                tries[user] = out
        found = read_only_findings(tries)
        hostrun.indented("\n".join([json.dumps(tries), *found]))
        self.result(
            not found,
            "nothing in the running container writes the settings volume: as the monitor's user and as root, a new "
            "file in it and a write to its layer file are refused, root's by the read-only mount",
        )

    def monitor(self) -> str:
        """The deployment's container, by the labels docker compose gives it: its project's, its service's, no
        one-off."""
        from rkmon_lib import deploy_plan, identity  # noqa: E402

        names = self.must(
            "ps",
            "--format",
            "{{.Names}}",
            "--filter",
            f"label={identity.PROJECT}={self.scope}",
            "--filter",
            f"label=com.docker.compose.service={deploy_plan.SERVICE}",
            "--filter",
            "label=com.docker.compose.oneoff=False",
        ).split()
        if len(names) != 1:
            raise RuntimeError(f"the deployment's container: {len(names)} run, not 1")
        return names[0]

    def inside(
        self,
        *argv: str,
        user: str = "",
        stdin: bytes = b"",
        env=None,
        workdir: str = "",
        timeout: float = 300,
    ) -> tuple:
        """ARGV run in the deployment's container (monitor) with docker exec, as USER, the monitor's user when none is
        given: with STDIN, the environment ENV beside the container's, and in the folder WORKDIR when one is given. Its
        exit code and its output."""
        from rkmon_lib import container_layout  # noqa: E402

        opts = ["-u", user or container_layout.MONITOR.name]
        opts += [w for k, v in (env or {}).items() for w in ("-e", f"{k}={v}")]
        opts += ["-w", workdir] if workdir else []
        where = self.monitor()
        return self.docker(
            "exec", "-i", *opts, where, *argv, stdin=stdin, timeout=timeout
        )

    def must_inside(self, *argv: str, **how) -> str:
        """ARGV's output, run in the deployment's container as inside runs it (HOW: its keywords); RuntimeError when it
        fails."""
        rc, out = self.inside(*argv, **how)
        if rc != 0:
            raise RuntimeError(
                f"{shlex.join(argv[:2])} in the deployment's container exited {rc}: {out[-1500:]}"
            )
        return out

    def as_monitor(self, *args: str) -> str:
        """ARGS run in the deployment's container as the monitor's user, in the folder of the code it runs (`rk
        --which-code`), where python3 imports the package: their output; RuntimeError when they fail (must_inside).
        """
        code = self.must_inside("rk", "--which-code").strip()
        return self.must_inside(*args, workdir=code)

    def change_a_setting(self) -> None:
        """Once the path is done, CHANGE made as a person changes a setting: its line edited in the
        settings file; `rkmon confirm`, which shows the change and whose next line names apply; then the command that
        line gives, `rkmon apply`, and each next line after it to the done line (to_done: the change restarts the
        container); after which `rk config show --source` shows deploy for exactly the keys the file sets, at their
        values, the changed one at its new value (differences).
        """
        name, value = CHANGE
        self.file.write_text(
            edited(self.file.read_text(encoding="utf-8"), name, value), encoding="utf-8"
        )
        # confirm asks its question with no terminal to answer it on
        rc, out = self.rkmon("confirm", "--yes")
        found = outcome(out, CHANGE_OUTCOMES["confirm"])
        # the change's line: its key, then what runs now -> what apply makes it, each in JSON
        shown = [
            x.strip()
            for x in out.splitlines()
            if x.strip().startswith(f"~ {name} ")
            and x.rstrip().endswith(f" -> {json.dumps(value)}")
        ]
        ok = (
            rc == 0
            and len(shown) == 1
            and bool(found)
            and next_command(next_line(out))[1] == ["apply"]
        )
        said = [
            f"confirm: exit {rc}, the change's line {shown!r}, outcome {found!r}, next line {next_line(out)!r}"
        ]
        if ok:
            rc, out = self.follow(out)
            found = outcome(out, CHANGE_OUTCOMES["apply"])
            ok = rc == 0 and bool(found)
            said.append(
                f"apply: exit {rc}, outcome {found!r}, next line {next_line(out)!r}"
            )
        ok = ok and self.to_done(out, said)
        hostrun.indented("\n".join(said))
        if ok:
            rc, values, bad = self.differences()
            ok = rc == 0 and not bad and values.get(name) == value
        self.result(
            ok,
            f"{name} changed as a person changes a setting, edited in the settings file, confirmed and applied, each "
            "next line then followed to the done line, after which rk config show --source shows the file's keys at "
            "their values",
        )

    def to_done(self, out: str, said: list) -> bool:
        """From OUT, a command's output, each next line followed while it names deploy step 7's or step 8's command,
        as the state walks them again after a start (rkmon_lib/states.py next_line): whether the last next line is the
        done line, each step followed having exited 0 with its outcome line. What each step did goes into SAID.
        """
        for n in (7, 8):
            if next_line(out).startswith(DONE):
                break
            if next_command(next_line(out))[0] != n:
                continue
            rc, out = self.follow(out, timeout=step_timeout(n))
            found = outcome(out, OUTCOMES[n])
            said.append(
                f"step {n}: exit {rc}, outcome {found!r}, next line {next_line(out)!r}"
            )
            if rc != 0 or not found:
                return False
        return next_line(out).startswith(DONE)

    def stop_drained(self) -> None:
        """The monitor, once it has started, stopped as a person stops it: `rkmon stop --drain`, the
        work in flight given STOP_DRAIN_MINUTES to finish (stopped_drained); after which docker lists no running
        container of the deployment's compose project."""
        from rkmon_lib import identity  # noqa: E402

        rc, out = self.rkmon(
            "stop",
            "--drain",
            "--timeout-minutes",
            str(STOP_DRAIN_MINUTES),
            # the drain's limit, then a step's time for rk stop and docker compose stop
            timeout=STOP_DRAIN_MINUTES * 60 + STEP_TIMEOUT_S,
        )
        code, names = self.docker(
            "ps",
            "--format",
            "{{.Names}}",
            "--filter",
            f"label={identity.PROJECT}={self.scope}",
        )
        running = names.split() if code == 0 else [f"docker ps exited {code}"]
        ok = rc == 0 and stopped_drained(out) and not running
        if not ok:
            hostrun.indented(
                f"stop --drain: exit {rc}, outcome {outcome(out, STOP_OUTCOME)!r}, next line {next_line(out)!r}; "
                f"still running: {running}"
            )
        self.result(
            ok,
            "rkmon stop --drain lets the work in flight finish, then stops the monitor and its container",
        )

    # --- what the deployment did -------------------------------------------------------------------------------------
    def requests(self) -> list:
        """The stand-ins' log (tests/fakes/fake_net.py): each request, as a dict; [] when it cannot be read."""
        rc, out = self.docker("exec", self.fakes, "cat", f"{FAKES_STATE}/requests.log")
        return (
            [json.loads(x) for x in out.splitlines() if x.startswith("{")]
            if rc == 0
            else []
        )

    def after(self, sends: str = "", clone: bool = True, gateway: bool = True) -> None:
        """What the deployment did: the stand-ins' log (requests_findings: SENDS, the command from which on it sends;
        CLONE, whether it cloned the repository; GATEWAY, whether it reached the gateway with its key), then the docker
        commands it ran (docker_calls).
        """
        bad, good = requests_findings(
            self.requests(), self.marks, sends, clone, gateway
        )
        hostrun.indented("\n".join(bad + good))
        did = (
            "the deployment read GitHub with its read login"
            + (", cloned the repository" if clone else "")
            + (", reached the Claude gateway with its key" if gateway else "")
        )
        self.result(
            not bad,
            (
                f"{did}, wrote nothing before {sends}, and after it sent with its write login"
                if sends
                else f"{did}, and wrote nothing"
            ),
        )
        self.docker_calls()

    def docker_calls(self) -> None:
        """The docker commands the deployment ran, from the guard's log (tests/fakes/docker_guard): each one its own,
        every listing narrowed to its objects; before confirm, none that touched an object but through a
        listing filtered exactly on its own name (before_confirm_findings)."""
        lines = (
            (self.top / "guard.log").read_text(encoding="utf-8").splitlines()
            if (self.top / "guard.log").exists()
            else []
        )
        calls = [json.loads(x) for x in lines]
        refused, counts = guard_findings(calls)
        hostrun.indented(
            "\n".join(
                refused + [f"docker commands: {json.dumps(counts, sort_keys=True)}"]
            )
        )
        self.result(
            not refused,
            "every docker command of the path was the deployment's own, and every listing showed only its objects",
        )
        early = before_confirm_findings(calls, self.marks, self.scope)
        if early:
            hostrun.indented("\n".join(early))
        self.result(
            not early,
            "before confirm, no docker command touched an object but through listings filtered exactly on "
            "the deployment's own name",
        )

    # --- the end -----------------------------------------------------------------------------------------------------
    def left(self) -> list:
        """What is left on this host of the deployment: each container, volume and network docker compose labeled for
        its compose project, or whose name begins with the scope; and each listing docker did not answer.
        """
        from rkmon_lib import identity  # noqa: E402

        found = set()
        for args in (
            ("container", "ls", "--all", "--format", "{{.Names}}"),
            ("volume", "ls", "--format", "{{.Name}}"),
            ("network", "ls", "--format", "{{.Name}}"),
        ):
            for by in (f"label={identity.PROJECT}={self.scope}", self.begins):
                rc, out = self.docker(*args, "--filter", by)
                if rc != 0:
                    found.add(f"{args[0]} ls --filter {by}: exit {rc}: {out[-200:]}")
                    continue
                found.update(
                    f"{args[0]} {n}"
                    for n in out.split()
                    if by.startswith("label=") or n.startswith(self.scope)
                )
        return sorted(found)

    def clean(self) -> None:
        """Every object of the scope removed, and the temporary folder: the stand-ins' container first, as it is on
        the deployment's network; then the deployment, once confirm has recorded it, by its own `rkmon remove
        --delete-data --yes`, with a result that nothing of it is left (left); then whatever is left of the scope, by
        name, each name checked to begin with the scope."""
        self.docker("rm", "-f", self.fakes)
        if (self.copy / "docker" / ".rkmon").exists() and hasattr(self, "env"):
            rc, _out = self.rkmon("remove", "--delete-data", "--yes", timeout=300)
            left = self.left()
            if left:
                hostrun.indented("left: " + "; ".join(left))
            self.result(
                rc == 0 and not left,
                "rkmon remove --delete-data --yes leaves nothing of the deployment on this host",
            )

        def ours(names: str) -> list:
            return [n for n in names.split() if n.startswith(self.scope)]

        _rc, out = self.docker(
            "ps", "-a", "--format", "{{.Names}}", "--filter", self.begins
        )
        for name in ours(out):
            self.docker("rm", "-f", "-v", name)
        _rc, out = self.docker(
            "volume", "ls", "--format", "{{.Name}}", "--filter", self.begins
        )
        for name in ours(out):
            self.docker("volume", "rm", "-f", name)
        _rc, out = self.docker(
            "network", "ls", "--format", "{{.Name}}", "--filter", self.begins
        )
        for name in ours(out):
            self.docker("network", "rm", name)
        if self.made_image:
            self.docker("rmi", self.derived)
        shutil.rmtree(self.top, ignore_errors=True)


def deploy_path_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = DeployPath(engine, image, pkg, names, run_opts, res)
    try:
        t.prepare()
        if t.path() >= 6:  # the monitor has started
            t.sources()
            t.settings_read_only()
            if t.done:
                t.change_a_setting()
            t.stop_drained()
        t.after()
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
