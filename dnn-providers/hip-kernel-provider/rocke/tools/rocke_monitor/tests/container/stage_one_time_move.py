# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's one-time-move stage (tests/container/image.py): the one-time move as this
package's first start makes it on an older install's work volume, on the image under test, and its end, which
`rkmon standby off` makes.

The deployment is the deploy path's (tests/container/stage_deploy_path.py, DeployPath: the guide's steps on a fresh
machine, each checked, its results under this stage's name), walked to step 5, the logins. Its objects are named after
the scope <names>-one-time-move. Before step 1, the stage makes its work volume and its secrets volume an
older install's, under the names and labels docker compose gives the deployment's own, every value invented
(`older_files`): the monitor's home holds the config.json an older setup wrote (tests/data/older_setup_config.json,
with values of the install's own: OLDER) and the monitor's state, and the secrets volume holds the older copies of the
secrets as an older package kept them, the monitor's user's, mode 600. An older install's first start sets nothing up
(rkmon_lib/provision.py: the older package did), so the volume holds no clone, and the stage runs no doctor and no
first tick; step 5's walk takes the claude login's older copy without calling the gateway, and the stage makes the
test call of the login the move copied instead, once the move has ended: what the deployment did is checked as the
deploy path checks it, but for the clone (DeployPath.after).

In order:
- step 5's walk takes the claude login's older copy for the login set: the stage sets no claude login.
- `rkmon next`, then the start it gives: STARTED, in standby, and its next line gives doctor. Then, in the container:
  - the move's report in the start's log (logs/start.log), once: config.json moved, split by owner; the state key
    carried (CARRIED); the private key's line with neither value, which the move's record keeps; a package key's value
    not carried; a name no key has, dropped; and reviews.unnamed_are_ours_until at the move's time (MOVE_REPORT);
  - the move's record, open, of config.json's bytes; in the internal state, the volume an older install's, standby on
    for the move (one_time_move.REASON), reviews.unnamed_are_ours_until at the move's time, and the key carried;
  - root's copies (docker/rk_root.py), each in its own place, its keeper's user's, mode 600, holding what the move
    copies (COPIES), with root's line of each in the container's log; config.json and each older copy as the stage
    made them.
- `rkmon stop --drain`, as the deploy path stops the monitor (DeployPath.stop_drained), then `rkmon start` again before
  standby off: STARTED, in standby, and its next line gives doctor. Once rk start has ended again in the start's log
  (its last line, rkmon_lib/control.py CLAUDE_SIDE), the move is not made again: its record is the one the first start
  wrote, the start's log reports one move, and config.json and the copies are as the move left them.
- `rkmon standby off`: RELEASED, its lines saying config.json renamed and each older copy removed (END), and its next
  line gives doctor, which has not passed since the start. In the container: config.json.migrated holds config.json's
  bytes, read-only, and config.json is gone; each older copy is gone and each new copy still there; the move's record
  ended at the time reviews.unnamed_are_ours_until says; standby is off.
- `rk claude-test` in the container, as the monitor's user: the claude login the move copied, the one the session
  reads, makes its one test call, sent as the settings say, and its row passes.
Each check that the next ones need ends the stage when it fails: the walk, the start and the move. Then what the
deployment did (DeployPath.after, the clone not asked for): it read GitHub with its read login, reached the gateway
with its key once the login's test call has been made, and wrote nothing, and each docker command it ran was its own;
and the end (DeployPath.clean).
"""

import hashlib
import json
import os
import pathlib
import re
import time

import hostrun
import stage_deploy_path
from stage_deploy_path import (
    GATEWAY,
    KEY_HEADER,
    MOVE_WAIT_S,
    RELEASED,
    STARTED_IN_STANDBY,
    WORK_VOLUME,
    next_command,
    next_line,
    outcome,
    polled,
    says,
)

STAGE = "one-time-move"
# config.json as an older setup wrote it, with values of the older install's own over it: its repository, a private
# deploy key, which the deployment's settings name otherwise; a state key the move carries; a package key's value of
# its own; and a name no key of this package has.
OLDER_SETUP = ("tests", "data", "older_setup_config.json")
OLDER = {
    "repo": "acme/gadgets",
    "clone_since": "2026-01-05",
    "drift_batch": 97,
    "no_such_key": 1,
}
CARRIED = "clone_since"
# The container's folders: the monitor's home and its config.json, the secrets volume and the write volume.
HOME = "/work/monitor"
CONFIG_JSON = f"{HOME}/config.json"
MIGRATED = f"{CONFIG_JSON}.migrated"
SECRETS = "/secrets"
WRITE_SECRETS = "/secrets-write"
START_LOG = "/work/logs/start.log"
# The volumes an older install leaves, by the names docker compose gives them in the deployment's compose project, and
# where the container mounts each.
VOLUMES = {WORK_VOLUME: "/work", "rkmon-secrets": SECRETS}
# The older copies root copies at the first start (docker/rk_root.py COPIES, CLAUDE): each older copy's file, its new
# copy's, and the new copy's user: byte for byte, or the claude login's key alone, a line.
COPIES = {
    f"{SECRETS}/teams_webhook": (f"{WRITE_SECRETS}/teams_webhook", "rkwrite"),
    f"{SECRETS}/claude_env": (f"{SECRETS}/claude_key", "rkmon"),
}
# The older files no entry claims, which the end removes (secrets_registry.OLDER_SECRET_FILES).
UNCLAIMED = (f"{SECRETS}/worker_headers", f"{SECRETS}/jira_mcp_headers")
# The move's report in the start's log: a pattern for each of its lines the stage reads, OLDER's values in them,
# searched for in the log's text, each run of white space read as one space, however a line is wrapped.
MOVE_REPORT = (
    rf"moved: {re.escape(CONFIG_JSON)}, an older install's configuration, split by owner, once; the monitor is in "
    r"standby: it sends nothing until you run \S+ standby off",
    rf"carried into the internal state: ([a-z0-9_.]+, )*{CARRIED}\b",
    r"repo: config\.json's value is not carried; your deploy settings' repository\.name runs",
    r"drift_batch: config\.json's 97 is not carried; the package's \S+ runs",
    r"dropped \(no key of this package has the name\): ([a-z0-9_.]+, )*no_such_key\b",
)
# The end's lines (rkmon_lib/one_time_move.py end): config.json renamed, and each older copy removed.
END = (
    f"renamed {CONFIG_JSON} config.json.migrated, read-only",
    f"removed {SECRETS}/teams_webhook: an older copy of teams_webhook, which is set in its own place",
    f"removed {SECRETS}/claude_env: an older copy of claude, which is set in its own place",
    *(
        f"removed {p}: an older package kept a secret there, which nothing reads now"
        for p in UNCLAIMED
    ),
)
UNTIL = "reviews.unnamed_are_ours_until"
# What the monitor's home says of the move, as JSON: python3 -c, as the monitor, in the folder of the code it runs
# (DeployPath.as_monitor). The record's report keeps each value; the start's log shows none of a private key.
FACTS = (
    "import json, pathlib\n"
    "from rkmon_lib import container_layout, one_time_move, provision, registry, state_store\n"
    "home = pathlib.Path(container_layout.MONITOR_HOME)\n"
    "state = state_store.read(home)\n"
    "def get(name):\n"
    "    v = registry.get(state, name)\n"
    "    return None if v is registry.MISSING else v\n"
    "rec = one_time_move.record(home) or {}\n"
    f"log = pathlib.Path({START_LOG!r})\n"
    "print(json.dumps({\n"
    "    'older_install': get(provision.OLDER_INSTALL),\n"
    "    'standby': get('standby.on'),\n"
    "    'reason': get('standby.reason'),\n"
    "    'until': get(one_time_move.UNTIL),\n"
    f"    'carried': get({CARRIED!r}),\n"
    "    'record': {k: rec.get(k) for k in ('moved_at', 'ended_at', 'config_json_sha256', 'report')},\n"
    "    'log': [x.strip() for x in log.read_text().splitlines()] if log.is_file() else [],\n"
    "}))\n"
)
# Each file named after the code, as root reads it in the container: its owner's name, its mode and the sha256 of its
# bytes, or None where there is none; as JSON. No value is ever printed.
FILES = (
    "import hashlib, json, os, pwd, stat, sys\n"
    "out = {}\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n"
    "        s = os.lstat(p)\n"
    "    except FileNotFoundError:\n"
    "        out[p] = None\n"
    "        continue\n"
    "    with open(p, 'rb') as f:\n"
    "        sha = hashlib.sha256(f.read()).hexdigest()\n"
    "    out[p] = {'owner': pwd.getpwuid(s.st_uid).pw_name, 'mode': stat.S_IMODE(s.st_mode), 'sha256': sha}\n"
    "print(json.dumps(out))\n"
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def older_files(pkg: str, key: str) -> dict:
    """What an older install left in the two volumes, by each file's path in the container: its text, every value
    invented. config.json, OLDER_SETUP with OLDER; the monitor's state as its store writes one (rkmon_lib/store.py);
    and the older copies of the secrets: the team chat's webhook, the claude login's older settings, which carry KEY,
    the stand-ins' gateway's, as the deployment's settings send it (integrations.claude_code: DeployPath.fill_in), and
    the two files no entry claims (UNCLAIMED).
    """
    from rkmon_lib import store  # noqa: E402

    config = json.loads(pathlib.Path(pkg, *OLDER_SETUP).read_text(encoding="utf-8"))
    config.update(OLDER)
    claude_env = {
        "ANTHROPIC_BASE_URL": f"https://{GATEWAY}",
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_CUSTOM_HEADERS": f"{KEY_HEADER}: {key}",
    }
    files = {
        CONFIG_JSON: json.dumps(config, indent=1) + "\n",
        f"{HOME}/state/state.json": json.dumps(
            {**store.empty_state(), "enabled": True}
        ),
        f"{SECRETS}/teams_webhook": "https://hooks.example.com/workflows/older\n",
        f"{SECRETS}/claude_env": json.dumps(claude_env) + "\n",
    }
    files.update({p: f"X-Invented-{i}: {i}\n" for i, p in enumerate(UNCLAIMED)})
    return files


def copied(files: dict, key: str) -> dict:
    """What each new copy holds once root has made it, from FILES (older_files): {its path: (its user, the sha256 of
    its bytes)}: the older copy's bytes, or the claude login's KEY alone, a line."""
    out = {}
    for older, (new, user) in COPIES.items():
        text = f"{key}\n" if older.endswith("/claude_env") else files[older]
        out[new] = (user, sha256(text))
    return out


def login_findings(rc: int, out: str) -> list:
    """How `rk claude-test`'s exit code RC and output OUT differ from a claude login that works: exit 0, and the
    login's row, claude, PASS: its one test call answered (rkmon_lib/reach.py claude_test). [] when they do not.
    """
    from rkmon_lib import deployment, secrets_registry  # noqa: E402

    rows = [
        r.status
        for r in deployment.doctor_rows(out)
        if r.name == secrets_registry.CLAUDE_LOGIN
    ]
    if rc == 0 and rows == ["PASS"]:
        return []
    return [f"rk claude-test: exit {rc}, the claude login's rows {rows}"]


def report_findings(facts: dict, moved_at: str) -> list:
    """How the start's log in FACTS (the monitor's home's: FACTS) differs from the move's report (MOVE_REPORT), made
    at MOVED_AT, once, with no private value; and whether the move's record keeps that value. [] when it does not.
    """
    text = " ".join(" ".join(facts.get("log") or []).split())
    found = []
    for pattern in MOVE_REPORT:
        if not re.search(pattern, text):
            found.append(f"the start's log has no line {pattern!r}")
    moves = text.count(f"moved: {CONFIG_JSON}")
    if moves != 1:
        found.append(f"the start's log reports {moves} moves, not 1")
    if not moved_at or f"{UNTIL} is {moved_at}: " not in text:
        found.append(f"the start's log does not say that {UNTIL} is {moved_at}")
    repo = OLDER["repo"]
    if repo in text:
        found.append(f"the start's log shows {repo}, a private key's value")
    if not any(repo in x for x in (facts.get("record") or {}).get("report") or []):
        found.append(f"the move's record does not keep {repo}")
    return found


def starts_ended(facts: dict) -> int:
    """How many times rk start has ended in the start's log in FACTS (FACTS: each line, stripped): the lines that begin
    as rk start's last line does (rkmon_lib/control.py CLAUDE_SIDE), which it writes at each start of the container,
    after the move's report when that start makes the move."""
    from rkmon_lib import control  # noqa: E402

    return sum(x.startswith(control.CLAUDE_SIDE) for x in facts.get("log") or [])


def moved_findings(facts: dict, files: dict, made: dict, key: str) -> list:
    """How FACTS and FILES (the container's: FACTS, FILES) differ from the move as the first start makes it on the
    older install whose files the stage MADE (older_files), the older copies' first FILES; [] when they do not.
    """
    from rkmon_lib import one_time_move  # noqa: E402

    rec = facts.get("record") or {}
    found = []
    moved_at = rec.get("moved_at") or ""
    if not moved_at or rec.get("ended_at"):
        found.append(
            f"no open record of a move: {rec.get('moved_at')!r}, ended {rec.get('ended_at')!r}"
        )
    if rec.get("config_json_sha256") != sha256(made[CONFIG_JSON]):
        found.append("the move's record is not of config.json's bytes")
    if facts.get("older_install") is not True:
        found.append(
            "the internal state does not say that the volume is an older install's"
        )
    if facts.get("standby") is not True or facts.get("reason") != one_time_move.REASON:
        found.append(
            f"standby {facts.get('standby')!r} for {facts.get('reason')!r}, not on for the move"
        )
    if not moved_at or facts.get("until") != moved_at:
        found.append(f"{UNTIL} is {facts.get('until')!r}, not the move's time")
    if facts.get("carried") != OLDER[CARRIED]:
        found.append(
            f"{CARRIED} is {facts.get('carried')!r}, not config.json's {OLDER[CARRIED]!r}"
        )
    found += report_findings(facts, moved_at)
    for path, text in made.items():
        want = {"owner": "rkmon", "mode": 0o600, "sha256": sha256(text)}
        if path.startswith(SECRETS + "/") or path == CONFIG_JSON:
            if files.get(path) != want:
                found.append(
                    f"{path} is not as the older install left it: {files.get(path)}"
                )
    if files.get(MIGRATED) is not None:
        found.append(f"{MIGRATED} is there before the end")
    for path, (user, sha) in copied(made, key).items():
        want = {"owner": user, "mode": 0o600, "sha256": sha}
        if files.get(path) != want:
            found.append(
                f"{path} is not the copy the move makes, {user}'s, mode 600: {files.get(path)}"
            )
    return found


def ended_findings(facts: dict, files: dict, made: dict, key: str) -> list:
    """How FACTS and FILES differ from the end of the move, as standby off makes it: config.json.migrated config.json's
    bytes, read-only, and config.json gone; each older copy gone, each new copy still there; the move's record ended
    at the time reviews.unnamed_are_ours_until says; standby off. [] when they do not.
    """
    rec = facts.get("record") or {}
    found = []
    migrated = files.get(MIGRATED) or {}
    if migrated.get("sha256") != sha256(made[CONFIG_JSON]) or files.get(CONFIG_JSON):
        found.append(
            "config.json.migrated is not config.json's bytes, or config.json is still there"
        )
    if migrated.get("mode", 0o222) & 0o222:
        found.append(f"config.json.migrated is not read-only: {migrated}")
    for path in [*COPIES, *UNCLAIMED]:
        if files.get(path) is not None:
            found.append(f"the older copy {path} is still there")
    for path, (user, sha) in copied(made, key).items():
        if (files.get(path) or {}).get("sha256") != sha:
            found.append(f"the new copy {path} is not there as the move made it")
    if not rec.get("ended_at") or facts.get("until") != rec.get("ended_at"):
        found.append(
            f"the move's record ended at {rec.get('ended_at')!r}, {UNTIL} is {facts.get('until')!r}"
        )
    if facts.get("standby") is not False:
        found.append(f"standby is {facts.get('standby')!r}")
    return found


class OneTimeMove(stage_deploy_path.DeployPath):
    """One run of the stage: the deploy path's (DeployPath) on an older install's volumes, to step 5, then the move."""

    def __init__(
        self, engine: str, image: str, pkg: str, names: str, run_opts: list, res
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        self.made = {}
        self.record = None  # the move's record, as the first start wrote it (moved)

    def prepare(self) -> None:
        """The deploy path's fresh machine (DeployPath.prepare), then the older install's volumes (older_volumes)."""
        super().prepare()
        self.older_volumes()

    def older_volumes(self) -> None:
        """The work volume and the secrets volume of an older install (older_files), made under the names and labels
        docker compose gives the deployment's own (VOLUMES), and filled as root in a one-off container of the image,
        with no network: each file the monitor's user's, mode 600."""
        from rkmon_lib import identity  # noqa: E402

        self.made = older_files(self.pkg, self.key)
        seed = self.top / "older"
        for path, text in self.made.items():
            p = seed / path.lstrip("/")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        mounts = []
        for volume, where in VOLUMES.items():
            name = f"{self.scope}_{volume}"
            self.must(
                "volume",
                "create",
                "--label",
                f"{identity.PROJECT}={self.scope}",
                "--label",
                f"com.docker.compose.volume={volume}",
                name,
            )
            mounts += ["-v", f"{name}:{where}"]
        secrets_files = " ".join(p for p in self.made if p.startswith(SECRETS + "/"))
        script = (
            "set -e; cp -R /seed/work/. /work/; cp -R /seed/secrets/. /secrets/; "
            f"chown -R rkmon:rkmon {HOME} {secrets_files}; chmod 600 {' '.join(self.made)}"
        )
        filler = f"{self.scope}-older"
        self.must(
            "create",
            "--name",
            filler,
            "--network",
            "none",
            "--user",
            "0:0",
            *mounts,
            "--entrypoint",
            "sh",
            self.derived,
            "-c",
            script,
        )
        try:
            self.must("cp", f"{seed}/.", f"{filler}:/seed")
            self.must("start", "--attach", filler)
        finally:
            self.docker("rm", "-f", filler)

    # --- what the stage reads ----------------------------------------------------------------------------------------
    def read(self, ready) -> tuple:
        """(FACTS, FILES) once READY is true of FACTS, or MOVE_WAIT_S have passed: the container's own start makes the
        move, which may finish after the start has said STARTED. FILES: each file the stage made, and each new copy and
        config.json.migrated, as root reads them."""
        facts = polled(
            lambda: json.loads(
                self.as_monitor("python3", "-c", FACTS).splitlines()[-1]
            ),
            ready,
            MOVE_WAIT_S,
        )
        paths = [*self.made, MIGRATED, *(new for new, _user in COPIES.values())]
        out = self.must_inside("python3", "-I", "-c", FILES, *paths, user="0")
        return facts, json.loads(out.splitlines()[-1])

    # --- the checks --------------------------------------------------------------------------------------------------
    def walked(self) -> bool:
        """Step 5's walk took the claude login's older copy for the login set: the stage set no claude login."""
        asked = [what for _t, what in self.marks if what == "./rkmon login claude"]
        return self.result(
            not asked,
            "step 5's walk takes the claude login's older copy, in the older secrets volume, for the login set",
        )

    def start(self) -> bool:
        """`rkmon next`, then the start it gives: STARTED, in standby, and its next line gives doctor."""
        _rc, said = self.rkmon("next")
        rc, out = self.follow(said)
        ok = (
            next_command(next_line(said))[1] == ["start"]
            and rc == 0
            and bool(outcome(out, STARTED_IN_STANDBY))
            and next_command(next_line(out))[0] == 7
        )
        if not ok:
            hostrun.indented(
                f"next: {next_line(said)!r}; start: exit {rc}, outcome {outcome(out, STARTED_IN_STANDBY)!r}, next "
                f"line {next_line(out)!r}"
            )
        return self.result(
            ok,
            "rkmon next gives the start, which starts the monitor on the older install's volumes in standby, and its "
            "next line gives doctor",
        )

    def moved(self) -> bool:
        """The move the start made, checked in the container (moved_findings), and root's line of each copy in the
        container's log."""
        facts, files = self.read(
            lambda f: bool((f.get("record") or {}).get("moved_at"))
        )
        found = moved_findings(facts, files, self.made, self.key)
        self.record = facts.get("record")
        logs = self.must("logs", self.monitor())
        for older, (new, user) in COPIES.items():
            whose = (
                "the write user's only" if user == "rkwrite" else "the monitor user's"
            )
            what = (
                f"the key alone in {older}" if older.endswith("/claude_env") else older
            )
            line = (
                f"the one-time move copied {what} to {new} ({whose}, mode 600); the older copy stays until the move "
                "ends"
            )
            if not says(logs, line):
                found.append(f"the container's log has no line {line!r}")
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "the start made the one-time move once: its report in the start's log, config.json split by owner with no "
            "private value shown, its record, the monitor in standby for the move, reviews.unnamed_are_ours_until at "
            "its time; root's copies of the secrets in their own places, their keepers', mode 600, each said; "
            "config.json and the older copies unchanged",
        )

    def restarted(self) -> None:
        """`rkmon stop --drain` (DeployPath.stop_drained), then `rkmon start` before standby off, checked: STARTED, in
        standby, its next line giving doctor; and, once rk start has ended in the start's log again (starts_ended), the
        move not made again: its record the one the first start wrote, and all else as the move left it, the start's
        log reporting one move (moved_findings)."""
        self.stop_drained()
        rc, out = self.rkmon("start")
        facts, files = self.read(lambda f: starts_ended(f) >= 2)
        found = moved_findings(facts, files, self.made, self.key)
        if starts_ended(facts) < 2:
            found.append(
                f"the start's log shows {starts_ended(facts)} ends of rk start, not 2"
            )
        if facts.get("record") != self.record:
            found.append(
                f"the move's record is {facts.get('record')}, not {self.record}"
            )
        if not (
            rc == 0
            and bool(outcome(out, STARTED_IN_STANDBY))
            and next_command(next_line(out))[0] == 7
        ):
            found.append(
                f"start: exit {rc}, outcome {outcome(out, STARTED_IN_STANDBY)!r}, next line {next_line(out)!r}"
            )
        if found:
            hostrun.indented("\n".join(found))
        self.result(
            not found,
            "stopped, then started again before standby off, the monitor starts in standby and the start does not make "
            "the move again: its record is the one the first start wrote, the start's log reports one move, and "
            "config.json and the copies are as the move left them",
        )

    def released(self) -> None:
        """`rkmon standby off`, checked (END, ended_findings)."""
        rc, out = self.rkmon("standby", "off")
        ok = (
            rc == 0
            and bool(outcome(out, RELEASED))
            and all(says(out, line) for line in END)
            and next_command(next_line(out))[0] == 7
        )
        facts, files = self.read(lambda f: f.get("standby") is False)
        found = ended_findings(facts, files, self.made, self.key)
        if found or not ok:
            missing = [line for line in END if not says(out, line)]
            hostrun.indented(
                "\n".join(
                    [
                        f"standby off: exit {rc}, next line {next_line(out)!r}, lines missing {missing}",
                        *found,
                    ]
                )
            )
        self.result(
            ok and not found,
            "rkmon standby off ends the move: config.json is config.json.migrated, read-only, each older copy is "
            "removed and each new copy stays, each said; the record ended at the time reviews.unnamed_are_ours_until "
            "says, standby is off, and the next line gives doctor",
        )

    def moved_login(self) -> None:
        """Once the move has ended (released), the claude login the move copied, the one the session reads: `rk
        claude-test` in the container (DeployPath.inside), marked as it starts, makes the login's one test call, sent
        as the settings say, and its row passes (login_findings). That the call reached the gateway with the key,
        after() checks (requests_findings)."""
        self.marks.append((time.time(), "rk claude-test"))
        rc, out = self.inside("rk", "claude-test")
        hostrun.indented(f"$ rk claude-test\n{out}")
        found = login_findings(rc, out)
        if found:
            hostrun.indented("\n".join(found))
        self.result(
            not found,
            "the claude login the move copied works once the move has ended: rk claude-test's one test call with it "
            "is answered",
        )


def one_time_move_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = OneTimeMove(engine, image, pkg, names, run_opts, res)
    try:
        t.prepare()
        tested = False  # whether the claude login's test call was made
        if t.path(last=5) == 5 and t.walked() and t.start() and t.moved():
            t.restarted()
            t.released()
            t.moved_login()
            tested = True
        t.after(clone=False, gateway=tested)
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
