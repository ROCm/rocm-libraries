# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The canary stage, in the image, as root for its setup only (tests/container/image.py runs it with RKMON_AS=root, in
a throwaway container with no network; tests/container/canary.py runs it on the image its canary build made): a
canary (canary_search.py) in each place a private value lives, then a search for them as each user that must find
none.

  python3 -B stages.py canary

- The setup is the isolation stage's (stage_isolation.py: the logins, the configuration, the clone, the job scripts'
  functions, the start's layout step), and the start's root step. Then a canary of the monitor's in each place one of
  the monitor's private values lives: each secret the monitor keeps (rk secret-set, as the monitor's rk sets it; a
  model binding's key among them), the Claude session's key (the claude login, which rk secret-set sets as `rkmon
  login claude` does), the read login, private values of its configuration, Claude Code's credentials and a
  transcript, a log, a file of its state, and the Claude session's environment (a stand-in process with the session's
  environment, which carries the session's key: rkmon_lib/orchestrator.py session_env); and a canary of the writer's
  in each place one of the
  writer's lives: each secret the writer keeps (rk secret-set as the writer, as `rkmon secret NAME` runs it), the
  GitHub write login and the atlassian login. No canary is ever on a command line: each value goes in on stdin.
- The search works: root, which reads every file, finds each canary in its place, and the session user finds the
  session's in the session's environment.
- A PR runner, as the job scripts run PR code (prslot, pc) from the session's environment, searches the whole
  container (every name, every link, every file changed since the stage began, every process's command line and
  environment) and finds none.
- The writer's volume: the monitor and the session (one user, each with its own environment) list nothing of it and
  read none of its files, by name included, since only file modes keep them out of it; and the session, searching
  the whole container, finds none of the writer's canaries.
"""

import json
import os
import pathlib
import pwd
import shlex
import time

import canary_search as cs
import stage_isolation as iso  # first: it puts the image's code on the path
from rkmon_lib import atlassian, config, gh, orchestrator  # noqa: E402
from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402

SEARCH = f"{iso.PKG}/tests/container/canary_search.py"
ENTRY = f"{iso.PKG}/docker/entrypoint.sh"  # `rkmon secret NAME` runs rk through it, as the secret's keeper
CONFIG = f"{L.MONITOR_HOME}/config.json"
LOG = f"{L.WORK_DIR}/logs/canary.log"
STATE = f"{L.MONITOR_HOME}/state/canary.json"
BINDING = "canary"  # a model binding: its key is a secret of the monitor's
CONFIG_URLS = (
    "jira.url",
)  # private values of the configuration: an address, and hosts
CONFIG_HOSTS = ("gpu.login_host", "publish.html.host")
USERS = {
    sr.MONITOR: L.MONITOR.name,
    sr.WRITER: L.WRITER.name,
}  # each keeper's user; a canary's kind is its keeper
SWEEP_S = 1800  # a search of the whole container


class Proof(iso.Proof):
    """The isolation proof's checks, with the canaries among the secrets: a check whose output holds one fails, and
    its output is shown with each one masked (canary_search.shown)."""

    def run(self, user, cmd, env=None, timeout=600, stdin=None):
        rc, out = super().run(user, cmd, env, timeout, stdin)
        return rc, cs.shown(out)

    def leaked(self, out: str) -> list:
        return super().leaked(out) + (["a canary"] if cs.MASK in out else [])


def value(entry, canary: str) -> str:
    """A value of the entry's shape that holds CANARY: a secret's, or a login's kept in a file (the claude login's
    key: a token)."""
    if entry.shape == sr.HTTPS_URL:
        return f"https://hooks.example.invalid/{canary}"
    if entry.shape == sr.SSH_PRIVATE_KEY:
        # The key's words apart from its markers: the privacy check reads a whole marker as a key.
        key = " OPENSSH PRIVATE KEY-----"
        return f"-----BEGIN{key}\n{canary}\n-----END{key}"
    return canary


def login_text(file: str, canary: str) -> str:
    """A login's file that holds CANARY: gh's hosts file, or a JSON file of a token."""
    if file == gh.HOSTS_FILE:
        return f"github.com:\n    oauth_token: {canary}\n    user: canary-user\n    git_protocol: https\n"
    return json.dumps({"token": canary}) + "\n"


def _set(cfg: dict, dotted: str, v) -> None:
    *parents, last = dotted.split(".")
    for k in parents:
        cfg = cfg.setdefault(k, {})
    cfg[last] = v


def configure(canary: str) -> dict:
    """The configuration with the container's secrets folder (as the load gives it: rkmon_lib/derived.py), a model
    binding whose key is a secret, and CANARY in its private values; written in place, so its owner and mode stay.
    """
    cfg = json.loads(pathlib.Path(CONFIG).read_text(encoding="utf-8"))
    _set(cfg, sr.FOLDER_KEY, L.SECRETS_DIR)
    cfg["model_bindings"] = {
        BINDING: {"api_key_secret": BINDING + sr.BINDING_KEY_SUFFIX}
    }
    for key in CONFIG_URLS:
        _set(cfg, key, f"https://{canary}.example.invalid")
    for key in CONFIG_HOSTS:
        _set(cfg, key, f"{canary}.example.invalid")
    pathlib.Path(CONFIG).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def plant(p: Proof, canaries: dict) -> tuple[dict, dict]:
    """Each canary in its places (the module's docstring), each written as its keeper: a login kept in a folder in its
    files, every other entry through rk secret-set: (the configuration, {path: canary kind}).
    """
    mine = canaries[sr.MONITOR]
    cfg = configure(mine)
    placed = config.normalize(
        cfg
    )  # with the places the load derives, where rk keeps each one
    places = {CONFIG: sr.MONITOR}
    for e in sr.entries(cfg):
        user, canary = USERS[e.keeper], canaries[e.keeper]
        env = {"HOME": pwd.getpwnam(user).pw_dir}
        where = sr.path(placed, e, iso.SESSION_ENV)
        if sr.held_files(e) is not None:
            for file in e.files:
                p.check(
                    "ok",
                    f"{e.name}: its login's {file}, as {user}",
                    user,
                    f"umask 077 && mkdir -p {shlex.quote(where)} && cat > {shlex.quote(f'{where}/{file}')}",
                    env=env,
                    stdin=login_text(file, canary),
                )
                places[f"{where}/{file}"] = e.keeper
            continue
        p.check(
            f"{e.name}: set (",
            f"{e.name}: rk secret-set, as {user}",
            user,
            f"{ENTRY} rk secret-set {e.name}",
            env=env,
            stdin=value(e, canary),
        )
        places[where] = e.keeper
    claude = pathlib.Path(orchestrator.session_env(cfg)["CLAUDE_CONFIG_DIR"])
    for path, text in (
        (claude / ".credentials.json", json.dumps({"key": mine}) + "\n"),
        (
            claude / "projects" / "canary" / "transcript.jsonl",
            json.dumps({"text": mine}) + "\n",
        ),
        (pathlib.Path(LOG), f"a line that names {mine}\n"),
        (pathlib.Path(STATE), json.dumps({"value": mine}) + "\n"),
    ):
        p.check(
            "ok",
            f"{path.name}, as {L.MONITOR.name}",
            L.MONITOR.name,
            f"umask 027 && mkdir -p {shlex.quote(str(path.parent))} && cat > {shlex.quote(str(path))}",
            stdin=text,
        )
        places[str(path)] = sr.MONITOR
    return cfg, places


def session_stand_in(p: Proof, cfg: dict) -> tuple[dict, str]:
    """A process of the session user's with the session's environment (orchestrator.session_env, on the image's), as
    the orchestrator gives it to the Claude session: (that environment, its pid)."""
    env = dict(iso.SESSION_ENV, **orchestrator.session_env(cfg))
    _rc, out = p.run(
        L.MONITOR.name, "sleep 900 >/dev/null 2>&1 & echo $!; disown", env=env
    )
    return env, (out.split() or ["0"])[-1]


def canary_stage(home: pathlib.Path, port: int) -> int:
    if os.geteuid() != 0:
        print("the canary stage sets up as root (image.py runs it with RKMON_AS=root)")
        return 2
    since = int(time.time()) - 1
    search = f"python3 -I {SEARCH} sweep"
    p = Proof()
    iso.setup(p)
    p.check(
        "ok",
        "the start's root step (docker/rk_root.py start), as the entry point's root phase runs it",
        "root",
        iso.ROOT_START,
    )
    print("\n--- the canaries, each in its place, written as its keeper")
    canaries = {keeper: cs.make(keeper) for keeper in USERS}
    cfg, places = plant(p, canaries)
    session, pid = session_stand_in(p, cfg)
    print("\n--- the search works")
    out = p.check(
        "CANARIES: found",
        "root, which reads every file, finds the canaries",
        "root",
        f"{search} --changed-since {since}",
        timeout=SWEEP_S,
    )
    missing = [place for place in places if f" in {place}\n" not in out + "\n"]
    p.expect(
        not missing,
        f"each of the {len(places)} places holds its canary (missing: {', '.join(missing) or 'none'})",
    )
    p.check(
        f"FOUND canary {sr.MONITOR} in /proc/{pid}/environ",
        "the session user finds the session's canary in the session's environment",
        L.MONITOR.name,
        f"{search} --root /proc --kinds {sr.MONITOR}",
    )
    print("\n--- a PR runner")
    p.check(
        "CANARIES: none",
        "a PR runner, as the job scripts run PR code from the session, finds no canary anywhere in the container",
        L.MONITOR.name,
        iso.SLOT7 + f"pc {search} --changed-since {since}",
        need=iso.RAN,
        env=session,
        timeout=SWEEP_S,
    )
    print("\n--- the writer's volume")
    writers = [path for path, keeper in places.items() if keeper == sr.WRITER]
    roots = " ".join(
        f"--root {shlex.quote(r)}" for r in [L.WRITE_SECRETS_DIR, *writers]
    )
    for who, env in (("the monitor", {}), ("the session", session)):
        p.check(
            "CANARIES: none",
            f"{who} lists nothing of the writer's volume and reads none of its files, each named",
            L.MONITOR.name,
            f"{search} {roots}",
            need="SWEPT files=0 ",
            env=env,
        )
    p.check(
        "CANARIES: none",
        "the session finds none of the writer's canaries anywhere in the container",
        L.MONITOR.name,
        f"{search} --kinds {sr.WRITER} --changed-since {since}",
        env=session,
        timeout=SWEEP_S,
    )
    p.run("root", f"kill {pid} $(cat /work/session_pid) 2>/dev/null")
    print(f"RESULT: {p.n - p.failed} PASS, {p.failed} FAIL of {p.n} checks")
    print(f"STAGE canary: {'FAIL' if p.failed else 'PASS'}", flush=True)
    return 1 if p.failed else 0


STAGES = {"canary": canary_stage}
