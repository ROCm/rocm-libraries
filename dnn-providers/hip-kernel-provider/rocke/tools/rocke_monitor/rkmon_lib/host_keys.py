# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The own keys of the hosts the monitor reaches over ssh: the team dashboard's web host and the GPU login host.
ssh never asks (BatchMode), and trusts a host by the keys the internal state records for it alone
(rkmon_lib/state_store.py), never by its own known_hosts or the system's. `rkmon secret NAME FILE`, which sets the ssh
key for the host, shows the host's own keys as the container reads them, where ssh reaches the host, and records them
once you accept them (docker/rkmon secret, `rk host-key`); `rkmon login`, deploy step 5, does the same for a key that
is set already, as one the one-time move copies (docker/rkmon logins_walk). A host's key is public data, never a
secret.

The Claude session may not run `rk host-key` (rkmon_lib/outward.py). Its guard knows the command by its command line
alone (rkmon_lib/orchestrator.py guard): another spelling of it, or a plain write of the state file, gets past. That
gap is harmless: the host's name comes from the delivered layer, which the session does not write, so a key recorded
that way points ssh at no other host; a key that is not the host's own only makes the team dashboard's push, or the
remote GPU tests, fail host key verification; ssh never sends a private key to the host it reaches, and the web
host's never leaves the writer; and the PR runners read no state.

- `Host(secret, state, address, port, known_hosts, words)`: one host, by the secret that holds the ssh key for it
  (rkmon_lib/secrets_registry.py). `state`: the state key that records the host's keys, as known_hosts lines.
  `address`, `port`: the configuration keys of the host's name and of its ssh port ("" for a host always on port
  22). `known_hosts`: the name of the file in the working folder that ssh reads the host's keys from
  (derived.known_hosts_path). `words`: the host, in words.
- `WEB`, `GPU`, and `HOSTS`, the two. `host(name)`: the Host of the secret NAME; None for another name.
- `target(cfg, h) -> (name, port)`: the host's name ("" for none) and its ssh port, as the configuration names them.
- `recorded(cfg, h) -> list`: the keys the state records for the host the settings name now, as known_hosts lines:
  each line `refusal` takes for its name and port, so a key recorded for another host, or for the host's former name
  or port, is none of its keys, and the file ssh reads, the doctor, rk logins and `read` agree; none while no host is
  named. ssh-keygen read each when it was recorded.
- `record_fix(program="")`: how a key of a host comes to be recorded (rkmon_lib/checks.py): `PROGRAM login`, the
  wrapper's name PROGRAM (doctor.wrapper_name when none is given), shows the host's own key and records it once you
  accept it, once it has set the ssh key when that is not set (rkmon_lib/logins.py).
- `unrecorded(cfg, h) -> str`: why ssh does not reach the host, in plain words with `record_fix`, while no key of it
  is recorded; "" once one is. Each caller that reaches the host over ssh asks it first and runs no ssh while it
  says why: the team dashboard's push and its read of the requests file (rkmon_lib/publish.py), and the remote GPU
  tests (rkmon_lib/gpu.py).
- `known_hosts_file(cfg, h) -> str`: the file ssh reads for the host, holding the recorded keys and nothing else,
  written whole (mode 644, as the keys are public) whenever it holds anything else; os.devnull, which trusts no host,
  when no key is recorded, the working folder is not known or the file cannot be written.
- `ssh_options(cfg, h) -> list`: ssh's options that trust the host by the recorded keys alone (known_hosts_file),
  checked strictly: ssh takes no key it is shown, reads neither its own known_hosts nor the system's, asks no
  command and no DNS record for a key, and shares no connection another ssh opened, which skips the check
  (ControlPath=none), whatever its configuration says.
- `scan(name, port=22, run=subprocess.run) -> (lines, why)`: the host's own keys as known_hosts lines, read from this
  side with ssh-keyscan, and why none were read ("" when some were).
- `fingerprints(lines, run=subprocess.run) -> list`: each line's key type and SHA256 fingerprint (`ssh-keygen -lf -`),
  as you compare them with those the host's admin gives you: one per line, "" for a line ssh-keygen reads no key in;
  [] when ssh-keygen does not run.
- `refusal(lines, name, port=22, run=subprocess.run) -> str`: why LINES are not keys of the host NAME on PORT as
  ssh-keyscan reads them; "" when they are. Each line names the host alone (`[NAME]:PORT` for a port but 22): no
  other name, pattern or negation, for which ssh would trust the key too, or not trust it for the host. Then a key
  type and a key, which ssh-keygen reads (`fingerprints`): no line is recorded that you were not shown.
- `read(cfg, h, run=subprocess.run) -> dict`: what `rk host-key NAME` says of the host: {"host", "port"; "lines" and
  "fingerprints", its own keys as `scan` reads them, each with its fingerprint (a line ssh-keygen reads no key in is
  left out, so the keys shown are those recorded); "recorded", whether the state records each one; "others", how
  many keys it records that were not read; "why", why none was read}.
- `record(home, h, lines)`: LINES recorded as the host's keys in the internal state of the monitor's home HOME
  (state_store.set), in the place of those recorded before. state_store.StateError, and nothing recorded, when the
  state refuses them.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
from typing import NamedTuple

from . import derived


class Host(NamedTuple):
    """One host whose own keys the state records (the module's docstring)."""

    secret: str
    state: str
    address: str
    port: str
    known_hosts: str
    words: str


WEB = Host(
    "web_host_ssh_key",
    "publish.html.host_keys",
    "publish.html.host",
    "",
    derived.WEB_HOST_KNOWN_HOSTS,
    "the web host",
)
GPU = Host(
    "gpu_ssh_key",
    "remote_gpu_tests.login_host_keys",
    "gpu.login_host",
    "gpu.ssh_port",
    derived.GPU_KNOWN_HOSTS,
    "the GPU login host",
)
HOSTS = (WEB, GPU)
SSH_PORT = 22
# ssh-ed25519, ecdsa-sha2-nistp256, ssh-rsa, sk-...
KEY_TYPE = re.compile(r"(?:ssh|ecdsa|sk)-[a-z0-9@.-]+")
KEY = re.compile(r"[A-Za-z0-9+/]+={0,3}")  # the key, in base64


def host(name: str) -> Host | None:
    """The Host of the secret NAME; None for another name."""
    return next((h for h in HOSTS if h.secret == name), None)


def target(cfg: dict, h: Host) -> tuple[str, int]:
    """The host's name and its ssh port, as the configuration names them (the module's docstring)."""
    from . import registry

    name = str(registry.value(cfg, h.address) or "").strip()
    return name, registry.number(cfg, h.port) if h.port else SSH_PORT


def recorded(cfg: dict, h: Host) -> list[str]:
    """The keys the state records for the host the settings name now, as known_hosts lines (the module's
    docstring)."""
    from . import registry

    keys = registry.value(cfg, h.state) or []
    if isinstance(keys, str):
        keys = keys.splitlines()
    if not isinstance(keys, (list, tuple)):
        return []
    name, port = target(cfg, h)
    lines = [str(x).strip() for x in keys]
    return [x for x in lines if name and not _not_its_key(x, name, port)]


def record_fix(program: str = ""):
    """How a key of a host comes to be recorded (the module's docstring)."""
    from . import checks

    if not program:
        from . import doctor

        program = doctor.wrapper_name()
    return checks.Steps(
        (
            checks.Command((program, "login")),
            "shows it, and records it once you accept it",
        )
    )


def unrecorded(cfg: dict, h: Host) -> str:
    """Why ssh does not reach the host while no key of it is recorded (the module's docstring); "" once one is."""
    if recorded(cfg, h):
        return ""
    return f"{h.words}'s own key is not recorded: {record_fix()}"


def known_hosts_file(cfg: dict, h: Host) -> str:
    """The file ssh reads for the host, holding the recorded keys alone (the module's docstring)."""
    from . import paths, whole_file

    lines = recorded(cfg, h)
    if not lines:
        return os.devnull
    try:
        path = derived.known_hosts_path(paths.get(cfg, "working_dir"), h.known_hosts)
    except paths.PathsError:
        return os.devnull
    text = "".join(x + "\n" for x in lines)
    if not _holds(path, text):
        try:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            whole_file.write(path, text, 0o644)
        except OSError:
            return os.devnull
    return path


def _holds(path: str, text: str) -> bool:
    """The file PATH holds TEXT, line ends as they are."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read() == text
    except (OSError, UnicodeDecodeError):
        return False


def ssh_options(cfg: dict, h: Host) -> list[str]:
    """ssh's options that trust the host by the recorded keys alone (the module's docstring)."""
    known = known_hosts_file(cfg, h)
    return [
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        # ssh splits the option's value at white space: a path that holds some is quoted
        "UserKnownHostsFile="
        + (f'"{known}"' if any(c.isspace() for c in known) else known),
        "-o",
        f"GlobalKnownHostsFile={os.devnull}",
        "-o",
        "KnownHostsCommand=none",
        "-o",
        "VerifyHostKeyDNS=no",
        "-o",
        "ControlPath=none",
    ]


def _known_lines(text: str) -> list[str]:
    """The known_hosts lines of TEXT: a host, a key type and a key at least, no comment."""
    return [
        x.strip()
        for x in (text or "").splitlines()
        if x.strip() and not x.startswith("#") and len(x.split()) >= 3
    ]


def scan(name: str, port: int = SSH_PORT, run=subprocess.run) -> tuple[list, str]:
    """The host's own keys, read from this side with ssh-keyscan (the module's docstring)."""
    argv = ["ssh-keyscan", "-T", "10", "-t", "ed25519,ecdsa,rsa"]
    if port != SSH_PORT:
        argv += ["-p", str(port)]
    try:
        p = run([*argv, "--", name], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"[:200]
    lines = _known_lines(p.stdout)
    if lines:
        return lines, ""
    err = [
        x for x in (p.stderr or "").splitlines() if x.strip() and not x.startswith("#")
    ]
    why = err[-1].strip() if err else f"no key came (ssh-keyscan exited {p.returncode})"
    return [], why[:200]


def fingerprints(lines: list, run=subprocess.run) -> list[str]:
    """Each line's key type and SHA256 fingerprint, one per line (the module's docstring). One ssh-keygen per
    line: given several, it skips a line it reads no key in, and what it prints no longer says which line each
    fingerprint is of."""
    out = []
    for line in lines:
        try:
            p = run(
                ["ssh-keygen", "-lf", "-"],
                input=line + "\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        f = ((p.stdout or "").splitlines() or [""])[0].split()
        out.append(
            f"{f[-1].strip('()')} {f[1]}" if p.returncode == 0 and len(f) >= 4 else ""
        )
    return out


def refusal(lines: list, name: str, port: int = SSH_PORT, run=subprocess.run) -> str:
    """Why LINES are not keys of the host NAME on PORT (the module's docstring); "" when they are."""
    if not name:
        return "no host is named"
    if not lines:
        return "no key is given"
    for i, line in enumerate(lines, 1):
        why = _not_its_key(line, name, port)
        if why:
            return f"line {i} {why}"
    prints = fingerprints(lines, run)
    if not prints:
        return "ssh-keygen did not run, so no key was read"
    for i, f in enumerate(prints, 1):
        if not f:
            return f"line {i} has no key ssh-keygen reads"
    return ""


def _not_its_key(line: str, name: str, port: int) -> str:
    """Why the known_hosts LINE is not a key of the host NAME on PORT alone, as ssh-keyscan prints one: a host, a key
    type and a key, the host NAME alone (`[NAME]:PORT` for a port but 22), whatever its case, as ssh matches it; ""
    when it is."""
    want = name if port == SSH_PORT else f"[{name}]:{port}"
    f = line.split()
    if len(f) < 3 or not KEY_TYPE.fullmatch(f[1]) or not KEY.fullmatch(f[2]):
        return "is no host's key (a host, a key type and a key)"
    if f[0].lower() != want.lower():
        return f"is a key of {f[0]}, not of {want} alone"
    return ""


def read(cfg: dict, h: Host, run=subprocess.run) -> dict:
    """What `rk host-key NAME` says of the host (the module's docstring)."""
    name, port = target(cfg, h)
    out = {
        "host": name,
        "port": port,
        "lines": [],
        "fingerprints": [],
        "recorded": False,
        "others": 0,
        "why": "",
    }
    if not name:
        from . import deploy_tree

        out["why"] = (
            f"your settings name no host for it ({deploy_tree.name(h.address)})"
        )
        return out
    scanned, why = scan(name, port, run)
    shown = [(x, f) for x, f in zip(scanned, fingerprints(scanned, run)) if f]
    if scanned and not shown:
        why = "ssh-keygen read none of the keys ssh-keyscan printed"
    lines, have = [x for x, _ in shown], recorded(cfg, h)
    out.update(
        lines=lines,
        fingerprints=[f for _, f in shown],
        recorded=bool(lines) and all(x in have for x in lines),
        others=len([x for x in have if x not in lines]),
        why=why,
    )
    return out


def record(home, h: Host, lines: list) -> None:
    """LINES recorded as the host's keys in the internal state (the module's docstring)."""
    from . import state_store

    state_store.set(home, h.state, list(lines), fix=record_fix())
