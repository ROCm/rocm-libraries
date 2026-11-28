# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Remote GPU tests (advanced gpu.enabled, with review automation on): the GPU-only test cases of a review
run on a GPU node of a Slurm cluster reached over SSH. Every site value (the login host and login, the key, the
partition, the accounts, the GPU types, the nodes to avoid) comes from config `gpu`; none is in the package. ssh
trusts the login host's own keys alone, those the state records (remote_gpu_tests.login_host_keys), which
`rkmon secret gpu_ssh_key FILE` shows and records once you accept them; never ssh's own known_hosts.

The operator's rules:
- GPU work only. A job runs only the cases the review's local checks skipped for want of a GPU (their skip reason
  matches gpu.skip_pattern, such as one that names a GPU or a device), on the PR's head merged with the develop the
  checks used. Builds, CPU tests, lint and goldens stay local.
- Queue, never wait inline. Staging and submitting run as a local background job: the login host fetches the PR's head
  and develop from GitHub itself (nothing is pushed from here), checks rocKE's provider folder out at the merge and at
  develop, and submits one batch job with sbatch: gpu.gpus_per_job GPUs of the target's type, under the quality of
  service gpu.qos names (the account's default when it names none). The Slurm job ID is recorded and the review goes
  on.
- Never block a review. `rk post` adds a "GPU tests" row to the review's first table, pending (which cases, which
  target); `rk gpu-row N` updates that row in place once the job ends (the tick's GPU_DONE line says when).
- Before blaming the PR, develop runs on the same node: in the same job, every failing case runs again on develop's
  tree (tools/pytest_vs_base.py). A failure develop passes there is the PR's: GPU_FAIL names a drafted notice for
  `rk notice N FILE --kind gpu`, graded like any finding. A node that fails the GPU probe (no device of the target, or
  a hipcc that cannot compile <hip/hip_runtime.h>) fails nobody's test: the job goes again without it, at most
  gpu.resubmits times, and the node is reported, to be added to
  features.review_automation.remote_gpu_tests.cluster.excluded_nodes (key_name('exclude')).
- Always clean up. A finished job's folder on the login host is removed once its results are copied here
  (<WORKING_DIR>/workers/gpu/<job>/); a job of this monitor that no record names (its name starts with `prefix(cfg)`)
  is cancelled; a job pending longer than gpu.pending_report_min is reported once, never resubmitted on its own
  (`rk gpu-cancel N` gives up on it).
- Account fallback: sbatch tries gpu.accounts in order; one the cluster refuses (an invalid account or association)
  passes to the next.

Modes: jobs start from the review checks, in prototype and live alike; `rk gpu-run` starts one by hand (it writes
nothing to GitHub). The GitHub writes, the row update and the notice, are rk commands the orchestrator runs; while
review automation is in prototype each one only journals what it would do.

Public text never names a node or a GPU product: the row and the notice give the gfx target only (posts name the GPU
architecture, not the node). The node's name stays in the local records
(the job's record, rk gpu-status, the tick lines, the journal), so a re-run can ask for the same machine:
`rk gpu-run N --same-node` (the node of PR N's last job on that target) or `--node NAME` (sbatch --nodelist).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import pathlib
import re
import shlex
import subprocess
import tarfile

from . import (
    deploy,
    deploy_tree,
    derived,
    file_types,
    host_keys,
    jobs,
    paths,
    private_files,
    registry,
    secrets_registry,
    settings_file,
    wrapping,
)
from .store import halted, minutes_since, now

FEATURE = "remote-gpu-tests"
SECRET = "gpu_ssh_key"  # the secret that holds the SSH key for the login host (rkmon_lib/secrets_registry.py)
ROW = "GPU tests"
# The checks' skip lines (tools/pytest_plugins/rkmon_skips.py, pytest_vs_base.py), in the folder the checks copy the
# tests' results to.
SKIPS_FILE = "tests/tests_skipped.tsv"
ACTIVE = ("staging", "queued", "running")
SSH_TIMEOUT_S, FETCH_TIMEOUT_S, STAGE_TIMEOUT_MIN = 60, 180, 30
UNREACHABLE_TICKS = (
    3  # ticks in a row without an answer from the login host before GPU_UNREACHABLE
)
CLEANUP_EVERY_MIN = (
    60  # with no job in flight, how often the queue is read for leftovers
)
KEEP_DAYS = 14  # finished job records kept in the state
GRADES = {"red": "🔴", "orange": "🟠", "yellow": "🟡"}
GFX = re.compile(r"\bgfx[0-9a-f]{2,5}x?\b", re.I)
# The results copied back from a job folder (never the checked-out trees).
RESULT_FILES = (
    "node",
    "probe",
    "result",
    "mode",
    "started",
    "ended",
    "report.txt",
    "slurm.out",
    "cases.txt",
    "out/tests_pr_failed.txt",
    "out/tests_old_failed.txt",
    "out/tests_develop_unrun.txt",
    "out/tests_develop_passed.txt",
    "out/tests_failed.txt",
    "out/tests.log",
    "out/tests_all.log",
)


class GpuError(RuntimeError):
    pass


# --- settings ---------------------------------------------------------------------------------------------------------


def settings(cfg: dict) -> dict:
    """Config `gpu` over its defaults (the registry's), with its lists and maps cleaned; `env`, the variables each job sets, holds
    gpu.base_env's first (the repository's) and gpu.env's over them; `skip_pattern` is text ("" marks no case
    GPU-only)."""
    g = dict(
        registry.defaults("gpu"),
        **{k: v for k, v in (cfg.get("gpu") or {}).items() if v is not None},
    )
    g["accounts"] = [str(a).strip() for a in g.get("accounts") or [] if str(a).strip()]
    g["exclude"] = [str(a).strip() for a in g.get("exclude") or [] if str(a).strip()]
    g["gres"] = {
        str(k).strip().lower(): str(v).strip()
        for k, v in (g.get("gres") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    env = dict(g.get("base_env") or {}, **(g.get("env") or {}))
    g["env"] = {str(k): str(v) for k, v in env.items()}
    g["skip_pattern"] = str(g.get("skip_pattern") or "")
    for key in (
        "max_cases",
        "max_jobs_per_pr",
        "pending_report_min",
        "resubmits",
        "ssh_port",
        "gpus_per_job",
    ):
        try:
            g[key] = int(g.get(key))
        except (TypeError, ValueError):
            g[key] = registry.default(f"gpu.{key}")
    return g


def targets(cfg: dict) -> list[str]:
    """The gfx targets the cluster has a GPU type for (gpu.gres), in their order."""
    return list(settings(cfg)["gres"])


def default_arch(cfg: dict) -> str:
    g = settings(cfg)
    return g["default_arch"].lower() if g["default_arch"] else next(iter(g["gres"]), "")


def key_file(cfg: dict) -> str:
    """The SSH key for the login host: gpu.ssh_key, the place the secrets registry sets for the secret SECRET (the
    load derives it: rkmon_lib/derived.py), "" while no secrets folder is known."""
    key = str(settings(cfg)["ssh_key"] or "")
    return os.path.expanduser(key) if key else ""


def usable_key(cfg: dict) -> str:
    """The key ssh is given: the configured one, or, when others may read it (a read-only mount from a Windows host
    shows every file as 0777, and ssh refuses such a key), a private copy beside it in the secrets folder (its name,
    then .private-copy), written as a private file (private_files.write: permissions 600, never through a link),
    refreshed when the key changes. The configured one when no copy can be made (a key that is not text, a copy
    refused): ssh then says why it refuses the key, and rk doctor checks the key. The key is never printed or logged.
    """
    src = pathlib.Path(key_file(cfg))
    if not src.is_file() or not src.stat().st_mode & 0o077:
        return str(src)
    dest = src.with_name(src.name + ".private-copy")
    try:
        with open(src, encoding="utf-8", newline="") as f:
            text = f.read()
    except UnicodeDecodeError:
        return str(src)
    if dest.is_file() and not dest.is_symlink() and _text(dest) == text:
        private_files.tighten(dest)
        return str(dest)
    try:
        private_files.write(
            dest,
            text,
            paths.PACKAGE,
            replace=True,
            retry=secrets_registry.retry(),
        )
    except private_files.Refused:
        return str(src)
    return str(dest)


def _text(path: pathlib.Path):
    """The file's text, line ends as they are; None when it cannot be read as UTF-8."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def key_name(key: str) -> str:
    """Config gpu's KEY as the rows and errors name it: its key in the deploy settings file (deploy_tree.name), else
    its own name, gpu.KEY."""
    return deploy_tree.name(f"gpu.{key}")


def problems(cfg: dict) -> list[str]:
    """What is wrong with config `gpu`, one line each (the configuration checks; blank values are features.missing's):
    each value a deploy settings file gives, by the form its key there takes (`forms`), and gpu.default_arch, one of
    gpu.gres' targets, which crosses keys (check 17); then the keys no settings file gives: the one-line
    gpu.ssh_key, and the package's gpu.skip_pattern and gpu.max_cases."""
    raw = cfg.get("gpu") or {}
    if not raw:
        return []
    g = settings(cfg)
    out = []

    def bad(key, why):
        out.append(f"{key_name(key)} is {raw.get(key)!r}; {why}")

    misfit = dict(forms(raw))
    for key, why in misfit.items():
        bad(key, why)
    arch = g["default_arch"] if "default_arch" not in misfit else ""
    if arch and arch not in g["gres"]:
        bad("default_arch", f"it must be one of the targets in {key_name('gres')}")
    if any(c in str(g["ssh_key"]) for c in "\n\r\0"):
        bad("ssh_key", "it must be one line")
    try:
        re.compile(g["skip_pattern"])
    except re.error as exc:
        bad("skip_pattern", f"it is not a regular expression ({exc})")
    v = registry.value(cfg, "gpu.max_cases")
    if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
        bad("max_cases", "it must be a whole number, 1 or more")
    return out


def forms(raw: dict) -> list[tuple[str, str]]:
    """(key, why) for each value of config `gpu` (RAW) that the form of its key in a deploy settings file refuses
    (deploy_tree.held), the deploy schema being each form's one home: settings_file.misfits' words, and a
    placeholder (<...>) left in it, as the file's checks refuse it, then the form in the words of the key's
    description. gpu.base_env, the repository's variables, takes gpu.env's form, since each job sets both. A blank
    value is left out, as a file's blank means the key's default."""
    fed = {
        k[len("gpu.") :]: f
        for k, f in deploy_tree.held().items()
        if k.startswith("gpu.")
    }
    fed["base_env"] = fed["env"]
    out = []
    for key, file_key in fed.items():
        value = raw.get(key)
        if settings_file.blank(value):
            continue
        values = {}
        deploy_tree.put(values, file_key, value)
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, values
        )
        said = [r.found for r in settings_file.misfits(doc)]
        said += [
            f"{settings_file.shown_value(t)} has a placeholder, {m.group(0)}"
            for t in _texts(value)
            for m in [settings_file.PLACEHOLDER.search(t)]
            if m
        ]
        if said:
            form = settings_file.closing_of(deploy_tree.description(file_key))[0]
            out.append((key, f"{wrapping.joined(said)}: {form}"))
    return out


def _texts(value) -> list[str]:
    """Each text in VALUE: itself, a list's items, and a mapping's names and values, at any depth."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [t for v in value for t in _texts(v)]
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in (*_texts(str(k)), *_texts(v))]
    return []


def missing(cfg: dict) -> list[str]:
    """What the feature still needs (features._gpu), each a features.Need: the cluster's values (settings), the SSH
    key (the secret SECRET), the login host's own key (host_keys.record_fix), the ssh client (the image's).
    """
    import shutil

    from . import doctor
    from .features import Need

    g = settings(cfg)
    out = [
        Need.setting(f"gpu.{k}", what)
        for k, what in (
            ("login_host", "the cluster's login host"),
            ("ssh_user", "your login there"),
            ("partition", "the Slurm partition"),
            (
                "remote_python",
                "the Python on the GPU nodes that has the tests' packages",
            ),
            ("accounts", "the Slurm accounts, in the order to try"),
            ("gres", "each gfx target and the cluster's GPU type for it"),
        )
        if not g[k]
    ]
    key = key_file(cfg)
    if not key or not os.path.isfile(key):
        out.append(
            Need.secret(
                secrets_registry.entry(cfg, SECRET), "the SSH key for the login host"
            )
        )
    if not host_keys.recorded(cfg, host_keys.GPU):
        out.append(
            Need.fixed_by(
                f"the login host's own key ({host_keys.record_fix()})",
                lambda environ: host_keys.record_fix(doctor.wrapper_name(environ)),
            )
        )
    if not shutil.which(ssh_binary()):
        out.append(Need.image_part("ssh, the OpenSSH client"))
    return out


def configured_with(cfg: dict) -> str:
    g = settings(cfg)
    key = key_file(cfg)
    return (
        f"{g['ssh_user'] or '?'}@{g['login_host'] or '(host not set)'}, partition {g['partition'] or '?'}, "
        f"accounts {', '.join(g['accounts']) or '(none)'}, targets {', '.join(g['gres']) or '(none)'}, key "
        f"{'set' if key and os.path.isfile(key) else 'missing'}"
    )


# --- the ssh side -----------------------------------------------------------------------------------------------------


def ssh_binary() -> str:
    """The ssh client: $RKMON_GPU_SSH (the tests' fake), else ssh."""
    return os.environ.get("RKMON_GPU_SSH") or "ssh"


def ssh_argv(cfg: dict, remote_cmd: str) -> list[str]:
    """ssh to the login host, never interactive: the key only (usable_key); the login host's own keys the state
    records only (host_keys.ssh_options: checked strictly, from no other file); a short connect timeout.
    """
    g = settings(cfg)
    argv = [
        ssh_binary(),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ServerAliveInterval=30",
        *host_keys.ssh_options(cfg, host_keys.GPU),
        "-i",
        usable_key(cfg),
    ]
    if g["ssh_port"] != 22:
        argv += ["-p", str(g["ssh_port"])]
    return argv + ["-l", g["ssh_user"], g["login_host"], remote_cmd]


def remote(
    cfg: dict, script: str, timeout: int = SSH_TIMEOUT_S, binary: bool = False
) -> subprocess.CompletedProcess:
    """Run a bash script on the login host (on stdin). A login host whose own key is not recorded
    (host_keys.unrecorded: ssh does not run), a timeout or a missing ssh comes back as returncode 255 with the reason
    in stderr; nothing here raises for the remote side."""
    why = host_keys.unrecorded(cfg, host_keys.GPU)
    if why:
        p = subprocess.CompletedProcess([], 255, b"", f"ssh: {why}".encode())
    else:
        p = _ssh(cfg, script, timeout)
    if binary:
        return p
    return subprocess.CompletedProcess(
        p.args,
        p.returncode,
        p.stdout.decode("utf-8", "replace"),
        p.stderr.decode("utf-8", "replace"),
    )


def _ssh(cfg: dict, script: str, timeout: int) -> subprocess.CompletedProcess:
    """ssh_argv's bash on the login host, SCRIPT on its stdin, its output as bytes; a timeout or a missing ssh as
    returncode 255 with the reason in stderr."""
    try:
        return subprocess.run(
            ssh_argv(cfg, "bash -s"),
            input=script.encode(),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            [], 255, b"", f"ssh: no answer within {timeout} s".encode()
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            [], 255, b"", b"ssh: the client is not installed"
        )


def _sh(value: str) -> str:
    """A remote path or command word, quoted; a leading ~/ (or a relative path, for remote_dir) is the login's home."""
    v = str(value)
    if v.startswith("~/"):
        return '"$HOME"/' + shlex.quote(v[2:])
    return shlex.quote(v)


def remote_dir(cfg: dict) -> str:
    """The login host's folder for the monitor's clone and job folders, as a bash word."""
    d = str(registry.value(cfg, "gpu.remote_dir", registry.FALSY)).rstrip("/")
    if d.startswith(("/", "~/")):
        return _sh(d)
    return '"$HOME"/' + shlex.quote(d)


def prefix(cfg: dict) -> str:
    """This monitor's Slurm job names start with it: rkmon- and a hash of the working folder, so another monitor of the
    same login (or the login's own jobs) is never cleaned up as ours."""
    wd = paths.resolve(cfg).get("working_dir") or "-"
    return "rkmon-" + hashlib.sha1(wd.encode()).hexdigest()[:6]


NODE_NAME = re.compile(r"[\w.-]{1,64}")  # one node, as sbatch --nodelist takes it


def last_node(st: dict, n: int | None, arch: str) -> str:
    """The node PR n's (develop's, when n is None) newest job on `arch` ran on, from the local records ("" when
    none ran on a known node)."""
    js = [
        j
        for j in state(st)["jobs"]
        if j.get("pr") == n and j.get("arch") == arch and j.get("node")
    ]
    return max(js, key=lambda j: j.get("created") or "")["node"] if js else ""


def local_dir(cfg: dict, job_id: str) -> pathlib.Path:
    """A job's local folder: its staging script and output, then its results."""
    return pathlib.Path(paths.get(cfg, "workers")) / "gpu" / job_id


# --- choosing the cases -----------------------------------------------------------------------------------------------


def read_skips(path: str | pathlib.Path) -> list[tuple[str, str]]:
    """(test id, skip reason) per skip line of a skips file, "rkmon-skip:<TAB>tool<TAB>test id<TAB>reason"
    (tools/pytest_plugins/rkmon_skips.py), each test once."""
    p = pathlib.Path(path)
    if not p.is_file():
        return []
    out, seen = [], set()
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4 and parts[0] == "rkmon-skip:" and parts[2].strip():
            item = (parts[2].strip(), parts[3].strip())
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _arch_for(reason: str, have: list[str], fallback: str) -> tuple[str, str]:
    """(target, named): the target a case runs on (the first gfx target its reason names that the cluster has, a family
    such as gfx94x taking its first member; else the default target), and the first target the reason names.
    """
    named = [m.lower() for m in GFX.findall(reason)]
    for a in named:
        family = (
            a[:-1] if a.endswith("x") else a if len(a) <= 5 else ""
        )  # gfx94x, gfx11 (of gfx11-generic)
        for t in have:
            if t == a or (family and t.startswith(family)):
                return t, a
    return ("", named[0]) if named else (fallback, "")


def select(cfg: dict, skips: list[tuple[str, str]]) -> dict:
    """The GPU-only cases of a skips list, by target: {"by_arch": {target: [ids]}, "no_type": {target: count},
    "over": {target: count}, "files": {target: bool}}. A case runs only when its skip reason matches gpu.skip_pattern;
    one whose reason names only targets the cluster lacks is counted under no_type. Past gpu.max_cases a target's cases
    run as their files; past gpu.max_jobs_per_pr targets the rest are counted under over.
    """
    g = settings(cfg)
    pat = re.compile(g["skip_pattern"] or r"(?!)")  # a blank pattern matches no reason
    have, fallback = list(g["gres"]), default_arch(cfg)
    by_arch: dict[str, list[str]] = {}
    no_type: dict[str, int] = {}
    seen = set()
    for nodeid, reason in skips:
        if nodeid in seen or not pat.search(reason):
            continue
        seen.add(nodeid)
        arch, named = _arch_for(reason, have, fallback)
        if not arch:
            no_type[named or "?"] = no_type.get(named or "?", 0) + 1
            continue
        by_arch.setdefault(arch, []).append(nodeid)
    order = [t for t in have if t in by_arch]
    keep, over, files = {}, {}, {}
    for t in order:
        ids = by_arch[t]
        if len(keep) >= g["max_jobs_per_pr"]:
            over[t] = len(ids)
            continue
        files[t] = len(ids) > g["max_cases"]
        keep[t] = sorted({i.split("::")[0] for i in ids}) if files[t] else ids
    return {"by_arch": keep, "no_type": no_type, "over": over, "files": files}


# --- the state --------------------------------------------------------------------------------------------------------


def state(st: dict) -> dict:
    g = st.setdefault("gpu", {})
    g.setdefault("jobs", [])
    return g


def records(
    st: dict, n: int | None = None, head: str | None = None, active: bool | None = None
) -> list[dict]:
    """The job records (read-only: a state without any keeps none)."""
    out = []
    for j in (st.get("gpu") or {}).get("jobs") or []:
        if n is not None and j.get("pr") != n:
            continue
        if head is not None and j.get("head") != head:
            continue
        if active is not None and (j.get("state") in ACTIVE) != active:
            continue
        out.append(j)
    return out


def _new_id(cfg: dict, st: dict, what: str, arch: str) -> str:
    base = (
        f"{prefix(cfg)}-{what}-{arch}-{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}"
    )
    taken = {j["id"] for j in state(st)["jobs"]}
    out, i = base, 2
    while out in taken:
        out, i = f"{base}-{i}", i + 1
    return out


def _prune(st: dict) -> None:
    keep = []
    for j in state(st)["jobs"]:
        ended = j.get("ended") or j.get("submitted") or now()
        if j.get("state") in ACTIVE or minutes_since(ended) < KEEP_DAYS * 1440:
            keep.append(j)
    state(st)["jobs"] = keep


# --- the scripts ------------------------------------------------------------------------------------------------------


def _heredoc(path_expr: str, text: str, mode: str = "") -> list[str]:
    tag = "RKMON_EOF"
    if any(line == tag for line in text.splitlines()):
        raise GpuError(f"a file to stage has the line {tag}")
    out = [f"cat > {path_expr} <<'{tag}'", text.rstrip("\n"), tag]
    if mode:
        out.append(f"chmod {mode} {path_expr}")
    return out


def job_script(cfg: dict, job: dict) -> str:
    """The batch job (gpu.gpus_per_job GPUs): the node probe, then the cases on the merge's tree with each failure
    run again on develop's (tools/pytest_vs_base.py, run by the node's python), into the job folder. The job folder is
    its one argument, which sbatch passes it (stage_script): it is never written into the script, so a folder with a
    space or any other character a one-line gpu.remote_dir holds reaches it whole.
    """
    g = settings(cfg)
    k = paths.repository_folder(cfg)
    py = str(g["remote_python"])
    pydir = _sh(py.rsplit("/", 1)[0]) if "/" in py else "."
    env = [
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(g["env"].items())
    ]
    arch = job["arch"]
    lines = [
        "#!/bin/bash",
        f"# rkmon remote GPU tests (generated): {job['what']} on {arch}; {g['gpus_per_job']} GPU(s), one command",
        "set -u",
        'J="$1"',
        'cd "$J" || exit 1',
        'NODE=${SLURMD_NODENAME:-$(hostname)}; echo "$NODE" > "$J/node"; date -u +%FT%TZ > "$J/started"',
        # The node's temp folder: TMPDIR, else the system's own (where mktemp makes its files when TMPDIR is unset).
        'export TMPDIR="${TMPDIR:-$(dirname "$(mktemp -u)")}/rkmon-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"',
        # The job's folder there goes when the job ends, fails or is cancelled (a SIGKILL, or an EXIT trap of the
        # node's setup command, leaves it). The trap keeps the folder's path as set here, so a setup command that
        # points TMPDIR elsewhere changes nothing it removes.
        'trap "rm -rf -- $(printf %q "$TMPDIR")" EXIT',
        (g["setup"] or ":"),
        f"export PATH={pydir}:$PATH",
        *env,
        f"ARCH={arch}; K={shlex.quote(k)}",
        # rocKE's own switch (rocke/tests/gpu.py): a case that names this target fails, instead of skipping, when the
        # node has no such GPU, so a job that lost its GPU never passes with no coverage.
        f"export ROCKE_REQUIRE_{arch.upper()}=1",
        # The node probe: the target's GPU is visible, and a hipcc (when the node has one) compiles the HIP header. A
        # node that fails it fails no test: the runner sends the job again without it.
        "P=ok",
        'if command -v rocminfo >/dev/null 2>&1 && ! rocminfo 2>/dev/null | grep -qi "$ARCH"; then '
        'P="no $ARCH device visible (rocminfo)"; fi',
        'if [ "$P" = ok ] && command -v hipcc >/dev/null 2>&1; then '
        "printf '#include <hip/hip_runtime.h>\\n__global__ void rkmon_probe() {}\\n' > \"$TMPDIR/probe.hip\"; "
        'hipcc --offload-arch="$ARCH" --genco -o "$TMPDIR/probe.co" "$TMPDIR/probe.hip" > "$TMPDIR/probe.log" 2>&1 '
        '|| P="hipcc cannot compile <hip/hip_runtime.h>"; fi',
        f'command -v python >/dev/null 2>&1 || P="CONFIG no python on the node ({key_name("remote_python")})"',
        'echo "$P" > "$J/probe"',
        'if [ "$P" != ok ]; then case "$P" in CONFIG*) echo "ERROR ${P#CONFIG }" > "$J/result" ;; '
        '*) echo "NODE $P" > "$J/result" ;; esac; date -u +%FT%TZ > "$J/ended"; exit 0; fi',
        'mapfile -t IDS < "$J/cases.txt"',
        'B="$J/dev/$K"; [ "$(cat "$J/mode")" = develop ] && B=-',
        'python -I -B "$J/tools/pytest_vs_base.py" "$J/out" "$J/pr/$K" "$B" "$J/pr/$K/platform/python:$J/pr/$K/library" '
        '"$J/dev/$K/platform/python:$J/dev/$K/library" "${IDS[@]}" > "$J/report.txt" 2>&1',
        'grep \'^TESTS \' "$J/report.txt" | tail -1 > "$J/result"',
        '[ -s "$J/result" ] || echo "ERROR no result (see report.txt)" > "$J/result"',
        'date -u +%FT%TZ > "$J/ended"',
    ]
    return "\n".join(lines) + "\n"


def stage_script(cfg: dict, job: dict) -> str:
    """For the login host (bash -s): fetch the PR's head and develop from GitHub into the monitor's blobless clone
    there, check rocKE's provider folder out at the merge (or the head alone, when the merge conflicts) and at
    develop, write the job's files, and submit it, trying each account in turn. Prints SUBMITTED <id> <account>,
    ACCOUNT_REFUSED <account> <why>, or STAGE_FAIL / SUBMIT_FAIL <why>. One staging at a time on the login host
    (flock)."""
    g = settings(cfg)
    url = g["clone_url"] or derived.github_address(cfg["repo"])
    since = cfg.get("clone_since") or deploy.default_since()
    sparse = " ".join(shlex.quote(s) for s in paths.sparse_dirs(cfg))
    n, head, dev = job.get("pr"), job["head"], job["develop"]
    refspecs = ([f"+refs/pull/{n}/head:refs/rkmon/pr{n}"] if n else []) + [
        "+refs/heads/develop:refs/rkmon/develop"
    ]
    exclude = sorted(
        (set(g["exclude"]) | set(job.get("excluded") or []))
        - {job.get("nodelist") or ""}
    )
    sb = [
        "sbatch",
        "--parsable",
        f"--job-name={job['id']}",
        f"--partition={shlex.quote(g['partition'])}",
        '--account="$1"',
        f"--gres=gpu:{shlex.quote(job['gres'])}:{g['gpus_per_job']}",
        f"--time={shlex.quote(str(g['time_limit']))}",
        # The job starts in its own folder, as job.sh's first step `cd "$J"` puts it (Slurm itself falls back to
        # the node's temp folder when a --chdir folder is missing there; job.sh then stops at its cd).
        '--chdir="$J"',
        '--output="$J/slurm.out"',
    ]
    if g["qos"]:
        sb.append(f"--qos={shlex.quote(g['qos'])}")
    if g["constraint"]:
        sb.append(f"--constraint={shlex.quote(g['constraint'])}")
    if exclude:
        sb.append(f"--exclude={shlex.quote(','.join(exclude))}")
    if job.get(
        "nodelist"
    ):  # a re-run on the node asked for (rk gpu-run --same-node / --node)
        sb.append(f"--nodelist={shlex.quote(job['nodelist'])}")
    tools = paths.PACKAGE / "tools"
    lines = [
        "set -u",
        "export LC_ALL=C GIT_TERMINAL_PROMPT=0",
        f"D={remote_dir(cfg)}; J=\"$D/jobs/{job['id']}\"; HEAD={head}; DEV={dev}; MODE={job['mode']}",
        'mkdir -p "$D/jobs" || { echo "STAGE_FAIL cannot make the remote folder"; exit 0; }',
        'if command -v flock >/dev/null 2>&1; then exec 9>"$D/.stage.lock"; flock -w 900 9 || '
        '{ echo "STAGE_FAIL the remote clone stayed busy for 15 min"; exit 0; }; fi',
        'if [ ! -d "$D/repo/.git" ]; then rm -rf "$D/repo"; '
        f'git clone -q --filter=blob:none --no-checkout --shallow-since={since} {shlex.quote(url)} "$D/repo" '
        '> "$D/clone.log" 2>&1 || { echo "STAGE_FAIL clone: $(tail -1 "$D/clone.log" | cut -c1-200)"; exit 0; }; '
        'git -C "$D/repo" config extensions.worktreeConfig true; fi',
        f'git -C "$D/repo" fetch -q --filter=blob:none --shallow-since={since} origin {" ".join(refspecs)} '
        '> "$D/fetch.log" 2>&1 || { echo "STAGE_FAIL fetch: $(tail -1 "$D/fetch.log" | cut -c1-200)"; exit 0; }',
        'for C in "$HEAD" "$DEV"; do git -C "$D/repo" cat-file -e "$C^{commit}" 2>/dev/null || '
        'git -C "$D/repo" fetch -q --filter=blob:none origin "$C" >/dev/null 2>&1 || '
        '{ echo "STAGE_FAIL ${C:0:10} is not on GitHub (the PR moved, or develop was rewritten)"; exit 0; }; done',
        'rm -rf "$J"; git -C "$D/repo" worktree prune; mkdir -p "$J/out" "$J/tools/pytest_plugins"',
        f'add() {{ git -C "$D/repo" worktree add -q --no-checkout --detach "$J/$1" "$2" && git -C "$J/$1" '
        f'sparse-checkout set --no-cone {sparse} && git -C "$J/$1" checkout -q --detach "$2"; }}',
        'add pr "$HEAD" > "$J/stage.log" 2>&1 || { echo "STAGE_FAIL checkout of ${HEAD:0:10}: $(tail -1 "$J/stage.log" '
        '| cut -c1-200)"; exit 0; }',
        'if [ "$MODE" = merge ]; then git -C "$J/pr" -c user.name=rkmon -c user.email=rkmon@localhost merge -q --no-ff '
        '--no-edit "$DEV" >> "$J/stage.log" 2>&1 || { git -C "$J/pr" merge --abort >/dev/null 2>&1; MODE=head; }; fi',
        'if [ "$MODE" != develop ]; then add dev "$DEV" >> "$J/stage.log" 2>&1 || { echo "STAGE_FAIL checkout of '
        'develop ${DEV:0:10}"; exit 0; }; fi',
        'echo "$MODE" > "$J/mode"',
        *_heredoc('"$J/cases.txt"', "\n".join(job["cases"]) + "\n"),
        *_heredoc('"$J/job.sh"', job_script(cfg, job), "755"),
        *_heredoc(
            '"$J/tools/pytest_vs_base.py"',
            (tools / "pytest_vs_base.py").read_text(encoding="utf-8"),
        ),
        *_heredoc(
            '"$J/tools/pytest_plugins/rkmon_env.py"',
            (tools / "pytest_plugins" / "rkmon_env.py").read_text(encoding="utf-8"),
        ),
        *_heredoc(
            '"$J/tools/pytest_plugins/rkmon_skips.py"',  # rkmon_env's skip lines, and the report's
            (tools / "pytest_plugins" / "rkmon_skips.py").read_text(encoding="utf-8"),
        ),
        "sub() { " + " ".join(sb) + ' "$J/job.sh" "$J"; }',  # job.sh's one argument
        "for A in " + " ".join(shlex.quote(a) for a in g["accounts"]) + "; do",
        '  if OUT=$(sub "$A" 2>&1); then echo "SUBMITTED ${OUT%%;*} $A"; exit 0; fi',
        '  WHY=$(echo "$OUT" | head -1 | cut -c1-200)',
        '  if echo "$OUT" | grep -qiE \'account|association|not permitted\'; then echo "ACCOUNT_REFUSED $A $WHY"; '
        'else echo "SUBMIT_FAIL $WHY"; exit 0; fi',
        "done",
        'echo "SUBMIT_FAIL every account was refused"',
    ]
    return "\n".join(lines) + "\n"


# --- starting ---------------------------------------------------------------------------------------------------------


def start(
    cfg: dict,
    st: dict,
    store,
    n: int | None,
    head: str,
    develop: str,
    picked: dict,
    origin: str,
    files: dict | None = None,
    nodes: dict | None = None,
) -> list[str]:
    """Queue one job per target of `picked` (select's "by_arch"; `files`: the targets whose cases run as their files;
    `nodes`: the node a target's job must run on, a re-run's): its record, then the staging job in the background.
    Nothing is queued while a stop or a drain holds (store.halted).
    """
    why = halted(st)
    if why:
        return [
            f"GPU_WAIT {'#' + str(n) if n else 'develop'}: {why}; no GPU job is queued"
        ]
    g = settings(cfg)
    out = []
    for arch, cases in picked.items():
        what = f"pr{n}" if n else "dev"
        job = {
            "id": _new_id(cfg, st, what, arch),
            "pr": n,
            "origin": origin,
            "head": head,
            "develop": develop,
            "arch": arch,
            "gres": g["gres"][arch],
            "cases": cases,
            "files": bool((files or {}).get(arch)),
            "mode": "merge" if n else "develop",
            "what": f"#{n} at {head[:10]}" if n else f"develop at {head[:10]}",
            "state": "staging",
            "created": now(),
            "submitted": None,
            "slurm": "",
            "account": "",
            "refused": [],
            "node": "",
            "result": None,
            "row": None,
            "notice": "",
            "pending_told": False,
            "resubmits": 0,
            "excluded": [],
            "nodelist": (nodes or {}).get(arch, ""),
        }
        state(st)["jobs"].append(job)
        out.append(_stage(cfg, store, job))
    return out


def _stage(cfg: dict, store, job: dict) -> str:
    d = local_dir(cfg, job["id"])
    d.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES + (
        "notice.md",
        "stage.out",
        "stage.err",
    ):  # a resend keeps nothing of its last try
        (d / name).unlink(missing_ok=True)
    (d / "stage.sh").write_text(stage_script(cfg, job), encoding="utf-8")
    argv = " ".join(shlex.quote(a) for a in ssh_argv(cfg, "bash -s"))
    body = (
        f'O={shlex.quote(str(d))}\n{argv} < "$O/stage.sh" > "$O/stage.out" 2> "$O/stage.err"\n'
        'echo "ssh rc=$?" >> "$O/stage.err"\n'
    )
    # Each try its own local job: job ids count seconds, and a resend can come within the second of the first try.
    job["stage_job"] = jobs.start(
        store,
        "gpu",
        f"try{job['resubmits']}-{job['id'][-34:]}",
        body,
        timeout_min=STAGE_TIMEOUT_MIN,
    )
    job["state"], job["created"] = "staging", now()
    store.journal(
        "gpu",
        f"{job['what']} on {job['arch']}: staging {len(job['cases'])} case(s) on the login host "
        f"({job['id']})",
    )
    return (
        f"GPU_STAGING {_who(job)} {job['arch']}: {len(job['cases'])} GPU-only case(s) "
        f"{'(as their files) ' if job.get('files') else ''}being queued on a remote GPU node"
        + (f" (on node {job['nodelist']}, as asked)" if job.get("nodelist") else "")
    )


def _who(job: dict) -> str:
    return f"#{job['pr']}" if job.get("pr") else "develop"


def after_checks(cfg: dict, st: dict, store, n: int, head: str) -> list[str]:
    """When a review's checks are done: queue its GPU-only cases (in prototype too: the jobs write nothing to GitHub),
    and say so in checks.md for the verifier. Nothing while remote GPU tests are off or not set up.
    """
    from . import features

    if not features.active(cfg, FEATURE):
        return []
    out_dir = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "checks"
    picked = select(cfg, read_skips(out_dir / SKIPS_FILE))
    dev = (
        (
            (out_dir / "develop_sha").read_text().strip()
            if (out_dir / "develop_sha").exists()
            else ""
        )
        or (st.get("develop") or {}).get("sha")
        or ""
    )
    lines, note = [], []
    queued = False
    if picked["by_arch"] and not dev:
        note.append("not queued: the develop these checks used is not known")
    elif picked["by_arch"] and halted(st):
        note.append(f"not queued: {halted(st)}; `rk gpu-run N` queues them later")
    elif picked["by_arch"]:
        for old in records(st, n, active=True):
            if old["head"] != head:
                lines += cancel_jobs(
                    cfg, st, store, [old], f"superseded by `{head[:8]}`"
                )
        lines += start(
            cfg, st, store, n, head, dev, picked["by_arch"], "review", picked["files"]
        )
        queued = True
    for arch, count in picked["no_type"].items():
        note.append(
            f"{count} case(s) need {arch}, which no GPU type in {key_name('gres')} serves: not run"
        )
    for arch, count in picked["over"].items():
        note.append(
            f"{count} case(s) on {arch} not run: at most {key_name('max_jobs_per_pr')} targets per review"
        )
    total = sum(len(v) for v in picked["by_arch"].values())
    text = (
        "\n## GPU tests (remote)\n\n"
        + (
            f"{total} GPU-only case(s) this machine skipped are queued on a remote GPU node: "
            + ", ".join(
                f"{a} ({len(v)}{' files' if picked['files'].get(a) else ''})"
                for a, v in picked["by_arch"].items()
            )
            + '. `rk post` adds the review\'s "GPU tests" row (pending); it is updated when they finish. '
            "Nothing to verify here.\n"
            if queued
            else (
                "No GPU-only case was skipped here: nothing to run on a GPU node.\n"
                if not total
                else f"{total} GPU-only case(s) this machine skipped.\n"
            )
        )
        + "".join(f"- {x}\n" for x in note)
    )
    try:
        with open(out_dir / "checks.md", "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    return lines


def run_by_hand(
    cfg: dict,
    st: dict,
    store,
    n: int | None,
    arch: str | None,
    cases: list[str],
    develop_only: bool = False,
    same_node: bool = False,
    node: str = "",
) -> list[str]:
    """`rk gpu-run`: PR n's GPU-only cases from its last checks (or the cases named), or develop's named cases. With
    `same_node`, each target's job runs on the node its last job there ran on (the local records name it); with
    `node`, on that node."""
    picked_start = start
    if same_node or node:
        if same_node and node:
            raise GpuError("--same-node or --node NAME, not both")
        if node and not NODE_NAME.fullmatch(node):
            raise GpuError(f"{node!r} is not a node name")

        def picked_start(
            cfg_, st_, store_, n_, head, develop, picked, origin, files=None
        ):
            nodes = {}
            for a in picked:
                want = node or last_node(st_, n_, a)
                if not want:
                    raise GpuError(
                        f"no earlier job of {'#' + str(n_) if n_ else 'develop'} on {a} ran on a known "
                        "node: run it without --same-node, or name one with --node NAME"
                    )
                if want in settings(cfg_)["exclude"]:
                    raise GpuError(
                        f"node {want} is in {key_name('exclude')}: take it out first, or run without the node"
                    )
                nodes[a] = want
            return start(
                cfg_, st_, store_, n_, head, develop, picked, origin, files, nodes
            )

    dev = (st.get("develop") or {}).get("sha") or ""
    if not dev:
        raise GpuError("develop's SHA is not known yet: run rk init (or a tick) first")
    have = targets(cfg)
    if arch and arch.lower() not in have:
        raise GpuError(
            f"{arch} has no GPU type in {key_name('gres')} (targets: {', '.join(have) or 'none'})"
        )
    if develop_only:
        if not cases:
            raise GpuError("--develop needs the cases to run (--case ID, repeatable)")
        return picked_start(
            cfg,
            st,
            store,
            None,
            dev,
            dev,
            {(arch or default_arch(cfg)).lower(): cases},
            "manual",
        )
    rec = (st.get("prs") or {}).get(str(n))
    if not rec or rec.get("status") != "open":
        raise GpuError(f"#{n} is not a tracked open PR")
    if cases:
        return picked_start(
            cfg,
            st,
            store,
            n,
            rec["head"],
            dev,
            {(arch or default_arch(cfg)).lower(): cases},
            "manual",
        )
    out_dir = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "checks"
    checked = (
        (out_dir / "head_sha").read_text().strip()
        if (out_dir / "head_sha").exists()
        else ""
    )
    if checked != rec["head"]:
        raise GpuError(
            f"#{n}'s checks are for {checked[:10] or 'no head'}, not its head {rec['head'][:10]}: run its "
            "review first, or name the cases (--case ID)"
        )
    sel = select(cfg, read_skips(out_dir / SKIPS_FILE))
    picked = {a: v for a, v in sel["by_arch"].items() if not arch or a == arch.lower()}
    if not picked:
        raise GpuError(
            f"#{n}'s checks skipped no GPU-only case" + (f" on {arch}" if arch else "")
        )
    dsha = (
        (out_dir / "develop_sha").read_text().strip()
        if (out_dir / "develop_sha").exists()
        else dev
    )
    return picked_start(
        cfg, st, store, n, rec["head"], dsha or dev, picked, "manual", sel["files"]
    )


# --- each tick --------------------------------------------------------------------------------------------------------


def tick(cfg: dict, st: dict, store) -> list[str]:
    """What the tick does for the GPU jobs: read finished staging jobs, cancel jobs whose PR moved on, poll the queue,
    collect finished jobs, clean up, report long waits. The GitHub side is left to `rk gpu-row` and `rk notice`.
    """
    gs = state(st)
    out = _staged(cfg, st, store)
    out += _superseded(cfg, st, store)
    live = [j for j in gs["jobs"] if j["state"] in ("queued", "running")]
    # With nothing in flight the queue is still read every CLEANUP_EVERY_MIN, so a job of this monitor that no record
    # names (a state restored from a backup, a crash between sbatch and the record) never stays behind.
    if (
        live
        or minutes_since(gs.get("last_poll") or "2000-01-01T00:00:00Z")
        >= CLEANUP_EVERY_MIN
    ):
        out += _poll(cfg, st, store, live)
    _prune(st)
    return out


def _staged(cfg: dict, st: dict, store) -> list[str]:
    out = []
    for j in records(st, active=True):
        if j["state"] != "staging":
            continue
        s = jobs.status(store, j.get("stage_job"))
        if s == "running":
            continue
        if s == "timeout":
            jobs.kill(store, j["stage_job"])
        d = local_dir(cfg, j["id"])
        text = (
            (d / "stage.out").read_text(encoding="utf-8", errors="replace")
            if (d / "stage.out").exists()
            else ""
        )
        err = (
            (d / "stage.err").read_text(encoding="utf-8", errors="replace")
            if (d / "stage.err").exists()
            else ""
        )
        for line in text.splitlines():
            if line.startswith("ACCOUNT_REFUSED "):
                parts = line.split(" ", 2)
                j["refused"].append(
                    [
                        parts[1] if len(parts) > 1 else "?",
                        parts[2] if len(parts) > 2 else "",
                    ]
                )
        sub = next((ln for ln in text.splitlines() if ln.startswith("SUBMITTED ")), "")
        if sub:
            _w, sid, acct = (sub.split() + ["", ""])[:3]
            j.update(state="queued", slurm=sid, account=acct, submitted=now())
            refused = "; ".join(a for a, _why in j["refused"])
            store.journal(
                "gpu",
                f"{j['what']} on {j['arch']}: job {sid} queued (account {acct})"
                + (f"; refused first: {refused}" if refused else ""),
            )
            out.append(
                f"GPU_QUEUED {_who(j)} {j['arch']} at {j['head'][:10]}: {len(j['cases'])} GPU-only case(s), "
                f"job {sid} (account {acct}"
                + (f"; {refused} refused it" if refused else "")
                + ")"
            )
            continue
        errs = [
            ln for ln in err.splitlines() if ln.strip() and not ln.startswith("ssh rc=")
        ]
        why = next(
            (
                ln.split(" ", 1)[1]
                for ln in text.splitlines()
                if ln.startswith(("STAGE_FAIL ", "SUBMIT_FAIL "))
            ),
            "",
        ) or (
            errs[-1]
            if errs
            else f"the staging job ended {s} with no word from the login host"
        )
        j.update(
            state="failed",
            ended=now(),
            result={"status": "error", "detail": f"not queued: {why[:200]}"},
        )
        store.journal("gpu", f"{j['what']} on {j['arch']}: not queued: {why[:200]}")
        out.append(
            f"GPU_NOT_QUEUED {_who(j)} {j['arch']}: {why[:200]} (rk gpu-status; tell the user if it repeats)"
        )
    return out


def _superseded(cfg: dict, st: dict, store) -> list[str]:
    """Jobs whose PR closed or moved to another head: cancelled (a review of the new head queues its own)."""
    out = []
    for j in records(st, active=True):
        if not j.get("pr"):
            continue
        rec = (st.get("prs") or {}).get(str(j["pr"]))
        if not rec or rec.get("status") != "open":
            out += cancel_jobs(cfg, st, store, [j], "the PR is closed")
        elif rec.get("head") and rec["head"] != j["head"]:
            out += cancel_jobs(
                cfg, st, store, [j], f"superseded by `{rec['head'][:8]}`"
            )
    return out


def _poll(cfg: dict, st: dict, store, live: list[dict]) -> list[str]:
    gs = state(st)
    ids = ",".join(j["slurm"] for j in live if j.get("slurm"))
    script = (
        "squeue -h -u \"$USER\" -o '%i|%j|%T|%N|%r' 2>&1; echo \"SQUEUE_RC=$?\"; echo '---'; "
        f"sacct -n -P -X -o JobID,State,ExitCode,NodeList -j {shlex.quote(ids)} 2>/dev/null; true\n"
    )
    p = remote(cfg, script)
    if p.returncode != 0 or "SQUEUE_RC=0" not in p.stdout:
        gs["poll_fails"] = gs.get("poll_fails", 0) + 1
        why = (
            p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}"
        ).splitlines()[-1][:200]
        if gs["poll_fails"] == UNREACHABLE_TICKS:
            return [
                f"GPU_UNREACHABLE: {settings(cfg)['login_host']} has not answered {UNREACHABLE_TICKS} ticks in a "
                f"row ({why}); the GPU jobs wait (tell the user)"
            ]
        return [f"(gpu: the login host did not answer: {why})"]
    gs["poll_fails"], gs["last_poll"] = 0, now()
    head, _, tail = p.stdout.partition("---")
    queue = {}
    for line in head.splitlines():
        parts = line.split("|")
        if len(parts) >= 5 and parts[0].strip():
            queue[parts[0].strip()] = {
                "name": parts[1],
                "state": parts[2],
                "node": parts[3],
                "reason": parts[4],
            }
    acct = {}
    for line in tail.splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            acct[parts[0].strip()] = {
                "state": parts[1],
                "exit": parts[2],
                "node": parts[3],
            }
    out, ended = [], []
    for j in live:
        q = queue.get(j["slurm"])
        if q:
            was = j["state"]
            if q["state"] in (
                "PENDING",
                "CONFIGURING",
                "REQUEUED",
                "RESIZING",
                "SUSPENDED",
            ):
                j["state"], j["reason"] = "queued", q["reason"]
            else:
                j["state"], j["node"] = "running", q["node"] or j.get("node", "")
                j.setdefault("started", now())
                if was != "running":
                    store.journal(
                        "gpu",
                        f"{j['what']} on {j['arch']}: job {j['slurm']} running on {j['node']}",
                    )
                    out.append(
                        f"GPU_RUNNING {_who(j)} {j['arch']}: job {j['slurm']} on node {j['node']}"
                    )
            limit = settings(cfg)["pending_report_min"]
            if (
                j["state"] == "queued"
                and limit
                and not j.get("pending_told")
                and j.get("submitted")
                and minutes_since(j["submitted"]) >= limit
            ):
                j["pending_told"] = True
                hours = minutes_since(j["submitted"]) / 60
                store.journal(
                    "gpu",
                    f"{j['what']} on {j['arch']}: job {j['slurm']} pending {hours:.1f} h "
                    f"({q['reason']})",
                )
                out.append(
                    f"GPU_PENDING_LONG {_who(j)} {j['arch']} job {j['slurm']}: waiting {hours:.1f} h for a node "
                    f"({q['reason'] or 'no reason given'}); reported once, not resubmitted: tell the user "
                    f"(`rk gpu-cancel {j['pr'] or j['id']}` gives up)"
                )
        else:
            j["sacct"] = acct.get(j["slurm"]) or {}
            ended.append(j)
    if ended:
        out += _collect(cfg, st, store, ended)
    out += _leftovers(cfg, st, store, queue)
    return out


def _leftovers(cfg: dict, st: dict, store, queue: dict) -> list[str]:
    """Jobs of this monitor in the queue that no active record names: cancelled (always clean up)."""
    mine = prefix(cfg) + "-"
    known = {j["id"] for j in records(st, active=True)}
    stray = sorted(
        sid
        for sid, q in queue.items()
        if q["name"].startswith(mine) and q["name"] not in known
    )
    if not stray:
        return []
    p = remote(cfg, f"scancel {' '.join(shlex.quote(s) for s in stray)} 2>&1; true\n")
    store.journal(
        "gpu", f"cleanup: cancelled leftover job(s) {', '.join(stray)} ({p.returncode})"
    )
    return [
        f"GPU_CLEANUP: cancelled {len(stray)} leftover job(s) of this monitor: {', '.join(stray)}"
    ]


def _collect(cfg: dict, st: dict, store, ended: list[dict]) -> list[str]:
    """Copy each ended job's results here, then remove its folder (and worktrees) on the login host."""
    names = " ".join(shlex.quote(j["id"]) for j in ended)
    wanted = " ".join(shlex.quote(f) for f in RESULT_FILES)
    script = (
        f'D={remote_dir(cfg)}; cd "$D/jobs" 2>/dev/null || exit 0\n'
        f'L=\'\'; for j in {names}; do for f in {wanted}; do [ -f "$j/$f" ] && L="$L $j/$f"; done; done\n'
        'if [ -n "$L" ]; then tar -czf - $L; fi\n'
    )
    p = remote(cfg, script, FETCH_TIMEOUT_S, binary=True)
    if p.returncode not in (0,):
        why = p.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [
            f"exit {p.returncode}"
        ]
        return [f"(gpu: results of {len(ended)} job(s) not read: {why[0][:200]})"]
    base = pathlib.Path(paths.get(cfg, "workers")) / "gpu"
    if p.stdout:
        _untar(p.stdout, base, {j["id"] for j in ended})
    out = []
    for j in ended:
        out += _result(cfg, st, store, j)
    gone = [j["id"] for j in ended if j["state"] not in ACTIVE]
    if gone:
        names = " ".join(shlex.quote(i) for i in gone)
        remote(
            cfg,
            f'D={remote_dir(cfg)}; for j in {names}; do for w in pr dev; do [ -d "$D/jobs/$j/$w" ] && '
            f'git -C "$D/repo" worktree remove --force "$D/jobs/$j/$w" >/dev/null 2>&1; done; '
            f'rm -rf "$D/jobs/$j"; done; git -C "$D/repo" worktree prune >/dev/null 2>&1; true\n',
        )
    return out


def _untar(data: bytes, base: pathlib.Path, allowed: set) -> None:
    """Extract only regular files under each allowed job folder (no links, no absolute or parent paths)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for m in t.getmembers():
            parts = pathlib.PurePosixPath(m.name).parts
            if (
                not m.isfile()
                or not parts
                or parts[0] not in allowed
                or ".." in parts
                or m.name.startswith("/")
            ):
                continue
            dest = base.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(m).read())


def _lines(path: pathlib.Path) -> list[str]:
    try:
        return [
            x
            for x in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if x.strip()
        ]
    except OSError:
        return []


def classify(d: pathlib.Path, sacct: dict | None = None) -> dict:
    """A finished job's result, from its copied files: status pass, pr-fail (fails here, develop passes or lacks it on
    the same node), develop-fail (develop fails it too), not-compared (develop skips it there), node (the node failed
    the probe), or error."""
    first = (_lines(d / "result") or [""])[0]
    res = {
        "status": "error",
        "detail": "",
        "pr_failed": _lines(d / "out" / "tests_pr_failed.txt"),
        "old_failed": _lines(d / "out" / "tests_old_failed.txt"),
        "unrun": _lines(d / "out" / "tests_develop_unrun.txt"),
        "dev_passed": _lines(d / "out" / "tests_develop_passed.txt"),
        "mode": (_lines(d / "mode") or ["?"])[0],
        "node": (_lines(d / "node") or [""])[0],
    }
    if first.startswith("NODE "):
        res.update(status="node", detail=first[5:])
    elif first.startswith("TESTS pass"):
        res.update(
            status="develop-fail" if res["old_failed"] else "pass",
            detail=first[len("TESTS pass") :].strip(),
        )
    elif first.startswith("TESTS fail"):
        detail = first[len("TESTS fail") :].strip()
        if res["pr_failed"]:
            res.update(
                status="pr-fail" if res["mode"] != "develop" else "fail", detail=detail
            )
        elif res["unrun"]:
            res.update(status="not-compared", detail=detail)
        else:
            res.update(status="error", detail=detail)
    elif first.startswith("ERROR"):
        res.update(detail=first[5:].strip() or "no result")
    else:
        s = (sacct or {}).get("state") or "no record"
        res.update(detail=f"the job ended without a result ({s.split()[0].lower()})")
    return res


def _result(cfg: dict, st: dict, store, j: dict) -> list[str]:
    d = local_dir(cfg, j["id"])
    res = classify(d, j.get("sacct"))
    j["node"] = res["node"] or j.get("node", "")
    g = settings(cfg)
    if res["status"] == "node":
        bad = j["node"]
        st_gpu = state(st)
        st_gpu.setdefault("broken_nodes", {})[bad or "?"] = {
            "why": res["detail"],
            "at": now(),
        }
        store.journal(
            "gpu",
            f"{j['what']} on {j['arch']}: node {bad} failed the probe: {res['detail']}",
        )
        if j.get(
            "nodelist"
        ):  # the node asked for is the broken one: another node is the operator's call
            j.update(
                state="ended",
                ended=now(),
                result=dict(
                    res, status="error", detail=f"no healthy node: {res['detail']}"
                ),
            )
            return [
                f"GPU_NODE_BROKEN {bad} ({j['arch']}): {res['detail']}: {_who(j)}'s job asked for this node, so "
                f"it is not sent again; run it without --same-node, and add the node to {key_name('exclude')}"
            ]
        if j["resubmits"] < g["resubmits"] and bad:
            j.update(
                state="staging",
                resubmits=j["resubmits"] + 1,
                excluded=sorted(set(j["excluded"]) | {bad}),
                slurm="",
                node="",
                submitted=None,
                pending_told=False,
            )
            line = _stage(cfg, store, j)
            return [
                f"GPU_NODE_BROKEN {bad} ({j['arch']}): {res['detail']}: add it to {key_name('exclude')}; "
                f"{_who(j)}'s job goes again without it",
                line,
            ]
        j.update(
            state="ended",
            ended=now(),
            result=dict(
                res, status="error", detail=f"no healthy node: {res['detail']}"
            ),
        )
        return [
            f"GPU_NODE_BROKEN {bad} ({j['arch']}): {res['detail']}: add it to {key_name('exclude')}; "
            f"{_who(j)}'s job is not sent again ({key_name('resubmits')} {g['resubmits']} reached)"
        ]
    j.update(state="ended", ended=now(), result=res)
    store.journal(
        "gpu",
        f"{j['what']} on {j['arch']}: {res['status']} on {j['node']}: {res['detail'][:200]}",
    )
    out = [
        f"GPU_DONE {_who(j)} {j['arch']} at {j['head'][:10]} on node {j['node']}: {res['status']}"
        f"{': ' + res['detail'][:160] if res['detail'] else ''}" + _row_hint(st, j)
    ]
    if res["status"] == "pr-fail" and j.get("pr"):
        path = d / "notice.md"
        path.write_text(notice_text(cfg, j), encoding="utf-8")
        j["notice"] = str(path)
        out.append(
            f"GPU_FAIL #{j['pr']} {j['arch']} at {j['head'][:10]}: {len(res['pr_failed'])} GPU-only case(s) "
            "fail here and pass on develop on the same node: check the drafted notice and grade it like a "
            f"finding (🔴 or 🟠), post it with `rk notice {j['pr']} {path} --kind gpu --key {j['id']}`, then "
            f"`rk gpu-row {j['pr']} --grade red|orange`"
        )
    return out


def _row_hint(st: dict, j: dict) -> str:
    if not j.get("pr"):
        return ""
    if (j.get("row") or {}).get("url"):
        return f"; update the review's row: `rk gpu-row {j['pr']}`"
    return "; `rk post` adds the review's row with this result"


def cancel_jobs(cfg: dict, st: dict, store, js: list[dict], why: str) -> list[str]:
    """scancel jobs (queued or running) and drop staging ones; their rows say not run, and why."""
    out = []
    sids = [
        j["slurm"] for j in js if j.get("slurm") and j["state"] in ("queued", "running")
    ]
    if sids:
        p = remote(
            cfg, f"scancel {' '.join(shlex.quote(s) for s in sids)} 2>&1; true\n"
        )
        if p.returncode != 0:
            out.append(
                f"(gpu: scancel {' '.join(sids)} did not answer: {p.stderr.strip()[-160:]}; the next poll "
                "cleans them up)"
            )
    for j in js:
        if j["state"] == "staging" and j.get("stage_job"):
            jobs.kill(store, j["stage_job"])
        j.update(
            state="cancelled",
            ended=now(),
            result={"status": "cancelled", "detail": why},
        )
        store.journal("gpu", f"{j['what']} on {j['arch']}: cancelled ({why})")
        out.append(
            f"GPU_CANCELLED {_who(j)} {j['arch']}"
            + (f" job {j['slurm']}" if j.get("slurm") else "")
            + f": {why}"
            + _row_hint(st, j).replace("this result", '"not run"')
        )
    return out


def cancel(cfg: dict, st: dict, store, target: str) -> list[str]:
    """`rk gpu-cancel N|JOB|--all`: give up on queued or running jobs."""
    act = records(st, active=True)
    if target == "all":
        js = act
    elif target.isdigit():
        js = [j for j in act if j.get("pr") == int(target) or j.get("slurm") == target]
    else:
        js = [j for j in act if j["id"] == target]
    if not js:
        return [f"no queued or running GPU job matches {target}"]
    return cancel_jobs(cfg, st, store, js, "given up by the operator (rk gpu-cancel)")


# --- the review's row and the notice ----------------------------------------------------------------------------------


def _cases_words(j: dict) -> str:
    files = sorted({c.split("::")[0].rsplit("/", 1)[-1] for c in j["cases"]})
    shown = ", ".join(f"`{f}`" for f in files[:3]) + (
        f" and {len(files) - 3} more" if len(files) > 3 else ""
    )
    return (
        f"{len(j['cases'])} GPU-only {'file' if j.get('files') else 'case'}{'s' if len(j['cases']) != 1 else ''} "
        f"from {shown}"
    )


def _point(j: dict) -> tuple[str, str]:
    """(status, point) of one job for the row."""
    res, a = j.get("result") or {}, j["arch"]
    at = f"`{j['head'][:8]}`"
    where = (
        " (tested at the head: the merge with develop conflicted there)"
        if res.get("mode") == "head"
        else ""
    )
    s = res.get("status")
    if j["state"] in ACTIVE:
        stage = {"staging": "being queued", "queued": "queued", "running": "running"}[
            j["state"]
        ]
        return (
            "pending",
            f"{a}: {_cases_words(j)}: {stage} on a remote GPU node; this row is updated when they finish",
        )
    if s == "pass":
        return "pass", f"{a}: {res.get('detail') or 'passed'} at {at}{where} (ran)"
    if s == "pr-fail":
        ids = ", ".join(f"`{i}`" for i in res["pr_failed"][:2]) + (
            " …" if len(res["pr_failed"]) > 2 else ""
        )
        return "fail", (
            f"{a}: {len(res['pr_failed'])} fail at {at}{where}; develop passes them on the same node "
            f"(ran): {ids}"
        )
    if s == "develop-fail":
        return "other", (
            f"{a}: {len(res['old_failed'])} fail at {at} and on develop on the same node: not the PR's "
            "(ran)"
        )
    if s == "not-compared":
        return (
            "other",
            f"{a}: {len(res['unrun'])} fail at {at}; develop skips them there: not compared",
        )
    if s == "cancelled":
        return "other", f"{a}: not run ({res.get('detail') or 'cancelled'})"
    return "other", f"{a}: not run ({_public_reason(res.get('detail') or '')})"


def _public_reason(detail: str) -> str:
    """Why a job gave no result, in public words: never a node, a path or a host from the local detail (the tick line
    and the journal keep that)."""
    if detail.startswith("not queued"):
        return "it could not be queued; the operator is told"
    if detail.startswith("no healthy node"):
        return "no healthy GPU node"
    m = re.match(r"the job ended without a result \(([a-z_]+)\)", detail)
    if m:
        return f"the job ended without a result: {m.group(1).replace('_', ' ')}"
    return "no result; the operator is told"


def row_cell(st: dict, n: int, head: str, grade: str | None = None) -> str:
    """The "GPU tests" cell for PR n at `head`: a status word, then one point per target ("" when no job ran)."""
    js = records(st, n, head)
    if not js:
        return ""
    pts = [_point(j) for j in js]
    kinds = {k for k, _p in pts}
    if "pending" in kinds:
        word = "⏳ Pending"
    elif "fail" in kinds:
        word = f"{GRADES.get(grade or 'orange', '🟠')} Failed"
    elif kinds == {"pass"}:
        word = "🟢 Passed"
    else:
        word = "🟡 Not conclusive"
    return word + "".join(f"<br>• {p}" for _k, p in pts)


def _row_line(cell: str) -> str:
    return f"| {ROW} | {cell} |"


def with_row(body: str, cell: str) -> str:
    """The review body with its "GPU tests" row set to `cell`: replaced in place, else added after the "Our checks" row,
    else after the first table's last row. Unchanged when cell is blank."""
    if not cell:
        return body
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(f"| {ROW} |"):
            lines[i] = _row_line(cell)
            return "\n".join(lines)
    at = next(
        (i for i, ln in enumerate(lines) if ln.startswith("| Our checks |")), None
    )
    if at is None:
        first = next((i for i, ln in enumerate(lines) if ln.startswith("|")), None)
        if first is None:
            return body
        at = first
        while at + 1 < len(lines) and lines[at + 1].startswith("|"):
            at += 1
    lines.insert(at + 1, _row_line(cell))
    return "\n".join(lines)


def for_post(st: dict, n: int, head: str, body: str) -> str:
    """`rk post`: the body with the GPU tests row for this head (unchanged when no GPU job ran for it)."""
    return with_row(body, row_cell(st, n, head))


def mark_posted(st: dict, n: int, head: str, url: str, body: str) -> None:
    """After `rk post`: each job of this head remembers the comment and the cell it carries."""
    m = re.search(rf"^\| {re.escape(ROW)} \| (.*) \|$", body, re.M)
    for j in records(st, n, head):
        j["row"] = {"url": url, "cell": m.group(1) if m else "", "at": now()}


def rows_due(st: dict, n: int, grade: str | None = None) -> dict[str, str]:
    """{head: the cell it should carry} for each head of PR n whose posted row is out of date."""
    out = {}
    for head in dict.fromkeys(
        j["head"] for j in records(st, n) if (j.get("row") or {}).get("url")
    ):
        cell = row_cell(st, n, head, grade)
        if any(
            (j.get("row") or {}).get("cell") != cell
            for j in records(st, n, head)
            if j.get("row")
        ):
            out[head] = cell
    return out


def update_row(
    cfg: dict, st: dict, store, n: int, grade: str | None = None
) -> list[str]:
    """`rk gpu-row N`: set the "GPU tests" row of each of our posted reviews of PR n to what its jobs say, editing the
    comment in place (guarded, ledgered). While review automation is in prototype (nothing posted) it only journals what
    it would do.
    """
    from . import features, gh, post

    if grade and grade not in GRADES:
        raise GpuError(f"--grade must be one of {', '.join(GRADES)}")
    due = rows_due(st, n, grade)
    if not due:
        return [f"#{n}: no posted GPU tests row is out of date"]
    out = []
    for head, cell in due.items():
        url = next(
            j["row"]["url"]
            for j in records(st, n, head)
            if (j.get("row") or {}).get("url")
        )
        m = re.search(r"issuecomment-(\d+)", url)
        if not m:
            out.append(
                f"#{n} at {head[:10]}: our review's comment id is not known ({url or 'no url'}): row not "
                "updated"
            )
            continue
        cid = m.group(1)
        key = f"{head[:12]}:{hashlib.sha1(cell.encode()).hexdigest()[:10]}"
        if not features.on(cfg, "review-posting"):
            store.journal(
                "gpu",
                f"#{n} PROTOTYPE: would set the GPU tests row of comment {cid} to: {cell[:300]}",
            )
            out.append(
                f"PROTOTYPE: #{n}'s GPU tests row would become: {cell[:200]} (review automation in prototype: "
                "nothing written)"
            )
            continue
        if store.ledger_has("gpu-row", n, key):
            for j in records(st, n, head):
                j["row"]["cell"] = cell
            out.append(f"#{n} at {head[:10]}: the row already says this")
            continue
        try:
            body = gh.rest(f"repos/{cfg['repo']}/issues/comments/{cid}", jq=".body")
        except gh.GhError as exc:
            raise post.PostRefused(f"reading comment {cid} failed: {exc}") from exc
        if not isinstance(body, str) or not body.strip():
            raise post.PostRefused(f"comment {cid} has no body")
        new = with_row(body, cell)
        problems = post.public(new, cfg)
        if problems:
            raise post.PostRefused("; ".join(problems))
        if new == body:
            for j in records(st, n, head):
                j["row"]["cell"] = cell
            out.append(f"#{n} at {head[:10]}: the row already says this")
            continue
        post._rate(cfg, store)
        tmp = local_dir(cfg, f"row-{n}") / f"{cid}.md"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(new, encoding="utf-8")
        p = post._gh_write(
            [
                "api",
                "-X",
                "PATCH",
                f"repos/{cfg['repo']}/issues/comments/{cid}",
                "-F",
                f"body=@{tmp}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if p.returncode:
            raise post.PostRefused(
                f"editing comment {cid} failed: {p.stderr.strip()[:300]}"
            )
        store.ledger_add("gpu-row", n, key, url, note=cid)
        store.journal(
            "post", f"#{n} GPU tests row updated in place ({cid}): {cell[:200]}"
        )
        for j in records(st, n, head):
            j["row"]["cell"] = cell
        out.append(f"updated: #{n}'s GPU tests row at {head[:10]}: {url}")
    return out


def notice_text(cfg: dict, j: dict) -> str:
    """The drafted notice for a failure that is the PR's: the failing cases and how develop did on the same node. The
    orchestrator grades it (🟠 by default) before posting."""
    from . import post

    res = j["result"]
    passed = set(res.get("dev_passed") or [])
    rows = [
        f"| `{i}` | fails | {'passes' if i in passed else 'not in develop (new in this PR)'} |"
        for i in res["pr_failed"][:12]
    ]
    more = (
        f"\n\n…and {len(res['pr_failed']) - 12} more."
        if len(res["pr_failed"]) > 12
        else ""
    )
    where = (
        " (at the head: the merge with develop conflicted)"
        if res.get("mode") == "head"
        else ""
    )
    return (
        f"{post.notice_line(cfg, 'gpu')}\n\n"
        f"**GPU tests on {j['arch']} at `{j['head'][:8]}`{where}: {len(res['pr_failed'])} GPU-only test case(s) "
        f"fail on this PR** — 🟠 should-fix\n\n"
        f"| Test | This PR | develop `{j['develop'][:8]}`, same node |\n|---|---|---|\n"
        + "\n".join(rows)
        + more
        + f"\n\nThese cases skip on machines without a GPU, so they may not run in every CI lane. They ran on one "
        f"{j['arch']} GPU (on a node that passed our GPU probe), and each failure ran again on develop in the same "
        "job.\n\n"
        f"_Automated: the GPU-only cases of our review of `{j['head'][:8]}`, run on a remote GPU node and graded "
        "like any review finding. Reply here if you think the node is at fault._\n"
    )


# --- views ------------------------------------------------------------------------------------------------------------


def status_lines(cfg: dict, st: dict) -> list[str]:
    """`rk gpu-status`: every job record, newest first (local: node names show here, never in posts)."""
    js = sorted(state(st)["jobs"], key=lambda j: j.get("created") or "", reverse=True)
    if not js:
        return ["no GPU jobs recorded"]
    out = []
    for j in js:
        res = j.get("result") or {}
        out.append(
            f"{j['id']}: {j['what']} on {j['arch']} ({j['gres']}): {j['state']}"
            + (f", job {j['slurm']}" if j.get("slurm") else "")
            + (f", account {j['account']}" if j.get("account") else "")
            + (f", node {j['node']}" if j.get("node") else "")
            + (
                f", waiting: {j['reason']}"
                if j.get("reason") and j["state"] == "queued"
                else ""
            )
            + (
                f"; result {res.get('status')}: {res.get('detail', '')[:120]}"
                if res
                else ""
            )
            + (
                f"; refused: {', '.join(a for a, _w in j['refused'])}"
                if j.get("refused")
                else ""
            )
            + (f"; notice draft {j['notice']}" if j.get("notice") else "")
            + (
                f"; files in {local_dir(cfg, j['id'])}"
                if j["state"] not in ACTIVE
                else ""
            )
        )
    broken = state(st).get("broken_nodes") or {}
    if broken:
        out.append(
            f"nodes that failed the probe (add them to {key_name('exclude')}): "
            + "; ".join(
                f"{k} ({v['why']}, {v['at']})" for k, v in sorted(broken.items())
            )
        )
    return out


def queue_lines(cfg: dict) -> list[str]:
    """`rk gpu-queue`: every job of the login in the cluster's queue (read-only), ours marked."""
    p = remote(
        cfg, 'squeue -h -u "$USER" -o \'%i|%j|%T|%N|%r\' 2>&1; echo "SQUEUE_RC=$?"\n'
    )
    if p.returncode != 0 or "SQUEUE_RC=0" not in p.stdout:
        raise GpuError(
            f"the login host did not answer: {(p.stderr or p.stdout).strip()[-200:]}"
        )
    mine = prefix(cfg) + "-"
    rows = [ln.split("|") for ln in p.stdout.splitlines() if "|" in ln]
    if not rows:
        return [f"no job of {settings(cfg)['ssh_user']} in the queue: clean"]
    return [
        f"{r[0]} {r[2]} {r[1]}{' (this monitor)' if r[1].startswith(mine) else ''}"
        + (f" on {r[3]}" if len(r) > 3 and r[3] else "")
        + (f" ({r[4]})" if len(r) > 4 and r[4] else "")
        for r in rows
    ]


def check(cfg: dict) -> list[str]:
    """`rk gpu-check` (and rk doctor's GPU cluster row, reach.gpu_cluster): what the login host has, read-only: the
    partition's nodes by GPU type (with the constraint), the login's accounts, the Slurm tools, git and the remote
    Python with pytest.
    """
    why = host_keys.unrecorded(cfg, host_keys.GPU)
    if why:
        return [f"FAIL {why}"]
    g = settings(cfg)
    py = _sh(g["remote_python"]) if g["remote_python"] else "python3"
    script = (
        f'for t in sbatch squeue sacct scancel sinfo git; do command -v $t >/dev/null 2>&1 && echo "TOOL $t"; '
        "done\n"
        'echo "GIT $(git --version 2>/dev/null)"\n'
        f"echo \"PY $({py} -c 'import sys, pytest; print(sys.version.split()[0], pytest.__version__)' 2>&1 | "
        'tail -1)"\n'
        f"sinfo -h -p {shlex.quote(g['partition'] or 'none')} -N -o '%N|%G|%t|%f' 2>&1 | sed 's/^/NODE /'\n"
        "sacctmgr -n -P show assoc user=\"$USER\" format=account,partition 2>&1 | sed 's/^/ASSOC /'\n"
    )
    p = remote(cfg, script)
    if p.returncode != 0:
        return [
            f"FAIL the login host did not answer: {(p.stderr or p.stdout).strip()[-200:]}"
        ]
    tools = {ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("TOOL ")}
    out = [
        ("PASS" if {"sbatch", "squeue", "scancel", "git"} <= tools else "FAIL")
        + " tools on the login host: "
        + ", ".join(sorted(tools))
    ]
    gv = next((ln[4:] for ln in p.stdout.splitlines() if ln.startswith("GIT ")), "")
    v = tuple(int(x) for x in re.findall(r"\d+", gv)[:2])
    out.append(
        ("PASS" if v >= (2, 25) else "FAIL")
        + f" git there: {gv or 'missing'} (sparse worktrees need 2.25)"
    )
    pv = next((ln[3:] for ln in p.stdout.splitlines() if ln.startswith("PY ")), "")
    out.append(
        ("PASS" if re.match(r"\d+\.\d+", pv) else "FAIL")
        + f" the remote Python with pytest: {pv or 'missing'}"
    )
    nodes = [
        ln[5:].split("|")
        for ln in p.stdout.splitlines()
        if ln.startswith("NODE ") and ln.count("|") >= 3
    ]
    for arch, gres in g["gres"].items():
        fit = [
            n
            for n in nodes
            if f"gpu:{gres}:" in n[1] + ":"
            and (not g["constraint"] or g["constraint"] in n[3])
        ]
        usable = [n for n in fit if n[0] not in g["exclude"]]
        out.append(
            ("PASS" if usable else "FAIL")
            + f" {arch} ({gres}): {len(usable)} node(s) in {g['partition']}"
            + (f" with {g['constraint']}" if g["constraint"] else "")
            + (
                f", {len(fit) - len(usable)} excluded"
                if len(fit) != len(usable)
                else ""
            )
            + (
                f"; idle: {sum(1 for n in usable if n[2].startswith('idle'))}"
                if usable
                else ""
            )
        )
    assoc = {
        ln[6:].split("|")[0]
        for ln in p.stdout.splitlines()
        if ln.startswith("ASSOC ") and "|" in ln
    }
    for a in g["accounts"]:
        out.append(
            ("PASS" if a in assoc else "WARN" if not assoc else "FAIL")
            + f" account {a}: "
            + (
                "yours"
                if a in assoc
                else (
                    "not among your associations"
                    if assoc
                    else "not checked (sacctmgr answered nothing)"
                )
            )
        )
    return out


def board_row(
    cfg: dict, st: dict
) -> tuple[tuple[str, str, str], str, list[str]] | None:
    """The board's "GPU tests" row, its fingerprint part, and what needs the user (a long wait); None when off."""
    from . import features

    if not features.on(cfg, FEATURE):
        return None
    js = state(st)["jobs"]
    q = [j for j in js if j["state"] in ("staging", "queued")]
    r = [j for j in js if j["state"] == "running"]
    day = [
        j
        for j in js
        if j.get("ended") and minutes_since(j["ended"]) < 1440 and j.get("result")
    ]
    by = {}
    for j in day:
        by[j["result"]["status"]] = by.get(j["result"]["status"], 0) + 1
    long = [j for j in q if j.get("pending_told")]
    cur = (
        f"{len(q)} queued, {len(r)} running: "
        + ", ".join(f"{_who(j)} {j['arch']}" for j in (r + q)[:6])
        if q or r
        else "idle"
    )
    nxt = f"last 24 h: {len(day)} finished" + (
        " (" + ", ".join(f"{v} {k}" for k, v in sorted(by.items())) + ")" if by else ""
    )
    if long:
        nxt += "; waiting long: " + ", ".join(f"{_who(j)} {j['arch']}" for j in long)
    needs = [
        f"GPU tests for {_who(j)} on {j['arch']} have waited {minutes_since(j['submitted']) / 60:.0f} h for a "
        f"node (job {j['slurm']}): keep waiting, or give up with `rk gpu-cancel {j['pr'] or j['id']}`?"
        for j in long
    ]
    key = f"gpu {len(q)}/{len(r)} {sorted(by.items())} {[j['id'] for j in long]}"
    return ("GPU tests", cur, nxt), key, needs
