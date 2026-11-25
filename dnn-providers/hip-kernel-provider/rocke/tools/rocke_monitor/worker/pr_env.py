# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The environment pull requests' code runs with: an allow-list, the one policy for it. A variable reaches PR code only
when this file names it (NAMES) or it starts with a prefix named here (PREFIXES) and is not shaped like a secret's
(`secret_shaped`: a name that holds TOKEN, SECRET, PASSW or CREDENTIAL, or has KEY, APIKEY, PAT, CREDS or COOKIE as a
word, such as ROCKE_API_KEY); nothing else of the caller's environment does, whatever its name: no login's folder,
token, key or secret, no model's, Claude's or atlassian's settings, none of the monitor's own. FIXED is in it whatever
the caller's says: no Python writes bytecode, so neither a tool's imports (into the package's folder) nor PR code's own
(into its tree) leave any.

On the list: the locale's (LANG, LANGUAGE, LC_*, TZ); the search paths (PATH, PYTHONPATH) and the Python switches the
image and the tools set; which GPUs are visible (the checks hide them: rkmon_lib/nogpu.py) and where ROCm is; rocKE's
own switches (ROCKE_*); and what the package's tools read in a PR job: RKMON_JOB (the job's folder), the folders and
settings the builds, tests and probes take (RKMON_BUILD, RKMON_VENV, ...), what the tests' re-run commands are made of
(RKMON_RERUN_VIA, the package's `rk pr-run`, and RKMON_RESULTS, the monitor's folder the job copies the results to),
the monitor's trees the job's git reads (TREES: the launcher lets git take them, and no other folder, as safe), the
pytest plugins' switches, the probes' and the golden gate's (PROBE_*, GOLDEN_TIMEOUT, PYTHON). Whether the reviewer
model runs on another machine (RKMON_WORKER_REMOTE, never its address or key) is there for tools/worker_grace.py. The
launcher (docker/rk_prcode.py) sets HOME, TMPDIR, RKMON_OUT, RKMON_BUILD, USER and LOGNAME itself, in the job's slot,
and git's own (no fetch, and the job's trees as safe).

- `allowed(env)`: `env` with the listed variables only, and FIXED.
- `data(env)`: `env`'s listed variables for the launcher's stdin, NUL-separated NAME=VALUE: only they ever reach a PR
  runner's process, which the launcher reduces to the list again. `parse(data)`: such data read back, as data;
  `read(stream)`: at most ENV_MAX bytes of it read from a stream and parsed (the launcher's stdin, `rk pr-run --env`).
- `host(env)`: the environment PR code runs with where there is no PR runner (a host): `allowed(env)`, and HOST's as
  they are: HOME, TMPDIR, and RKMON_RAMDISK, where worker/build_native.py builds a PR's tree (in the container a job's
  build folder is its slot's, which is on the RAM disk already, and no job is told the RAM disk's folder).
- `python3 -I pr_env.py --print0`: `data` of this process's environment on stdout: how a job script hands the
  launcher the job's environment (rkmon_lib/prcode.py SHELL, `prc`).
- `python3 -I pr_env.py CMD [ARGS...]`: CMD with `host` of this process's environment: how a job script runs PR code
  where there is no PR runner (a host: rkmon_lib/prcode.py SHELL).

Standard library only: the image's launcher and the worker tools read it too.
"""

import os
import sys

TREES = "RKMON_TREES"  # the monitor's trees the job's git reads, colon-separated
NAMES = frozenset(
    {
        # the locale
        "LANG",
        "LANGUAGE",
        "TZ",
        # the search paths, and Python's switches
        "PATH",
        "PYTHONPATH",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "PYTHONUTF8",
        # the GPUs, and ROCm
        "CUDA_VISIBLE_DEVICES",
        "HIP_PATH",
        "HIP_VISIBLE_DEVICES",
        "ROCM_HOME",
        "ROCM_PATH",
        "ROCR_VISIBLE_DEVICES",
        # the job, and what the package's tools read in it
        "RKMON_JOB",
        "RKMON_JOBS",
        "RKMON_BUILD",
        "RKMON_BUILD_DIR",
        "RKMON_HEAD_PYTHONPATH",
        "RKMON_HEAD_ROOT",
        "RKMON_LLVM_BIN",
        "RKMON_MAX_CONFIGS",
        "RKMON_OUT",
        "RKMON_PYEXTRA",
        "RKMON_REPO",
        "RKMON_REPO_DIR",
        "RKMON_RERUN_VIA",
        "RKMON_RESULTS",
        "RKMON_ROCKE_PATH",
        "RKMON_ROCM_LLVM_BIN",
        TREES,
        "RKMON_VENV",
        "RKMON_WORKER_REMOTE",
        "RKMON_WORKING_DIR",
        # the pytest plugins' switches (tools/pytest_plugins)
        "RKMON_CI_ABSENT",
        "RKMON_HOST_ABSENT",
        "RKMON_NOROCM_HIDE",
        "RKMON_WINSIM",
        "RKMON_WINSIM_ENCODING",
        # the probes' (tools/run_probes.py) and the golden gate's (tools/golden_vs_base.py)
        "GOLDEN_TIMEOUT",
        "PROBE_DETAIL",
        "PROBE_PYTHON",
        "PROBE_TIMEOUT",
        "PYTHON",
    }
)
PREFIXES = ("LC_", "ROCKE_")
# What makes a name a secret's (`secret_shaped`): one of these in it, or one of SECRET_WORDS as a word of it.
SECRET_PARTS = ("TOKEN", "SECRET", "PASSW", "CREDENTIAL")
SECRET_WORDS = frozenset({"KEY", "APIKEY", "PAT", "CREDS", "COOKIE"})
# In PR code's environment whatever the caller's says.
FIXED = {"PYTHONDONTWRITEBYTECODE": "1"}
HOST = ("HOME", "TMPDIR", "RKMON_RAMDISK")  # the caller's, on a host only (`host`)
ENV_MAX = 1 << 20  # the most of an environment's data `read` takes


def secret_shaped(name: str) -> bool:
    """Whether `name` looks like a secret's: it holds a SECRET_PARTS, or one of its words (between underscores) is one
    of SECRET_WORDS, in any case."""
    up = name.upper()
    words = set(up.split("_"))
    return any(p in up for p in SECRET_PARTS) or bool(words & SECRET_WORDS)


def listed(name: str) -> bool:
    return name in NAMES or (name.startswith(PREFIXES) and not secret_shaped(name))


def allowed(env: dict) -> dict:
    """`env` with the listed variables only, and FIXED."""
    return {**{k: v for k, v in env.items() if listed(k)}, **FIXED}


def data(env: dict) -> bytes:
    """`env`'s listed variables, NUL-separated NAME=VALUE."""
    return b"".join(
        f"{k}={v}".encode("utf-8", "surrogateescape") + b"\0"
        for k, v in allowed(env).items()
    )


def parse(data: bytes) -> dict:
    """NUL-separated NAME=VALUE read back: an item without "=", or one that is not UTF-8, is left out."""
    out = {}
    for item in data.split(b"\0"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        try:
            out[k.decode()] = v.decode()
        except UnicodeDecodeError:
            continue
    return out


def read(stream, limit: int = ENV_MAX) -> dict:
    """The environment data on `stream` (NUL-separated NAME=VALUE) read back (`parse`); ValueError when there is more
    than `limit` bytes of it."""
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"an environment over {limit} bytes")
    return parse(data)


def host(env: dict) -> dict:
    """`allowed(env)`, and HOST's variables as `env` has them: PR code's environment on a host, where it runs as the
    session's user."""
    out = allowed(env)
    out.update({k: env[k] for k in HOST if env.get(k)})
    return out


def main(argv: list[str], execvpe=os.execvpe, environ=None, out=None) -> int:
    e = os.environ if environ is None else environ
    if argv == ["--print0"]:
        o = out or sys.stdout.buffer
        o.write(data(e))
        o.flush()
        return 0
    if not argv:
        print("usage: pr_env.py --print0 | CMD [ARGS...]", file=sys.stderr)
        return 2
    env = host(e)
    try:
        execvpe(argv[0], argv, env)
    except OSError as exc:
        print(f"pr_env.py: {argv[0]}: {exc.strerror}", file=sys.stderr)
        return 127
    return 0  # execvpe does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
