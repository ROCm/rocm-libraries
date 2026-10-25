# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run pytest shards side by side in one allocation, one per GPU, on one
pre-built HSACO cache.

One pre-build process collects the union of the shards' tests and compiles
their backward kernels once (``ROCKE_BWD_PREBUILD``, see
``tests/_attention_bwd_prebuild.py``) into a cache directory; then one pytest
process per shard runs with ``HIP_VISIBLE_DEVICES`` set to its device and
``ROCKE_BWD_HSACO_CACHE`` set to that directory, so the shards only load code
objects. Shard logs go to ``--log-dir``; stdout carries one line per shard
(device, exit code, pytest's summary line) and the pre-build counts.

Usage (from ``library/``)::

    python tests/run_device_shards.py --devices 8 -- \\
        tests/test_attention_bwd_numeric.py -p no:cacheprovider -q
    # four shards of sixteen (4..7) on the one visible GPU, cache kept:
    python tests/run_device_shards.py --devices 1 --shards 4 \\
        --total-shards 16 --first-shard 4 --cache <dir> -- tests/<file>.py -q

Sharding (``--shard-env``):

* ``ROCKE_TEST_SHARD`` (default): the library ``conftest`` keeps the items
  whose shard key (module plus parameter id, so every test of one parameter
  set runs in one shard) hashes to the shard; works for every test file. The
  pre-build collects the job's shards together (``i,j,.../n``).
* ``ROCKE_BWD_SHARD``: the backward numeric file's own case sharding; the
  pre-build then collects every case (use ``--total-shards`` equal to
  ``--shards``).

Devices (``--devices``): a count ``N`` (devices ``0..N-1``), a list ``2,3``,
or ``auto`` (default): the entries of ``HIP_VISIBLE_DEVICES``, else the count
of ``ROCR_VISIBLE_DEVICES`` (``0..k-1``), else one device. Shard ``k`` of the
run gets device ``k mod len(devices)``. The runner itself imports neither torch
nor the HIP runtime.

Cache (``--cache``): default ``$ROCKE_BWD_HSACO_CACHE``, else a new temporary
directory removed at the end. ``--no-prebuild`` skips the pre-build (the shards
still share the cache and fill it lazily).

Processes: one pre-build pytest process (its dry-run and compile pools reuse
each worker for many tests and kernels), then one pytest process per shard
for all of that shard's tests; no process is started per kernel. Under
``PYTHONDONTWRITEBYTECODE`` (a job that keeps its source tree clean) every
one of them would compile all its Python sources again, so the children get a
shared bytecode directory instead (``PYTHONPYCACHEPREFIX``, a temporary
directory outside the source tree, removed at the end): the pre-build writes
it, the pools and shards read it. ``--no-pycache`` keeps the environment as
it is; an existing ``PYTHONPYCACHEPREFIX`` is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

SHARD_ENV = "ROCKE_TEST_SHARD"
BWD_SHARD_ENV = "ROCKE_BWD_SHARD"
CACHE_ENV = "ROCKE_BWD_HSACO_CACHE"
PREBUILD_ENV = "ROCKE_BWD_PREBUILD"
NO_TESTS = 5  # pytest exit code: nothing collected / selected


# --------------------------------------------------------------------------- sharding
def parse_shard(text: str) -> tuple[frozenset[int], int]:
    """``"i/n"`` or ``"i,j,.../n"`` -> ``({i, j, ...}, n)``."""
    picks, sep, total = text.strip().partition("/")
    if not sep:
        raise ValueError(f"shard {text!r} is not i/n")
    n = int(total)
    idx = frozenset(int(p) for p in picks.split(",") if p.strip())
    if n < 1 or not idx or any(not 0 <= i < n for i in idx):
        raise ValueError(f"shard {text!r} out of range")
    return idx, n


def shard_key(item) -> str:
    """Module path plus parameter id (the test function name drops out, so
    the tests of one parameter set share a shard); the node id without
    parameters."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return item.nodeid
    return f"{item.nodeid.split('::', 1)[0]}[{callspec.id}]"


def shard_of(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def select_items(config, items: list) -> None:
    """``pytest_collection_modifyitems`` body: keep this ``ROCKE_TEST_SHARD``."""
    text = os.environ.get(SHARD_ENV, "").strip()
    if not text:
        return
    picks, n = parse_shard(text)
    keep, drop = [], []
    for it in items:
        (keep if shard_of(shard_key(it), n) in picks else drop).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


# --------------------------------------------------------------------------- devices
def resolve_devices(spec: str, env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    spec = spec.strip()
    if spec != "auto":
        if "," in spec:
            return [d.strip() for d in spec.split(",") if d.strip()]
        n = int(spec)
        if n < 1:
            raise ValueError("--devices needs at least one device")
        return [str(i) for i in range(n)]
    hip = env.get("HIP_VISIBLE_DEVICES", "").strip()
    if hip:
        return [d.strip() for d in hip.split(",") if d.strip()]
    rocr = env.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr:
        return [str(i) for i in range(len([d for d in rocr.split(",") if d]))]
    return ["0"]


# --------------------------------------------------------------------------- runs
def _summary(log: Path) -> str:
    """pytest's final summary line of a shard log, after the shard's
    ``bwd-hsaco-cache:`` counters when it printed them (``""`` if none)."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    cache = [ln.strip() for ln in lines if ln.startswith("bwd-hsaco-cache:")]
    head = (cache[-1] + " | ") if cache else ""
    for line in reversed(lines):
        s = line.strip().strip("=").strip()
        if any(
            w in s
            for w in ("passed", "failed", "error", "skipped", "no tests ran", "xfail")
        ):
            return head + s
    return head


def shard_plan(args: argparse.Namespace, devices: Sequence[str]) -> list[tuple]:
    """``[(shard index, device)]`` of this run."""
    return [
        (args.first_shard + k, devices[k % len(devices)]) for k in range(args.shards)
    ]


def build_env(
    base: Mapping[str, str],
    *,
    cache: str,
    shard_env: str,
    shard: str,
    device: str | None,
    prebuild: bool,
    pycache: str | None = None,
) -> dict:
    env = dict(base)
    if pycache is not None:  # shared bytecode instead of none (module doc)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env["PYTHONPYCACHEPREFIX"] = pycache
    env[CACHE_ENV] = cache
    env.pop(SHARD_ENV, None)
    env.pop(BWD_SHARD_ENV, None)
    if shard:
        env[shard_env] = shard
    if device is not None:
        env["HIP_VISIBLE_DEVICES"] = device
    if prebuild:
        env[PREBUILD_ENV] = "1"
    else:
        env.pop(PREBUILD_ENV, None)
    return env


def shared_pycache(env: Mapping[str, str], *, disabled: bool = False) -> str | None:
    """A new temporary bytecode directory for the children when the
    environment forbids writing bytecode and names no prefix (module doc),
    else None (the environment is kept)."""
    if disabled or env.get("PYTHONPYCACHEPREFIX", "").strip():
        return None
    if not env.get("PYTHONDONTWRITEBYTECODE", "").strip():
        return None
    return tempfile.mkdtemp(prefix="rocke-pyc-")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--shards", type=int, default=None, help="default: one per device")
    ap.add_argument("--total-shards", type=int, default=None)
    ap.add_argument("--first-shard", type=int, default=0)
    ap.add_argument(
        "--shard-env", choices=(SHARD_ENV, BWD_SHARD_ENV), default=SHARD_ENV
    )
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-prebuild", action="store_true")
    ap.add_argument(
        "--no-pycache",
        action="store_true",
        help="no shared bytecode directory under PYTHONDONTWRITEBYTECODE",
    )
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    devices = resolve_devices(args.devices)
    args.shards = args.shards or len(devices)
    args.total_shards = args.total_shards or args.shards
    if args.first_shard < 0 or args.first_shard + args.shards > args.total_shards:
        raise SystemExit("shards first..first+shards-1 must lie in 0..total-1")
    plan = shard_plan(args, devices)
    own_cache = not (args.cache or os.environ.get(CACHE_ENV))
    cache = (
        args.cache
        or os.environ.get(CACHE_ENV)
        or tempfile.mkdtemp(prefix="rocke-bwd-hsaco-")
    )
    Path(cache).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or tempfile.mkdtemp(prefix="rocke-shard-logs-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    pycache = shared_pycache(os.environ, disabled=args.no_pycache)
    pytest = [sys.executable, "-m", "pytest"]
    total = args.total_shards
    rc_all = 0
    try:
        if not args.no_prebuild:
            if args.shard_env == SHARD_ENV:
                union = ",".join(str(i) for i, _d in plan) + f"/{total}"
            else:
                union = ""  # the numeric file's sharding takes one shard only
            env = build_env(
                os.environ,
                cache=cache,
                shard_env=args.shard_env,
                shard=union,
                device=plan[0][1],
                prebuild=True,
                pycache=pycache,
            )
            log = log_dir / "prebuild.log"
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.run(
                    pytest + ["--collect-only", "-q"] + pytest_args,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            pre = [ln for ln in lines if ln.startswith("bwd-prebuild:")]
            print(f"prebuild rc={rc} " + (pre[-1] if pre else "no-prebuild-line"))
            if rc not in (0, NO_TESTS):
                print("\n".join(lines[-40:]))
                return rc
        procs = []
        for idx, dev in plan:
            env = build_env(
                os.environ,
                cache=cache,
                shard_env=args.shard_env,
                shard=f"{idx}/{total}",
                device=dev,
                prebuild=False,
                pycache=pycache,
            )
            log = log_dir / f"shard-{idx}-of-{total}.log"
            fh = open(log, "w", encoding="utf-8")
            procs.append(
                (
                    idx,
                    dev,
                    log,
                    fh,
                    subprocess.Popen(
                        pytest + pytest_args,
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
        for idx, dev, log, fh, p in procs:
            rc = p.wait()
            fh.close()
            print(f"shard {idx}/{total} device={dev} rc={rc} {_summary(log)}")
            if rc not in (0, NO_TESTS):
                rc_all = rc_all or rc
                tail = log.read_text(encoding="utf-8", errors="replace")
                print("\n".join(tail.splitlines()[-40:]))
        print(f"logs: {log_dir}")
        return rc_all
    finally:
        if own_cache:
            shutil.rmtree(cache, ignore_errors=True)
        if pycache is not None:
            shutil.rmtree(pycache, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
