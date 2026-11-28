#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A fake Slurm for the remote GPU tests: sbatch, squeue, sacct, scancel, sinfo, sacctmgr and srun (bin/ holds one
wrapper for each), with the cluster and its jobs in $FAKE_SLURM_DIR:

- cluster.json: {"partitions": [names], "accounts": [the login's accounts], "nodes": [{"name", "gres", "state",
  "features", "partition"}]}; every value is invented.
- jobs/<id>.json: one per job (its options, script and the script's arguments, state, node, submit time); next_id.
- sbatch.log: each sbatch's arguments, one call per line.

Nothing runs on its own: a test moves a job on with the driver commands, `slurm.py run ID [--node N] [--path DIR]`
(run its script now with its arguments, on node N, with DIR first on PATH, as a node would; COMPLETED or FAILED by
its exit code),
`slurm.py start ID --node N` (RUNNING) and `slurm.py end ID --state S`."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

DIR = pathlib.Path(os.environ.get("FAKE_SLURM_DIR", "/nonexistent"))
ACTIVE = ("PENDING", "RUNNING")
FLAGS = {
    "h",
    "noheader",
    "n",
    "P",
    "X",
    "N",
    "parsable2",
    "allocations",
    "parsable",
}  # options without a value


def cluster() -> dict:
    return json.loads((DIR / "cluster.json").read_text(encoding="utf-8"))


def job_path(jid: str) -> pathlib.Path:
    return DIR / "jobs" / f"{jid}.json"


def load(jid: str) -> dict | None:
    p = job_path(jid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(job: dict) -> None:
    (DIR / "jobs").mkdir(parents=True, exist_ok=True)
    job_path(job["id"]).write_text(json.dumps(job, indent=1), encoding="utf-8")


def all_jobs() -> list[dict]:
    d = DIR / "jobs"
    return (
        [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(d.glob("*.json"), key=lambda p: int(p.stem))
        ]
        if d.is_dir()
        else []
    )


def options(argv: list[str], short: dict) -> tuple[dict, list[str]]:
    """--name=value, --name value, -x value; the rest positional."""
    opts, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            name, eq, value = a[2:].partition("=")
            if eq:
                opts[name] = value
            elif name in ("parsable", "noheader", "allocations", "help"):
                opts[name] = True
            else:
                opts[name], i = argv[i + 1], i + 1
        elif a.startswith("-") and len(a) == 2:
            name = short.get(a[1], a[1])
            if name in FLAGS:
                opts[name] = True
            else:
                opts[name], i = argv[i + 1], i + 1
        else:
            rest.append(a)
        i += 1
    return opts, rest


def sbatch(argv: list[str]) -> int:
    with open(DIR / "sbatch.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(argv) + "\n")
    opts, rest = options(
        argv,
        {
            "p": "partition",
            "A": "account",
            "J": "job-name",
            "t": "time",
            "o": "output",
            "D": "chdir",
            "C": "constraint",
            "x": "exclude",
            "w": "nodelist",
        },
    )
    c = cluster()
    if opts.get("account") not in c["accounts"]:
        print(
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination "
            "specified",
            file=sys.stderr,
        )
        return 1
    if opts.get("partition") not in c["partitions"]:
        print("sbatch: error: invalid partition specified", file=sys.stderr)
        return 1
    want = (opts.get("gres") or "").split(":")
    if len(want) < 3 or not any(
        n["gres"].startswith(f"gpu:{want[1]}:") for n in c["nodes"]
    ):
        print(
            "sbatch: error: Batch job submission failed: Requested node configuration is not available",
            file=sys.stderr,
        )
        return 1
    nid = int((DIR / "next_id").read_text()) if (DIR / "next_id").exists() else 1001
    (DIR / "next_id").write_text(str(nid + 1))
    save(
        {
            "id": str(nid),
            "name": opts.get("job-name", "sbatch"),
            "account": opts["account"],
            "partition": opts["partition"],
            "gres": opts.get("gres"),
            "constraint": opts.get("constraint", ""),
            "exclude": opts.get("exclude", ""),
            "nodelist": opts.get("nodelist", ""),
            "time": opts.get("time", ""),
            "chdir": opts.get("chdir", tempfile.gettempdir()),
            "output": opts.get("output", ""),
            "script": rest[0] if rest else "",
            "args": rest[1:],  # sbatch passes the words after the script to it
            "user": os.environ.get("USER", "op"),
            "state": "PENDING",
            "node": "",
            "reason": "Resources",
            "submit": time.time(),
            "exit": "0:0",
        }
    )
    print(f"{nid}" if opts.get("parsable") else f"Submitted batch job {nid}")
    return 0


def _fmt(fmt: str, job: dict) -> str:
    values = {
        "i": job["id"],
        "j": job["name"],
        "T": job["state"],
        "N": job["node"],
        "r": job["reason"] if job["state"] == "PENDING" else "None",
        "u": job["user"],
        "P": job["partition"],
        "M": "0:00",
        "V": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(job["submit"])),
    }
    out, i = "", 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            out += str(values.get(fmt[i + 1], ""))
            i += 2
        else:
            out += fmt[i]
            i += 1
    return out


def squeue(argv: list[str]) -> int:
    opts, _ = options(argv, {"u": "user", "o": "format", "j": "jobs", "h": "noheader"})
    ids = set(str(opts.get("jobs", "")).split(",")) - {""}
    fmt = opts.get("format", "%i %j %T")
    for job in all_jobs():
        if (
            job["state"] in ACTIVE
            and (not opts.get("user") or job["user"] == opts["user"])
            and (not ids or job["id"] in ids)
        ):
            print(_fmt(fmt, job))
    return 0


def sacct(argv: list[str]) -> int:
    opts, _ = options(
        argv,
        {
            "j": "jobs",
            "o": "format",
            "n": "noheader",
            "P": "parsable2",
            "X": "allocations",
        },
    )
    fields = str(opts.get("format", "JobID,State")).split(",")
    names = {
        "jobid": "id",
        "jobname": "name",
        "state": "state",
        "exitcode": "exit",
        "nodelist": "node",
    }
    for jid in str(opts.get("jobs", "")).split(","):
        job = load(jid.strip()) if jid.strip() else None
        if job:
            print(
                "|".join(
                    str(job.get(names.get(f.lower(), f.lower()), "")) for f in fields
                )
            )
    return 0


def scancel(argv: list[str]) -> int:
    for jid in argv:
        job = load(jid)
        if job and job["state"] in ACTIVE:
            job["state"] = "CANCELLED"
            save(job)
    return 0


def sinfo(argv: list[str]) -> int:
    opts, _ = options(
        argv, {"p": "partition", "o": "format", "h": "noheader", "N": "N"}
    )
    fmt = opts.get("format", "%N %G %t %f")
    for n in cluster()["nodes"]:
        if (
            opts.get("partition")
            and n.get("partition", opts["partition"]) != opts["partition"]
        ):
            continue
        print(
            fmt.replace("%N", n["name"])
            .replace("%G", n["gres"])
            .replace("%t", n["state"])
            .replace("%f", n.get("features", ""))
        )
    return 0


def sacctmgr(argv: list[str]) -> int:
    c = cluster()
    for a in c["accounts"]:
        print(f"{a}|{c['partitions'][0]}")
    return 0


def srun(argv: list[str]) -> int:
    _opts, rest = options(argv, {"p": "partition", "A": "account", "t": "time"})
    return subprocess.run(rest).returncode if rest else 0


# --- the test's driver ------------------------------------------------------------------------------------------------


def driver(argv: list[str]) -> int:
    cmd, jid = argv[0], argv[1]
    opts, _ = options(argv[2:], {})
    job = load(jid)
    if job is None:
        print(f"no job {jid}", file=sys.stderr)
        return 1
    if cmd == "start":
        job.update(state="RUNNING", node=opts.get("node", "node-a01"))
    elif cmd == "end":
        job.update(state=opts.get("state", "COMPLETED"))
    elif cmd == "run":
        node = opts.get("node", "node-a01")
        job.update(state="RUNNING", node=node)
        save(job)
        env = dict(
            os.environ,
            SLURMD_NODENAME=node,
            SLURM_JOB_ID=jid,
            TMPDIR=tempfile.gettempdir(),
        )
        if opts.get("path"):
            env["PATH"] = f"{opts['path']}:{env.get('PATH', '')}"
        out = job["output"] or os.devnull
        with open(out, "w", encoding="utf-8") as f:
            # Slurm starts the job in the system's temp folder when its --chdir folder is missing on the node.
            cwd = (
                job["chdir"]
                if job["chdir"] and os.path.isdir(job["chdir"])
                else tempfile.gettempdir()
            )
            rc = subprocess.run(
                ["bash", job["script"], *job.get("args", [])],
                env=env,
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode
        job.update(state="COMPLETED" if rc == 0 else "FAILED", exit=f"{rc}:0")
    save(job)
    return 0


COMMANDS = {
    "sbatch": sbatch,
    "squeue": squeue,
    "sacct": sacct,
    "scancel": scancel,
    "sinfo": sinfo,
    "sacctmgr": sacctmgr,
    "srun": srun,
}

if __name__ == "__main__":
    name, args = sys.argv[1], sys.argv[2:]
    sys.exit(COMMANDS[name](args) if name in COMMANDS else driver([name, *args]))
