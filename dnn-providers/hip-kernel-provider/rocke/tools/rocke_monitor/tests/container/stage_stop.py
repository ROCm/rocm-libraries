# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The stop levels with real processes, as a stage of the image (tests/container/image.py runs it inside the image built
from docker/Dockerfile, as root, with no network), on the deployment the image's stages share (stages.py set_up). The
local worker's model is a fake gateway on a loopback address in the container (tests/fakes/fake_models.py), and each
command runs as the monitor's user, as docker/rkmon runs one in the running container (stages.Deployment.in_container).

  python3 -B stages.py stop   (only as root inside the image: RKMON_CONTAINER=1, RKMON_AS=root)

  stop   review automation live, three drafts in flight against a slow fake gateway, the review driver as the
         container's orchestrator keeps it running (rk review-driver), the admin dashboard's server as the entry point
         starts it (tools/rkweb.py), and an unrelated process of the monitor's user that the monitor did not start. rk
         pause and rk resume (a pause: the jobs run on), rk stop (every monitor process ends; the tick, the driver and
         new work do nothing) and rk start (the admin dashboard again), rk stop --drain (the work in flight finishes
         first; --timeout-minutes 0 stops what is left), rk kill (every monitor process at once, the unrelated one
         untouched), and rk status after each. Then, the monitor stopped, a worker tool and the installed skill
         removed from the work volume: the start's rk provision puts them back and changes nothing else there, nor
         in the internal state but the record's time and package. No secret the deployment set is in any output, log
         or file outside the secrets folders.
"""

import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path[:0] = [str(TESTS.parent), str(TESTS), str(TESTS / "fakes")]

import fake_models  # noqa: E402
import stage_models  # noqa: E402
import stages  # noqa: E402
from rendered import one_line  # noqa: E402
from rkmon_lib import (  # noqa: E402
    agentic,
    container_layout,
    paths,
    runtime_config,
    state_store,
)

# the monitor's state folder (rkmon_lib/store.py): its state and its jobs
STATE = pathlib.Path(container_layout.MONITOR_HOME) / "state"
# what starts the admin dashboard's server, as the entry point runs it (docker/entrypoint.sh: dashboard)
RKWEB = f"{container_layout.IMAGE_CODE}/tools/rkweb.py"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def gone_within(pids, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.2)
    return not any(alive(p) for p in pids)


def job_pids(kind: str = "worker") -> list[int]:
    """The leaders of the monitor's jobs of `kind` that have not ended (no rc file)."""
    out = []
    for meta in (STATE / "jobs").glob(f"{kind}-*/meta.json"):
        if not (meta.parent / "rc").exists():
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8"))["pid"])
            except (OSError, ValueError, KeyError):
                pass
    return [p for p in out if alive(p)]


def state() -> dict:
    """The monitor's state as saved; empty when it cannot be read."""
    try:
        return json.loads((STATE / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status_line(text: str, head: str) -> str:
    return next((x for x in text.splitlines() if x.startswith(head)), "")


def worker_jobs(text: str) -> set:
    """The worker jobs TEXT names (rk status's processes line, rk stop's STOPPED lines), by their ids: a job runs
    several processes, each named with its job."""
    return set(re.findall(r"\bworker-pr\d+-\d+\b", text))


def started(d: stages.Deployment, *argv: str) -> subprocess.Popen:
    """ARGV started as a command runs in the running container (in_container), its output thrown away; it runs on while
    the stage goes on."""
    line, env = d.in_container(*argv)
    return subprocess.Popen(
        line,
        env=env,
        cwd=container_layout.WORK_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def rkweb(d: stages.Deployment, *args: str) -> tuple:
    """tools/rkweb.py ARGS as the entry point runs it, with the image's code (RKMON_CODE), as the monitor's user in the
    running container (in_container): (its exit code, its output), kept in `out`."""
    line, env = d.in_container(RKWEB, *args)
    env["RKMON_CODE"] = container_layout.IMAGE_CODE
    rc, text = stages.run(line, env, cwd=container_layout.WORK_DIR)
    return d.kept(" ".join(["tools/rkweb.py", *args]), rc, text)


def _running(st: stages.Stage, d: stages.Deployment, children: list) -> tuple:
    """The unrelated process, the admin dashboard's server, three drafts and the review driver started (those the stage
    starts itself kept in CHILDREN); rk status shows them: (the unrelated process, the driver).
    """
    # the monitor's user's, but not the monitor's
    unrelated = started(d, "sleep", "900")
    children.append(unrelated)
    _rc, text = rkweb(d)
    st.check(
        "RKWEB started" in text,
        "tools/rkweb.py starts the admin dashboard's server",
        text,
    )
    _rc, tick = d.rk("tick")
    st.check(
        len(stage_models.starts(tick)) == 3,
        "the tick starts three drafts",
        tick[-1500:],
    )
    drv = started(d, "rk", "review-driver")
    children.append(drv)
    time.sleep(3)
    _rc, status = d.rk("status")
    procs = status_line(status, "processes: ")
    st.check(
        status_line(status, "monitor: ") == "monitor: ON; running"
        and "driver" in procs
        and "serve" in procs
        and len(worker_jobs(procs)) == 3,
        "rk status: on and running; the driver, the server and three workers",
        status[:1500],
    )
    return unrelated, drv


def _pause(st: stages.Stage, d: stages.Deployment, drv) -> None:
    """rk pause: the tick does nothing; what runs goes on. rk resume ends it."""
    st.check(d.rk("pause")[0] == 0, "rk pause completes")
    _rc, text = d.rk("tick")
    workers = job_pids()
    st.check(
        "monitor OFF: nothing done" in text and len(workers) == 3 and alive(drv.pid),
        "rk pause: the tick does nothing, and the drafts and the driver run on",
        text[-800:],
    )
    _rc, status = d.rk("status")
    st.check(
        status_line(status, "monitor: ") == "monitor: OFF (paused until rk resume)",
        "rk status says OFF",
        status[:600],
    )
    d.rk("resume")


def _stop(st: stages.Stage, d: stages.Deployment, drv, unrelated) -> None:
    """rk stop: every monitor process ends, and nothing new starts until rk start, which starts the admin dashboard's
    server again."""
    workers = job_pids()
    rc, text = d.rk("stop")
    stopped = "\n".join(ln for ln in text.splitlines() if ln.startswith("STOPPED pid"))
    st.check(
        rc == 0
        and "review driver" in stopped
        and "admin dashboard server" in stopped
        and len(worker_jobs(stopped)) == 3,
        "rk stop names what it stopped: the driver, the server, the three workers",
        text[-2000:],
    )
    st.check(
        gone_within([drv.pid, *workers], 15) and alive(unrelated.pid),
        "after rk stop the monitor's processes are gone; the unrelated one runs on",
    )
    _rc, text = rkweb(d, "status")
    st.check(
        "RKWEB not running" in text, "the admin dashboard's server is stopped", text
    )
    _rc, status = d.rk("status")
    st.check(
        "STOPPED since" in status_line(status, "monitor: ")
        and status_line(status, "processes: ").startswith("processes: 0"),
        "rk status: STOPPED, no monitor process",
        status[:1500],
    )
    _rc, text = d.rk("tick")
    st.check(
        "monitor STOPPED" in text, "while stopped the tick does nothing", text[-500:]
    )
    rc, text = d.rk("review-driver", timeout=120)
    st.check(
        rc == 0 and "DRIVER: the monitor is stopped" in text,
        "while stopped the driver exits at once",
        text[-500:],
    )
    _rc, text = d.rk("review-advance")
    st.check(text.startswith("STOPPED:"), "while stopped no draft starts", text[-500:])
    _rc, text = d.rk("start")
    st.check(
        "STARTED" in text and "RKWEB started" in text,
        "rk start resumes, and starts the admin dashboard again",
        text[-1000:],
    )
    _rc, status = d.rk("status")
    st.check(
        status_line(status, "monitor: ") == "monitor: ON; running"
        and "serve" in status_line(status, "processes: "),
        "rk status: running again, with the server",
        status[:1500],
    )


def _drain(st: stages.Stage, d: stages.Deployment, gw) -> None:
    """rk stop --drain: --timeout-minutes 0 stops what is left; a longer wait lets the drafts finish first."""
    _rc, tick = d.rk("tick")
    st.check(
        len(stage_models.starts(tick)) == 3,
        "after rk start the stopped drafts start over",
        tick[-1500:],
    )
    workers = job_pids()
    _rc, text = d.rk("stop", "--drain", "--timeout-minutes", "0")
    st.check(
        "drain: nothing new starts" in text
        and "DRAIN TIMEOUT after 0 minutes" in text
        and "STOPPED:" in text,
        "rk stop --drain --timeout-minutes 0: the drafts still in flight are stopped",
        text[-1500:],
    )
    st.check(gone_within(workers, 15), "and their workers are gone")
    d.rk("start")
    gw.handler.delay_s = 2
    d.rk("tick")
    _rc, text = d.rk("stop", "--drain", "--timeout-minutes", "6", timeout=900)
    ready = {r.get("pr") for r in state().get("review_ready") or []}
    st.check(
        "drain: nothing new starts" in text
        and all(f"review #{n}" in text for n in stage_models.PRS)
        and "DRAINED: nothing is in flight" in text
        and set(stage_models.PRS) <= ready
        and "STOPPED:" in text
        and text.index("DRAINED") < text.index("STOPPED:"),
        "rk stop --drain: the three drafts finish (each ready for verification), then it stops",
        f"{text[-3000:]}\nready for verification: {sorted(map(str, ready))}",
    )
    _rc, status = d.rk("status")
    st.check(
        "STOPPED since" in status_line(status, "monitor: "),
        "rk status: STOPPED after the drain",
        status[:800],
    )
    d.rk("start")


def _kill(st: stages.Stage, d: stages.Deployment, gw, unrelated, children) -> None:
    """rk kill: every monitor process at once; the unrelated process is never touched. The review driver it starts is
    kept in CHILDREN."""
    gw.handler.delay_s = 25
    for n in stage_models.PRS:  # three drafts again, in flight
        d.rk("review-start", str(n), "--now")
    drv = started(d, "rk", "review-driver")
    children.append(drv)
    time.sleep(3)
    workers = job_pids()
    rc, text = d.rk("kill", "--wait", "2")
    st.check(
        rc == 0
        and "KILLED pid" in text
        and "review driver" in text
        and "Nothing else was touched" in text,
        "rk kill names what it killed",
        text[-2000:],
    )
    st.check(
        gone_within([drv.pid, *workers], 10) and alive(unrelated.pid),
        "after rk kill every monitor process is gone; the unrelated process is untouched",
    )
    _rc, status = d.rk("status")
    st.check(
        "STOPPED since" in status_line(status, "monitor: ")
        and "by rk kill" in status
        and status_line(status, "processes: ").startswith("processes: 0"),
        "rk status: STOPPED by rk kill",
        status[:1500],
    )
    st.check(d.rk("start")[0] == 0, "rk start after a kill")
    # the admin dashboard's server that rk start started ends with the stage
    d.rk("kill", "--wait", "1")
    rkweb(d, "stop")


def volume() -> dict:
    """Each entry of the work volume by its path: a file by its content's digest, a link by where it points, a folder
    as such; and the internal state by its path and each of its keys, the key's value in JSON, all but the record's
    time and package (provisioned.at and provisioned.package_version), which each run of rk provision writes.
    """
    internal = state_store.path(container_layout.MONITOR_HOME)
    body = json.loads(internal.read_text(encoding="utf-8"))
    for key in ("at", "package_version"):
        body.get("provisioned", {}).pop(key, None)
    out = {f"{internal} {k}": json.dumps(v, sort_keys=True) for k, v in body.items()}
    for p in sorted(pathlib.Path(container_layout.WORK_DIR).rglob("*")):
        if p == internal:
            continue
        if p.is_symlink():
            out[str(p)] = f"a link to {os.readlink(p)}"
        elif p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            out[str(p)] = "a folder" if p.is_dir() else "another kind"
    return out


def _repair(st: stages.Stage, d: stages.Deployment) -> None:
    """The monitor stopped, a worker tool and the installed skill removed from the work volume: the start's rk
    provision, in a one-off container as docker/rkmon runs it, puts them back and changes nothing else there, nor in
    the internal state but the record's time and package (volume).
    """
    values = runtime_config.from_settings(d.plan.layer["settings"]).values
    claude = pathlib.Path(values["claude_workspace"]) / ".claude"
    skill = f"skills/{agentic.SKILL}/SKILL.md"
    removed = (
        pathlib.Path(paths.resolve(values)["worker_tools"]) / "mkwt.py",
        claude / skill,
    )
    before = volume()
    for f in removed:
        f.unlink()
    text = one_line(d.one_off("rk", "provision"))
    st.check(
        all(f.is_file() for f in removed)
        and "worker tools: installed in" in text
        and f"agentic files: put back in {claude}: {skill}" in text,
        "with the monitor stopped, the start's rk provision puts back a worker tool and the installed skill removed "
        "from the work volume",
        text[-2000:],
    )
    after = volume()
    changed = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    st.check(
        not changed,
        "and changes nothing else in the work volume, nor in the internal state but the record's time and package",
        ", ".join(changed[:20]),
    )


@stages.image_stage("stop")
def stop(st: stages.Stage, home: pathlib.Path, port: int) -> None:
    """The stop levels with real processes: see the module's docstring."""
    children = []
    with tempfile.TemporaryDirectory() as keys:
        # the gateway's own copy of the key, outside every folder the stage searches for it
        key_file = pathlib.Path(keys) / "gateway_key"
        key_file.write_text(stage_models.GW_KEY + "\n", encoding="utf-8")
        # a gateway at 127.0.0.2 (a remote one, to the monitor), so the three drafts run at once
        gw = fake_models.start(
            home / "fakes",
            "gateway",
            "gateway",
            host="127.0.0.2",
            port=port + 14,
            key_file=str(key_file),
        )
        try:
            over = {
                "features.review_automation.mode": "live",
                "features.review_automation.parallel_reviews": 3,
                "model_bindings.worker": stage_models.gateway(
                    gw, fake_models.GOOD, "header"
                ),
            }
            secrets = dict(
                stages.secret_values(), **{stage_models.KEY: stage_models.GW_KEY}
            )
            with stages.set_up(
                home, port, over=over, secrets=secrets, upstream=stage_models.make_prs
            ) as d:
                # from here on each chat takes 25 s, so the drafts stay in flight; the start's own checks had no wait
                gw.handler.delay_s = 25
                unrelated, drv = _running(st, d, children)
                _pause(st, d, drv)
                _stop(st, d, drv, unrelated)
                _drain(st, d, gw)
                _kill(st, d, gw, unrelated, children)
                _repair(st, d)
                stages.secrets_absent(st, d)
        finally:
            for p in children:
                if p.poll() is None:
                    p.send_signal(signal.SIGKILL)
                p.wait()
            gw.stop()


STAGES = {"stop": stop}
