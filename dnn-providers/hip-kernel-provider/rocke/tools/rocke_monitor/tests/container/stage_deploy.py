# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container deployment's own stage (tests/container/image.py runs it inside the image built from docker/
Dockerfile, as root, with no network): the deployment the image's stages share (stages.py set_up), its admin dashboard
published at a port of its own and every secret set from the host, the claude login too; then `run`, the container's
main process, started as the container starts it, with the dashboard service, the tick and the orchestrator, against
the fake servers and a stand-in for the Claude Agent SDK (tests/fakes/claude_agent_sdk); then `docker stop`'s signal.

  python3 -B stages.py deploy     (only as root inside the image: RKMON_CONTAINER=1, RKMON_AS=root)

Checks: the monitor's secrets, the claude login among them, in /secrets (mode 600); the skill installed in the Claude
workspace on the volume, with the container note and forward-slash paths; `run` serves the admin dashboard, with the
access token, to a browser that names the published port, asks for the token without it and refuses another Host; it
ticks, makes each PR runner's slot folder, and starts the orchestrator, which resumes with the session's options and
records its turn's tokens; rk doctor finds the session ready; the stop signal stops everything (rk stop) and leaves no
process of the run, and rk status says the monitor is stopped; no secret or login the deployment set in any output, log
or file outside the secrets folders.
"""

import base64
import grp
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path[:0] = [str(TESTS.parent), str(TESTS), str(TESTS / "fakes")]

import stages  # noqa: E402
from rkmon_lib import container_layout, secrets_registry  # noqa: E402

# The port the admin dashboard is published on (features.dashboards.admin.port): neither the setting's default nor the
# port the dashboard listens on in the container (container_layout.ADMIN_PORT), so the page shows that the dashboard's
# Host check takes the deploy settings' port.
PUBLIC = 18624
CLAUDE_KEY = "sk-fake-" + "deploy-session-key-909090"
WORK = pathlib.Path(container_layout.WORK_DIR)
LOGS = WORK / "logs"
STATE = pathlib.Path(container_layout.MONITOR_HOME) / "state"
SKILL = WORK / "claude-workspace" / ".claude" / "skills" / "rocke-monitor" / "SKILL.md"
# The monitor's secrets the stage sets, by their files in /secrets: the claude login, the first review pass's key and
# the admin dashboard's access token.
MONITOR_FILES = (
    secrets_registry.entry({}, secrets_registry.CLAUDE_LOGIN).place,
    secrets_registry.FIRST_PASS_KEY,
    stages.ACCESS_TOKEN,
)
STOPPED_LINE = (
    "rkmon-container: stopped"  # the entry point's last line after the stop signal
)
WAIT_SECONDS = (
    150  # for the page, a tick and the session's resume, once `run` has started
)
STOP_SECONDS = 120  # for `run` to end after the stop signal


def get_page(host_port: int, token: str = "") -> tuple[int, str]:
    """The admin dashboard's page as a browser that names `host_port` asks for it: its status and text, or 0 and why
    when nothing answers. With `token`, the request carries it (HTTP Basic, any user name).
    """
    headers = {
        "Host": f"127.0.0.1:{host_port}",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
    }
    if token:
        basic = base64.b64encode(f"op:{token}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"
    url = f"http://127.0.0.1:{container_layout.ADMIN_PORT}/"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except OSError as exc:
        return 0, str(exc)


def slot_folders() -> dict:
    """Each PR runner's slot folder by the runner's name: [its owner, its group, its mode], or why it cannot be read."""
    out = {}
    for u in container_layout.PR_RUNNERS:
        try:
            s = os.stat(f"{container_layout.SLOTS_DIR}/{u.name}", follow_symlinks=False)
            out[u.name] = [s.st_uid, s.st_gid, oct(s.st_mode & 0o7777)]
        except OSError as exc:
            out[u.name] = exc.strerror
    return out


def serving(st, d, runner, sdk_log, output) -> None:
    """What `run` (RUNNER, its output in OUTPUT) does once it has started, waited for up to WAIT_SECONDS: the admin
    dashboard's page, a tick, the PR runners' slots, and the orchestrator's session (its options and its resume, in
    the stand-in's log SDK_LOG; its state)."""
    token = d.secrets[stages.ACCESS_TOKEN]
    deadline, page, ticked, resumed = time.time() + WAIT_SECONDS, (0, ""), False, False
    while time.time() < deadline and runner.poll() is None:
        if page[0] == 200 and ticked and resumed:
            break
        time.sleep(3)
        page = get_page(PUBLIC, token)
        ticked = "TICK" in stages.read(LOGS / "tick.log")
        resumed = '"/rocke-monitor resume"' in stages.read(sdk_log)
    st.check(
        page[0] == 200 and "<html" in page[1].lower(),
        f"the admin dashboard answers a browser that names the published port {PUBLIC}, with the access token",
        page[1][:400] + "\n" + stages.read(output)[-1000:],
    )
    st.check(get_page(PUBLIC)[0] == 401, "without the token, it asks for it (401)")
    st.check(get_page(9999, token)[0] in (0, 403), "and refuses another Host")
    st.check(ticked, "the tick ran", stages.read(LOGS / "tick.log")[-800:])
    group = grp.getgrnam(container_layout.MONITOR.group).gr_gid
    slots = slot_folders()
    st.check(
        all(
            slots[u.name] == [u.uid, group, oct(container_layout.SLOT_MODE)]
            for u in container_layout.PR_RUNNERS
        ),
        "the start made each PR runner's slot folder: the runner's, the monitor's group, mode 2751 "
        "(docker/rk_layout.py)",
        json.dumps(slots)[:800],
    )
    sdk = stages.jsonl(sdk_log)
    opts = next((r["options"] for r in sdk if "options" in r), {})
    st.check(
        resumed
        and opts.get("permission_mode") == "dontAsk"
        and opts.get("setting_sources") == ["project"]
        and "ANTHROPIC_API_KEY" in opts.get("env_names", [])
        and opts.get("hooks") == ["PreToolUse", "SubagentStop"]
        and opts.get("effort") == "high",
        "the orchestrator connected (dontAsk, the project's skill, the key in its environment, the guard and the "
        "usage hooks, effort high) and resumed",
        json.dumps(sdk)[:800] + stages.read(LOGS / "orchestrator.log")[-800:],
    )
    state = json.loads(stages.read(STATE / "orchestrator.json") or "{}")
    usage = stages.jsonl(STATE / "claude_usage.jsonl")
    st.check(
        state.get("session_id") == "fake-session"
        and any(u.get("kind") == "resume" for u in usage),
        "it keeps the session id and records the turn's tokens",
        json.dumps(usage)[:600],
    )


def stopped(runner, output) -> tuple:
    """`docker stop`'s signal to `run` (RUNNER, its output in OUTPUT): (its exit code, its output) once it has ended,
    or once it and its process group are killed, when it has not ended within STOP_SECONDS (its output then says so).
    """
    runner.send_signal(signal.SIGTERM)
    try:
        rc, killed = runner.wait(timeout=STOP_SECONDS), ""
    except subprocess.TimeoutExpired:
        os.killpg(runner.pid, signal.SIGKILL)
        rc = runner.wait(timeout=30)
        killed = (
            f"\n(the stop signal did not end the run within {STOP_SECONDS} s: killed)"
        )
    return rc, stages.read(output) + killed


def group_left(pid: int) -> bool:
    """Whether a process of PID's process group is still there 10 seconds on."""
    for _ in range(20):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.5)
    return True


@stages.image_stage("deploy")
def deploy_stage(st, home: pathlib.Path, port: int) -> None:
    """The deployment of deploy.yaml, its admin dashboard published at PUBLIC and every secret set, the claude login
    too; then `run` and the stop signal (the module's docstring)."""
    secrets = dict(
        stages.secret_values(), **{secrets_registry.CLAUDE_LOGIN: CLAUDE_KEY}
    )
    over = {"features.dashboards.admin.port": PUBLIC}
    with stages.set_up(home, port, over=over, secrets=secrets) as d:
        modes = {
            p.name: oct(p.stat().st_mode & 0o777)
            for p in pathlib.Path(container_layout.SECRETS_DIR).iterdir()
            if p.is_file()
        }
        st.check(
            set(MONITOR_FILES) <= set(modes)
            and all(m == "0o600" for m in modes.values()),
            "the monitor's secrets the host set, the claude login among them, are in /secrets, mode 600",
            json.dumps(modes),
        )
        skill = stages.read(SKILL)
        st.check(
            bool(skill), "the skill is installed in the Claude workspace on the volume"
        )
        st.check(
            "**In the container.**" in skill and "\\.claude" not in skill,
            "the installed skill carries the container note and forward-slash paths",
        )
        sdk_log, output = home / "sdk.log", home / "run.log"
        sdk_log.touch()  # the stand-in writes it as the monitor's user
        os.chown(sdk_log, container_layout.MONITOR.uid, container_layout.MONITOR.gid)
        sdk_log.chmod(0o660)
        env = d.environment(PYTHONPATH=str(TESTS / "fakes"), FAKE_SDK_LOG=str(sdk_log))
        with open(output, "w", encoding="utf-8") as out:
            runner = subprocess.Popen(
                ["sh", str(stages.ENTRY), "run"],
                env=env,
                cwd=container_layout.WORK_DIR,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            serving(st, d, runner, sdk_log, output)
            status, row = stages.report_row(
                d.rk("doctor")[1], "Claude session (rk orchestrate)"
            )
            st.check(
                status == "PASS",
                "rk doctor finds the session ready",
                row or d.out["rk doctor"][1][-1200:],
            )
            _rc, text = d.kept("run", *stopped(runner, output))
            st.check(
                any(x.endswith(STOPPED_LINE) for x in text.splitlines()),
                "docker stop's signal stops everything (rk stop)",
                text[-1200:],
            )
            ps = ["ps", "-eo", "pid,pgid,args"]
            st.check(
                not group_left(runner.pid),
                "nothing of the run is left after it (no process of its group: no pause between ticks or restarts)",
                subprocess.run(ps, capture_output=True, text=True).stdout[-1200:],
            )
        finally:
            if runner.poll() is None:
                os.killpg(runner.pid, signal.SIGKILL)
        _rc, text = d.rk("status")
        st.check(
            "STOPPED" in text.upper(),
            "rk status says the monitor is stopped",
            text[-600:],
        )
        stages.secrets_absent(st, d)


STAGES = {"deploy": deploy_stage}
