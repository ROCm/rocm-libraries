# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's team-prototype stage (tests/container/image.py): the team dashboard in prototype, served by the
monitor from this host, is not reachable from another address of this host, whatever
features.dashboards.team.serve.address says: docker publishes its port on 127.0.0.1 alone.

The deployment is the deploy path's (tests/container/stage_deploy_path.py, DeployPath: the guide's steps on a fresh
machine, each checked, its results under this stage's name), with the team dashboard on, as a person turns a feature
on (QUICK_USER_GUIDE.md, "Turn on a feature"): in prototype, served by the monitor ("serve") on every address of this
host as serve.address gives it (EVERY), at a free port of its own, with public_url set, as the schema asks of a page
served on every address, and no priority box. Its objects are named after the scope <names>-team-prototype. Once the
path is done:
- check 7's row of its own (coherence.TEAM_ROW), in step 2's check and in step 3's confirm, says that in prototype the
  page is served on 127.0.0.1 only, not on EVERY, until it is live; docker/.env gives 127.0.0.1 and serve.port for the
  team dashboard; and docker's record of the container's port bindings publishes its team port
  (container_layout.TEAM_PORT) on 127.0.0.1 alone, at serve.port.
  The deployment's network is internal, so that nothing leaves this host, and on it docker publishes no port at all:
  the stage reads what docker was asked to publish, as no address of this host reaches the port to try;
- the monitor serves the page in the container at the container's own address on the deployment's network, where
  docker delivers what reaches the port it publishes, on that team port: an HTTP GET of / there, as the monitor's
  user, answers 200 (served). A page served on 127.0.0.1 alone in the container would never answer the published
  port, so the stage does not ask there.
Then what the deployment did (DeployPath.after), and the end (DeployPath.clean).
"""

import json
import os

import hostrun
import stage_deploy_path
from stage_deploy_path import STEP_TIMEOUT_S, command_label, free_port, polled, says

STAGE = "team-prototype"
# serve.address: every address of this host, as a person gives it to open the page to other machines
EVERY = "0.0.0.0"
PUBLIC_URL = "https://dashboards.example.com/widgets/"
# How long the stage waits for the page in the container once the path is done
PAGE_WAIT_S = 120
# In the container: an HTTP GET of / at the address argv[1] and the port argv[2] gives, straight, through no proxy; its
# status
GET = (
    "import http.client, sys\n"
    "c = http.client.HTTPConnection(sys.argv[1], int(sys.argv[2]), timeout=10)\n"
    "c.request('GET', '/')\n"
    "print(c.getresponse().status)\n"
)


def env_file(path) -> dict:
    """docker/.env's variables at PATH, NAME -> value; {} when there is none."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        pairs = [x.strip().partition("=") for x in f if not x.lstrip().startswith("#")]
    return {k: v for k, eq, v in pairs if eq}


class TeamPrototype(stage_deploy_path.DeployPath):
    """One run of the stage: the deploy path's (DeployPath), the team dashboard on in prototype, then its checks."""

    def __init__(
        self, engine: str, image: str, pkg: str, names: str, run_opts: list, res
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        self.port = 0  # serve.port
        # each rkmon command, as command_label names it -> its last output
        self.shown = {}

    def rkmon(
        self, *args: str, stdin: bytes = b"", timeout: float = STEP_TIMEOUT_S
    ) -> tuple:
        """DeployPath.rkmon, its output kept (shown)."""
        rc, out = super().rkmon(*args, stdin=stdin, timeout=timeout)
        self.shown[command_label(args)] = out
        return rc, out

    def fill_in(self) -> None:
        """The deploy path's values (DeployPath.fill_in), then the team dashboard's (the module's docstring), through
        the package's one writer of settings files, at a port of its own beside the admin dashboard's.
        """
        super().fill_in()
        from rkmon_lib import file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        dashboards = doc.values["features"]["dashboards"]
        while self.port in (0, dashboards["admin"]["port"]):
            self.port = free_port()
        dashboards["team"] = {
            "enabled": True,
            "mode": "prototype",
            "public_url": PUBLIC_URL,
            "priority_requests": {"enabled": False},
            "delivery": "serve",
            "serve": {"address": EVERY, "port": self.port},
        }
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    def published(self) -> bool:
        """Check 7's row in step 2's check and step 3's confirm, docker/.env, and docker's record of the port bindings
        (the module's docstring)."""
        from rkmon_lib import (  # noqa: E402
            coherence,
            container_layout,
            deploy_plan,
            settings_file,
        )

        here, found = container_layout.LOOPBACK, []
        row = (
            f"PASS {coherence.TEAM_ROW}: in prototype, its page is served on {here} only, not on "
            f"{settings_file.shown_value(EVERY)} ({deploy_plan.TEAM}.serve.address), until it is live"
        )
        for label in ("./rkmon check", "./rkmon confirm"):
            if not says(self.shown.get(label, ""), row):
                found.append(f"{label} has no row that says {row!r}")
        env = env_file(self.copy / "docker" / ".env")
        team = {k: env.get(k) for k in ("RKMON_TEAM_HOST", "RKMON_TEAM_PORT")}
        if team != {"RKMON_TEAM_HOST": here, "RKMON_TEAM_PORT": str(self.port)}:
            found.append(f"docker/.env gives the team dashboard {team}")
        bindings = json.loads(
            self.must(
                "inspect",
                "--format",
                "{{json .HostConfig.PortBindings}}",
                self.monitor(),
            )
            or "null"
        )
        team_port = f"{container_layout.TEAM_PORT}/tcp"
        got = (bindings or {}).get(team_port)
        if got != [{"HostIp": here, "HostPort": str(self.port)}]:
            found.append(f"docker publishes {team_port} as {got}")
        if found:
            hostrun.indented("\n".join(found))
        return self.result(
            not found,
            "check 7's row, at check and at confirm, says the team dashboard in prototype is served on 127.0.0.1 only, "
            "not on the address serve.address gives, until it is live; docker/.env gives 127.0.0.1 and serve.port for "
            "it, and docker publishes the container's team port on 127.0.0.1 alone, at serve.port",
        )

    def own_address(self) -> str:
        """The container's address on the deployment's network, as docker records it: where docker delivers what
        reaches a port it publishes."""
        where = self.must(
            "inspect",
            "--format",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
            self.monitor(),
        ).split()
        if len(where) != 1:
            raise RuntimeError(f"the container has {len(where)} addresses, not 1")
        return where[0]

    def served(self) -> bool:
        """The page in the container at its own address (own_address) and the team port, as the monitor's user sees it
        (GET), within PAGE_WAIT_S."""
        from rkmon_lib import container_layout  # noqa: E402

        port = str(container_layout.TEAM_PORT)
        try:
            address = self.own_address()
        except RuntimeError as exc:
            address, said = "", {"unread": str(exc)[-300:]}
        if address:
            said = polled(
                lambda: {
                    "status": self.as_monitor("python3", "-I", "-c", GET, address, port)
                },
                lambda a: a["status"].strip() == "200",
                PAGE_WAIT_S,
            )
        ok = said.get("status", "").strip() == "200"
        if not ok:
            hostrun.indented(f"GET / at {address}:{port} in the container: {said}")
        return self.result(
            ok,
            "the monitor serves the team dashboard's page in the container at the container's own address, where "
            "docker delivers the team port it publishes, not on 127.0.0.1 alone",
        )


def team_prototype_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = TeamPrototype(engine, image, pkg, names, run_opts, res)
    try:
        t.prepare()
        if t.path() >= 6 and t.done:
            t.published()
            t.served()
        t.after()
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
