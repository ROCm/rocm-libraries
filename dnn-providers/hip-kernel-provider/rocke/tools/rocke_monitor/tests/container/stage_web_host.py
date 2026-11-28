# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's web-host stage (tests/container/image.py): `rkmon web-host install`, which puts
the request service on the team dashboard's web host, and the doctor's row of that service, against a
stand-in web host, on the image under test.

The deployment is the deploy path's (tests/container/stage_deploy_path.py, DeployPath: the guide's eight steps on a
fresh machine, each checked, its results under this stage's name), its settings with the team dashboard on, in
prototype, published to the stand-in with priority requests on. Its objects are named after the scope <names>-web-host,
the stand-in's container too (<scope>-web); each is removed at the end, the stand-in's first, as it is on the
deployment's network.

The stand-in web host is a container of the image under test on the deployment's internal network, made before the
path starts, which Docker's DNS gives the web host's name there (a network alias). It runs tests/fakes/fake_sshd.py on
port 22, with a host key and a login key made there, and has a systemctl, loginctl and crontab of its own
(tests/fakes/web_host/bin), whose systemctl runs the unit the install puts there as its service. The login key's private
half goes to the person's folder (the stage's temporary one) and leaves the stand-in, as an admin hands a key over;
nothing prints it. The stand-in's log says what each connection did: whether one reached it, and as whom it logged in.

Once the monitor has started, in order, each command run as the guide's person types it (DeployPath.rkmon):
- `rkmon web-host install` before web_host_ssh_key is set: FAILED as an earlier step's (exit 3), its fix the command
  that sets the key, which the writer's rk shows as the wrapper was typed (./rkmon), and no connection reaches the
  stand-in.
- `rkmon secret web_host_ssh_key FILE` with no terminal: the stand-in's own key shown by its fingerprint, then
  CANCELLED, the key set and nothing recorded; then the install: FAILED as an earlier step's, the web host's own key
  not recorded, and no connection.
- The same secret on a terminal, its question answered yes: SET, the web host's own key recorded.
- In standby, entered as the one-time move enters it (the internal state's standby.on, set in place), the install is
  REFUSED, naming standby and its end in the words both refusals say (rkmon's in state 14, and rk web-host-install's
  own where the state does not show standby), and no connection; standby then ends as it began.
- With a config.json in the monitor's home that names another login and port, the install is INSTALLED as the settings
  you confirmed say: each login the stand-in saw was theirs, so the install read no config.json. The file then goes.
- The service answers at the team dashboard's address, asked from the monitor's container.
- The team dashboard taken live, as QUICK_USER_GUIDE.md's "Go live" takes it (the settings file, confirm, then the apply
  its next line gives): doctor passes the request service's row; once the service stops, doctor fails the row, whose
  fix is the install, then doctor; that fix run, doctor passes the row again.
- The stand-in's host key changed: the install FAILED (exit 4), ssh refusing the host before any login.
Each check that the next ones need ends the stage when it fails: the key recorded, the service installed, the team
dashboard live.
"""

import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import time

import hostrun
import stage_deploy_path
from stage_deploy_path import FAKES_USER, STEP_TIMEOUT_S, says

STAGE = "web-host"
WEB_HOST = "web.example.com"  # the team dashboard's web host: the stand-in's name on the deployment's network
WEB_LOGIN = "pages"  # the web host's login
WEB_PORT = 8080  # the request service's, in the team dashboard's public address
SITE_FOLDER = "team/site"
WEB_DIR = "/tmp/web-host"  # in the stand-in's container: its code
# and its keys, its log and its login's home, on a tmpfs of its own: the image's /tmp is root's
WEB_STATE = "/web-host-state"
WEB_HOME = f"{WEB_STATE}/home"
# The question `rkmon secret` asks on a terminal before it records the web host's own keys, and the answer typed.
RECORD = (b"the only ones ssh trusts it by?", b"y\n")
# A config.json the install must not read, which names the web host with another login, site folder and port.
DECOY = {
    "publish": {
        "html": {
            "host": WEB_HOST,
            "user": "decoy",
            "site": "decoy/site",
            "url": f"http://{WEB_HOST}:{WEB_PORT + 1}/",
        }
    }
}
# Standby entered, or ended, as the one-time move enters it, there being no command that does: python3 -c in the
# folder of the code the monitor runs, as the monitor, with "on" or "off".
STANDBY = (
    "import sys\n"
    "from rkmon_lib import config, one_time_move, state_store, store\n"
    "on = sys.argv[1] == 'on'\n"
    "for name, value in (\n"
    "    ('standby.on', on),\n"
    "    ('standby.reason', one_time_move.REASON if on else ''),\n"
    "    ('standby.since', store.now() if on else ''),\n"
    "):\n"
    "    state_store.set(config.home(), name, value)\n"
)
# The answer at HOST:PORT to a GET of /: its status and its Server header (python3 -c HOST PORT).
ASK = (
    "import http.client, sys\n"
    "c = http.client.HTTPConnection(sys.argv[1], int(sys.argv[2]), timeout=10)\n"
    "c.request('GET', '/')\n"
    "r = c.getresponse()\n"
    "print(r.status, r.getheader('Server') or '')\n"
)


def events_of(events: list, kind: str) -> list:
    """The events of the stand-in's log of the kind KIND: connect, kex, auth, exec, exit, closed."""
    return [e for e in events if e.get("event") == kind]


class WebHost(stage_deploy_path.DeployPath):
    """One run of the stage: the deploy path's (DeployPath), under the stage's name, with the stand-in web host."""

    def __init__(
        self, engine: str, image: str, pkg: str, names: str, run_opts: list, res
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        self.web = f"{self.scope}-web"
        self.key_file = self.home / "web_host_ssh_key"
        self.url = f"http://{WEB_HOST}:{WEB_PORT}/"

    # --- the fresh machine, with a web host ---------------------------------------------------------------------------
    def prepare(self) -> None:
        """The deploy path's fresh machine (DeployPath.prepare), then the stand-in web host on its network, and the
        login key's private half in the person's folder."""
        super().prepare()
        code = self.top / "web-host"
        fakes = pathlib.Path(self.pkg) / "tests" / "fakes"
        shutil.copytree(fakes / "web_host", code)
        shutil.copy(fakes / "fake_sshd.py", code)
        keygen = "ssh-keygen -q -t ed25519 -N '' -C '' -f"
        server = shlex.join(
            [
                "python3",
                "-I",
                "-B",
                f"{WEB_DIR}/fake_sshd.py",
                "--host-key",
                f"{WEB_STATE}/host_key",
                "--authorized-key",
                f"{WEB_STATE}/login_key.pub",
                "--user",
                WEB_LOGIN,
                "--home",
                WEB_HOME,
                "--port",
                "22",
                "--path",
                f"{WEB_DIR}/bin:/usr/local/bin:/usr/bin:/bin",
                "--log",
                f"{WEB_STATE}/sshd.log",
                "--ready",
                f"{WEB_STATE}/ready",
            ]
        )
        uid, gid = FAKES_USER.split(":")
        self.must(
            "create",
            "--name",
            self.web,
            "--network",
            self.network,
            "--network-alias",
            WEB_HOST,
            "--init",  # which reaps the service once it stops: the server is no init
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            FAKES_USER,
            # ssh's port 22, for a user who is not root
            "--sysctl",
            "net.ipv4.ip_unprivileged_port_start=0",
            "--tmpfs",
            f"{WEB_STATE}:uid={uid},gid={gid},mode=0700,size=64m",
            *self.run_opts,
            "--entrypoint",
            "sh",
            self.image,
            "-c",
            f"mkdir {WEB_HOME} && {keygen} {WEB_STATE}/host_key && {keygen} {WEB_STATE}/login_key && exec {server}",
        )
        self.must("cp", f"{code}/.", f"{self.web}:{WEB_DIR}")
        self.must("start", self.web)
        ready = ("exec", self.web, "test", "-f", f"{WEB_STATE}/ready")
        for _ in range(60):
            if self.docker(*ready)[0] == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"the stand-in web host did not start: {self.docker('logs', self.web)[1][-1500:]}"
            )
        # docker cp reads no tmpfs: the stand-in's user reads the key out, never shown
        key = self.must("exec", self.web, "cat", f"{WEB_STATE}/login_key")
        fd = os.open(self.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as w:
            w.write(key + "\n")
        self.must("exec", self.web, "rm", f"{WEB_STATE}/login_key")

    def fill_in(self) -> None:
        """The deploy path's values (DeployPath.fill_in), and the team dashboard on, in prototype, published to the
        stand-in with priority requests on."""
        super().fill_in()
        self.team(
            enabled=True,
            mode="prototype",
            delivery="publish",
            public_url=self.url,
            priority_requests={"enabled": True},
            publish={
                "web_host": WEB_HOST,
                "username": WEB_LOGIN,
                "site_folder": SITE_FOLDER,
            },
        )

    def team(self, **values) -> None:
        """The settings file's features.dashboards.team with VALUES set, through the package's one writer of settings
        files."""
        from rkmon_lib import file_types, settings_file  # noqa: E402

        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        doc.values["features"]["dashboards"]["team"].update(values)
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    # --- what the stage reads ----------------------------------------------------------------------------------------
    def monitor(self) -> str:
        """The deployment's container, by the labels docker compose gives it: its project's, its service's, no
        one-off."""
        from rkmon_lib import deploy_plan, identity  # noqa: E402

        names = self.must(
            "ps",
            "--format",
            "{{.Names}}",
            "--filter",
            f"label={identity.PROJECT}={self.scope}",
            "--filter",
            f"label=com.docker.compose.service={deploy_plan.SERVICE}",
            "--filter",
            "label=com.docker.compose.oneoff=False",
        ).split()
        if len(names) != 1:
            raise RuntimeError(f"the deployment's container: {len(names)} run, not 1")
        return names[0]

    def as_monitor(self, *args: str) -> str:
        """ARGS run in the deployment's container as the monitor's user, in the folder of the code it runs (`rk
        --which-code`), where python3 imports the package: their output; RuntimeError when they fail.
        """
        from rkmon_lib import container_layout  # noqa: E402

        where, user = self.monitor(), container_layout.MONITOR.name
        code = self.must("exec", "-u", user, where, "rk", "--which-code").strip()
        return self.must("exec", "-u", user, "-w", code, where, *args)

    def events(self) -> list:
        """What the stand-in's log says each connection did (tests/fakes/fake_sshd.py --log), in order."""
        rc, out = self.docker("exec", self.web, "cat", f"{WEB_STATE}/sshd.log")
        if rc != 0:
            return []
        return [json.loads(x) for x in out.splitlines() if x.startswith("{")]

    def fingerprint(self) -> str:
        """The stand-in's host key's SHA256 fingerprint, as ssh-keygen shows it."""
        out = self.must(
            "exec", self.web, "ssh-keygen", "-l", "-f", f"{WEB_STATE}/host_key.pub"
        )
        return next((w for w in out.split() if w.startswith("SHA256:")), "")

    def row(self, out: str, status: str) -> bool:
        """Whether OUT, doctor's, has the request service's row with STATUS: PASS, the service answers at the team
        dashboard's address; FAIL, it does not, and the row's fix is the install, then doctor.
        """
        text = " ".join(out.split())
        if status == "PASS":
            return (
                f"PASS team dashboard: the request service on the web host answers at {self.url}"
                in text
            )
        failed = (
            r"FAIL team dashboard: the request service on the web host does not answer at the team dashboard's "
            rf"address, {re.escape(self.url)} \(.*?\) fix: (\S+) web-host install, then \1 doctor"
        )
        return re.search(failed, text) is not None

    # --- the person's commands ---------------------------------------------------------------------------------------
    def install(self) -> tuple:
        """`rkmon web-host install`: its exit code, its output, and the stand-in's log of what happened meanwhile."""
        before = len(self.events())
        rc, out = self.rkmon("web-host", "install")
        return rc, out, self.events()[before:]

    def on_a_terminal(self, *args: str, replies=()) -> tuple:
        """`./rkmon ARGS` as DeployPath.rkmon runs it, but on a terminal of its own (tests/sandbox.py on_a_terminal),
        each of REPLIES typed once its prompt shows: its exit code and its output, shown.
        """
        sys.path.insert(0, str(pathlib.Path(self.pkg) / "tests"))
        import sandbox  # noqa: E402

        rc, out = sandbox.on_a_terminal(
            ["./rkmon", *args],
            self.copy / "docker",
            self.env,
            replies=replies,
            timeout=STEP_TIMEOUT_S,
        )
        out = out.replace("\r\n", "\n").rstrip("\n")
        hostrun.indented(f"$ ./rkmon {shlex.join(args)}   (on a terminal)\n{out}")
        return rc, out

    # --- the checks --------------------------------------------------------------------------------------------------
    def web_host(self) -> None:
        """The stage's checks once the monitor has started, in the module docstring's order."""
        self.before_the_key()
        self.not_recorded()
        if not self.recorded():
            return
        self.in_standby()
        if not self.installed():
            return
        self.answers()
        if not self.live():
            return
        self.stopped()
        self.another_host_key()

    def before_the_key(self) -> None:
        rc, out, new = self.install()
        self.result(
            rc == 3
            and says(out, "FAILED: web_host_ssh_key is not set; nothing changed")
            and says(out, "fix: ./rkmon secret web_host_ssh_key, and give ")
            and not events_of(new, "connect"),
            "web-host install before web_host_ssh_key is set fails as an earlier step's, its fix the command that sets "
            "the key, as the writer's rk shows it with the wrapper's name as typed (./rkmon), and reaches no host",
        )

    def not_recorded(self) -> None:
        shown = self.fingerprint()
        rc, out = self.rkmon("secret", "web_host_ssh_key", str(self.key_file))
        self.result(
            rc == 1
            and says(out, "CANCELLED: web_host_ssh_key is set")
            and bool(shown)
            and shown in out,
            "secret web_host_ssh_key with no terminal sets the key and shows the web host's own key by its "
            "fingerprint, which it records only once you accept it on a terminal",
        )
        rc, out, new = self.install()
        self.result(
            rc == 3
            and says(
                out,
                f"own key is not recorded, so ssh does not reach {WEB_HOST}; nothing changed",
            )
            and not events_of(new, "connect"),
            "web-host install while the web host's own key is not recorded fails as an earlier step's, and reaches no "
            "host",
        )

    def recorded(self) -> bool:
        shown = self.fingerprint()
        rc, out = self.on_a_terminal(
            "secret", "web_host_ssh_key", str(self.key_file), replies=(RECORD,)
        )
        return self.result(
            rc == 0
            and says(
                out,
                f"SET: web_host_ssh_key; the web host's own keys ({WEB_HOST}) are recorded",
            )
            and bool(shown)
            and shown in out,
            "secret web_host_ssh_key on a terminal shows the web host's own key by its fingerprint, and records it "
            "once you accept it",
        )

    def in_standby(self) -> None:
        self.as_monitor("python3", "-c", STANDBY, "on")
        try:
            rc, out, new = self.install()
        finally:
            self.as_monitor("python3", "-c", STANDBY, "off")
        # the words of both refusals: rkmon's in state 14, and rk web-host-install's own where the state does
        # not show standby
        self.result(
            rc == 4
            and says(out, "REFUSED: ")
            and says(out, "in standby")
            and says(out, "standby off")
            and not events_of(new, "connect"),
            "web-host install in standby is refused, naming standby, and reaches no host",
        )

    def installed(self) -> bool:
        from rkmon_lib import container_layout, runtime_config  # noqa: E402
        from rkmon_lib import web_host_install  # noqa: E402

        decoy = f"{container_layout.MONITOR_HOME}/{runtime_config.CONFIG_JSON}"
        self.as_monitor(
            "python3",
            "-c",
            "import sys; open(sys.argv[1], 'x').write(sys.argv[2])",
            decoy,
            json.dumps(DECOY),
        )
        try:
            rc, out, new = self.install()
        finally:
            self.as_monitor("rm", "-f", decoy)
        logins = {e.get("user") for e in events_of(new, "auth")}
        return self.result(
            rc == 0
            and says(
                out,
                f"INSTALLED: the request service on {WEB_HOST}, {WEB_LOGIN}'s user service "
                f"{web_host_install.UNIT}, answers on port {WEB_PORT}",
            )
            and logins == {WEB_LOGIN},
            "web-host install puts the request service on the web host over its key, as the settings you confirmed "
            "say, not as a config.json in the monitor's home says",
        )

    def answers(self) -> None:
        from rkmon_lib import container_layout  # noqa: E402

        sys.path.insert(0, str(pathlib.Path(self.pkg) / "tools"))
        import webhost_server  # noqa: E402

        rc, out = self.docker(
            "exec",
            "-u",
            container_layout.MONITOR.name,
            self.monitor(),
            "python3",
            "-I",
            "-c",
            ASK,
            WEB_HOST,
            str(WEB_PORT),
        )
        hostrun.indented(f"GET {self.url} from the monitor's container: {out}")
        self.result(
            rc == 0 and out.split()[1:2] == [webhost_server.Handler.server_version],
            "the request service answers at the team dashboard's address, asked from the monitor's container",
        )

    def live(self) -> bool:
        self.team(mode="live")
        rc, out = self.rkmon("confirm", "--settings", str(self.file), "--yes")
        if rc == 0:
            rc, out = self.follow(out)
        if not self.result(
            rc == 0 and says(out, "APPLIED:"),
            "the team dashboard taken live: the settings file, confirm, then the apply its next line gives",
        ):
            return False
        _rc, out = self.rkmon("doctor")
        return self.result(
            self.row(out, "PASS"),
            "live, doctor passes the request service's row: it answers at the team dashboard's address",
        )

    def stopped(self) -> None:
        from rkmon_lib import web_host_install  # noqa: E402

        self.must(
            "exec",
            "-e",
            f"HOME={WEB_HOME}",
            self.web,
            f"{WEB_DIR}/bin/systemctl",
            "--user",
            "stop",
            web_host_install.UNIT,
        )
        rc, out = self.rkmon("doctor")
        self.result(
            rc != 0 and self.row(out, "FAIL"),
            "once the service stops, doctor fails its row, whose fix is web-host install, then doctor",
        )
        rc, out, _new = self.install()
        again = rc == 0 and says(out, "INSTALLED:")
        _rc, out = self.rkmon("doctor")
        self.result(
            again and self.row(out, "PASS"),
            "that fix run, web-host install, then doctor passes the row again",
        )

    def another_host_key(self) -> None:
        new_key = f"{WEB_STATE}/new_key"
        self.must(
            "exec",
            self.web,
            "sh",
            "-c",
            f"ssh-keygen -q -t ed25519 -N '' -C '' -f {new_key} && mv {new_key}.pub {WEB_STATE}/host_key.pub "
            f"&& mv {new_key} {WEB_STATE}/host_key",
        )
        rc, out, new = self.install()
        self.result(
            rc == 4
            and says(
                out,
                f"FAILED: ssh did not reach {WEB_LOGIN}@{WEB_HOST} (Host key verification failed.); nothing changed",
            )
            and bool(events_of(new, "connect"))
            and not events_of(new, "auth"),
            "web-host install to a web host whose own key is not the one recorded fails at the host key, before any "
            "login",
        )

    # --- the end -----------------------------------------------------------------------------------------------------
    def clean(self) -> None:
        """The stand-in web host's container first, as it is on the deployment's network; then what the deploy path's
        clean removes (DeployPath.clean)."""
        self.docker("rm", "-f", self.web)
        super().clean()


def web_host_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = WebHost(engine, image, pkg, names, run_opts, res)
    try:
        t.prepare()
        if t.path() >= 6:  # the monitor has started
            t.web_host()
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
