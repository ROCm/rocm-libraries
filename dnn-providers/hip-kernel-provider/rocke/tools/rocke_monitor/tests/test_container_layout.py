# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's layout (rkmon_lib/container_layout.py): its volumes, ports, listen address and users fit together;
the image, the compose file, the entry point and the settings' defaults name the same ones; the scripts the image runs
as root or through sudo (docker/rk_*.py) and the Python code read them there (an installed copy of the worker tools
reads the values its install wrote, and runs no file it finds); and no Python file of the package, docker/rkmon among
them, keeps a copy of its own, not even inside a text, nor does the image's test (tests/container/) keep the path of
the image's ROCm record.
"""

import ast
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    deliver,
    deploy,
    deploy_plan,
    deployment,
    derived,
    paths,
    prcode,
    webserve,
)
import pr_user  # noqa: E402
from test_deploy_plan import plan as deploy_plan_of  # noqa: E402

# A container_layout.py where an installed copy of the worker tools must never look: it leaves a mark when it runs.
PLANTED = (
    "import pathlib, types\n"
    "pathlib.Path(__file__).with_name('RAN').write_text('ran')\n"
    "PR_RUNNER = types.SimpleNamespace(name='planted', group='planted')\n"
    "PR_GROUP = SLOT_HOME = 'planted'\n"
)
# pr_user.py in a fresh interpreter: the import path as worker/drafter.py makes it (the folder above the tools' own
# first), then what it says, with this machine's launcher and without, and the layouts it loaded.
PROBE = """
import json, sys
tools, top, launcher = sys.argv[1:]
sys.path[:0] = [top, tools]
import pr_user
env = {"RKMON_PR_RUNNER": launcher}
argv, data = pr_user.wrap(["a"], {"X": "1"}, environ=env)
other = ["sudo", "-n", "-u", "x", launcher, "run", "--", "a"]
print(json.dumps({
    "argv": argv,
    "data": data.decode(),
    "launched": [
        pr_user.launched(argv, env),
        pr_user.launched(["a"], env),
        pr_user.launched(argv, {}),
        pr_user.launched(other, {"RKMON_PR_RUNNER": "/elsewhere"}),
    ],
    "host": pr_user.wrap(["a"], {}, environ={}),
    "values": pr_user.values(),
    "layouts": sorted(m for m in sys.modules if m.endswith("container_layout")),
}))
"""

DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = (ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
ENTRY = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")


def service(name: str) -> str:
    """The compose file's service NAME: its lines, up to the next service or the next top-level key."""
    (lines,) = re.findall(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  \S|^\S|\Z)", COMPOSE
    )
    return lines


def instructions(name: str) -> list:
    """The Dockerfile's `name` instructions, each with its continued lines joined."""
    return [
        re.sub(r"\\\n\s*", " ", text)
        for text in re.findall(rf"(?ms)^{name} (.*?[^\\])$", DOCKERFILE)
    ]


class TheLayoutTest(unittest.TestCase):
    def test_the_volumes(self):
        self.assertEqual(L.VOLUMES, (L.WORK_DIR, L.SECRETS_DIR, L.WRITE_SECRETS_DIR))
        for v in L.VOLUMES:
            self.assertTrue(v.startswith("/") and not v.endswith("/"), v)
        self.assertEqual(len(set(L.VOLUMES)), 3)
        self.assertEqual(L.WRITE_LOGIN_DIR.rsplit("/", 1)[0], L.WRITE_SECRETS_DIR)
        self.assertEqual(L.MONITOR_HOME.rsplit("/", 1)[0], L.WORK_DIR)

    def test_the_settings_volumes_folders(self):
        """The container's read-only folder of the settings volume and the one-off writer's: apart, and in none of the
        three volumes, whose files each start renumbers (docker/rk_ids.py). The layer's file and the team's skills
        folder are two names in it."""
        for d in (L.SETTINGS_DIR, L.LAYER_WRITE_DIR):
            self.assertTrue(d.startswith("/") and not d.endswith("/"), d)
            for v in L.VOLUMES:
                self.assertFalse(d.startswith(v + "/") or v.startswith(d + "/"), d)
            self.assertNotIn(d, L.VOLUMES)
        self.assertNotEqual(L.SETTINGS_DIR, L.LAYER_WRITE_DIR)
        self.assertNotIn("/", L.LAYER_FILE)
        self.assertNotIn("/", L.SKILLS_FOLDER)
        self.assertNotEqual(L.SKILLS_FOLDER, L.LAYER_FILE)

    def test_the_ports_and_the_listen_address(self):
        self.assertNotEqual(L.ADMIN_PORT, L.TEAM_PORT)
        for p in (L.ADMIN_PORT, L.TEAM_PORT):
            self.assertTrue(1024 <= p <= 65535, p)
        self.assertEqual(L.EVERY_ADDRESS, "0.0.0.0")
        self.assertEqual(L.LISTEN_ADDRESS, L.EVERY_ADDRESS)

    def test_the_users(self):
        self.assertEqual(L.USERS, (L.MONITOR, L.WRITER, *L.PR_RUNNERS))
        self.assertEqual(len({u.name for u in L.USERS}), len(L.USERS))
        self.assertEqual(len({u.uid for u in L.USERS}), len(L.USERS))
        ids = deployment.rk_ids()
        for u in L.USERS:
            self.assertFalse(ids.system_id(u.uid), u)
        # The writer's group is the monitor's; no PR runner is in a group of the monitor's, the monitor in theirs.
        self.assertEqual(
            (L.WRITER.group, L.WRITER.gid), (L.MONITOR.group, L.MONITOR.gid)
        )
        self.assertNotEqual(L.PR_GID, L.MONITOR.gid)
        self.assertIn(L.PR_GROUP, L.MONITOR.groups)
        self.assertNotIn(L.PR_GROUP, L.WRITER.groups)
        with self.assertRaises(AttributeError):
            L.MONITOR.uid = 0

    def test_one_pr_runner_per_slot(self):
        self.assertEqual(len(L.PR_RUNNERS), L.SLOTS)
        self.assertGreaterEqual(L.SLOTS, 2)
        self.assertEqual(L.PR_RUNNER, L.PR_RUNNERS[0])
        for i, u in enumerate(L.PR_RUNNERS, 1):
            with self.subTest(runner=u.name):
                self.assertEqual(u.name, f"{L.PR_GROUP}{i}")
                # Its own group, of its name, is its primary one; it is in the runners' group too.
                self.assertEqual((u.group, u.groups), (u.name, (L.PR_GROUP,)))
                self.assertNotIn(L.MONITOR.group, u.groups)
        self.assertNotIn(L.PR_GID, (L.MONITOR.gid,))
        # The runners' own groups: one each, on a range of ids of their own (the runners' group's id is the first
        # runner's user id), none a system account's.
        ids = deployment.rk_ids()
        gids = {u.gid for u in L.PR_RUNNERS}
        self.assertEqual(len(gids), L.SLOTS)
        self.assertFalse(
            gids & {L.PR_GID, L.MONITOR.gid, *(u.uid for u in L.PR_RUNNERS)}, gids
        )
        self.assertFalse([g for g in gids if ids.system_id(g)])
        # The slots are on the RAM disk, in no volume: what a job leaves goes with the container, and the start's
        # renumbering of the volumes (docker/rk_ids.py) never meets a slot.
        self.assertEqual(L.SLOTS_DIR, f"{L.RAMDISK}/slots")
        self.assertFalse(
            any(L.RAMDISK == v or L.RAMDISK.startswith(v + "/") for v in L.VOLUMES)
        )
        self.assertTrue(L.SLOT_LOCKS.startswith(L.SLOTS_DIR + "/"))
        self.assertNotIn(L.SLOT_LOCKS.rsplit("/", 1)[1], {u.name for u in L.PR_RUNNERS})
        self.assertTrue(L.SLOT_INPUTS.startswith(L.SLOTS_DIR + "/"))
        self.assertNotIn(
            L.SLOT_INPUTS.rsplit("/", 1)[1], {u.name for u in L.PR_RUNNERS}
        )
        self.assertNotEqual(L.SLOT_INPUTS, L.SLOT_LOCKS)
        self.assertEqual(L.SLOT_FOLDERS, ("out", "build", "tmp", "home"))


class TheWrappersNameTest(unittest.TestCase):
    """The name the host's wrapper was run by, as the writer's rk gets it (shown_wrapper): the name as typed while it is
    a path that ends in rkmon, of the portable file name characters and / alone, not beginning with -, and at most
    WRAPPER_MAX characters long; else rkmon."""

    def shown(self, typed: str) -> str:
        return L.shown_wrapper({L.WRAPPER_VARIABLE: typed})

    def test_a_path_that_ends_in_rkmon_shows_as_typed(self):
        longest = "a" * (L.WRAPPER_MAX - len("/rkmon")) + "/rkmon"
        self.assertEqual(len(longest), L.WRAPPER_MAX)
        for typed in (
            "rkmon",
            "./rkmon",
            "docker/rkmon",
            "../docker/rkmon",
            "/opt/rocke_monitor-1.2/docker/rkmon",
            longest,
        ):
            with self.subTest(typed=typed):
                self.assertEqual(self.shown(typed), typed)

    def test_an_unset_name_shows_as_rkmon(self):
        self.assertEqual(L.shown_wrapper({}), "rkmon")
        self.assertEqual(self.shown(""), "rkmon")

    def test_a_name_with_another_character_shows_as_rkmon(self):
        for typed in (
            "/tmp/a b/rkmon",
            "./x;id;/rkmon",
            "$(id)/rkmon",
            "`id`/rkmon",
            "/tmp/'q'/rkmon",
            "/tmp/x*/rkmon",
            "~/rkmon",
            "/tmp/\x1b[2J/rkmon",
            "/tmp/\n/rkmon",
            "./rkmon\n",
            " ./rkmon",
            "/tmp/\u00e9/rkmon",
        ):
            with self.subTest(typed=typed):
                self.assertEqual(self.shown(typed), "rkmon")

    def test_a_name_that_does_not_end_in_rkmon_shows_as_rkmon(self):
        for typed in (
            "./rkmon-team",
            "./rkmonx",
            "./notrkmon",
            "docker/rkmon/",
            "docker/rkmon.py",
            "/bin/sh",
        ):
            with self.subTest(typed=typed):
                self.assertEqual(self.shown(typed), "rkmon")

    def test_a_name_that_begins_with_a_dash_shows_as_rkmon(self):
        for typed in ("-x/rkmon", "--help/rkmon"):
            with self.subTest(typed=typed):
                self.assertEqual(self.shown(typed), "rkmon")

    def test_a_name_over_the_cap_shows_as_rkmon(self):
        over = "a" * (L.WRAPPER_MAX - len("/rkmon") + 1) + "/rkmon"
        self.assertEqual(len(over), L.WRAPPER_MAX + 1)
        self.assertEqual(self.shown(over), "rkmon")

    def test_the_name_is_the_program_s_and_the_wrapper_s_file_s(self):
        """WRAPPER_NAME is the program's name and the wrapper's file's, and the doctor's fixes read the variable."""
        from rkmon_lib import doctor, program

        self.assertEqual(L.WRAPPER_NAME, program.NAME)
        self.assertTrue((ROOT / "docker" / L.WRAPPER_NAME).is_file())
        self.assertEqual(
            doctor.wrapper_name({L.WRAPPER_VARIABLE: "./rkmon"}), "./rkmon"
        )


class TheImageAgreesTest(unittest.TestCase):
    def test_nothing_of_the_image_is_every_users_to_write(self):
        """A PR runner writes only its job's folders: the image takes o+w off every folder and file of its own file
        system (never through a link); each start gives SYSTEM_TMP the monitor's group (docker/rk_layout.py), and
        SHM_DIR, mounted at each start, is not the image's."""
        self.assertIn(
            "&& find / -xdev -perm -0002 ! -type l -exec chmod o-w {} +", DOCKERFILE
        )
        self.assertEqual(L.SYSTEM_TMP, ("/tmp", "/var/tmp"))
        self.assertNotIn(L.SHM_DIR, L.SYSTEM_TMP)

    def test_the_image_makes_the_users_and_their_groups(self):
        groups = {
            n: int(g) for g, n in re.findall(r"groupadd -g (\d+) (\S+)", DOCKERFILE)
        }
        self.assertEqual(groups, {L.MONITOR.group: L.MONITOR.gid, L.PR_GROUP: L.PR_GID})
        users = {
            name: (int(uid), group, tuple(other.split(",")) if other else ())
            for uid, group, other, name in re.findall(
                r"useradd -m -u (\d+) -g (\S+)(?: -G (\S+))? -s \S+ (\S+)", DOCKERFILE
            )
        }
        self.assertEqual(
            users,
            {u.name: (u.uid, u.group, u.groups) for u in (L.MONITOR, L.WRITER)},
        )
        # The runners: a loop over the slots, each with a group of its own, its primary one, in the runners' group
        # too, and with its home in its slot's folder.
        ((count, gtop, gname, home, top, group, other, name),) = re.findall(
            r"for i in \$\(seq 1 (\d+)\); do \\\n\s+groupadd -g \$\(\((\d+) - i\)\) (\S+)\$i \\\n"
            r"\s+&& useradd -M -d (\S+)\$i/home -u \$\(\((\d+) - i\)\) -g (\S+)\$i -G (\S+) -s \S+ (\S+)\$i ",
            DOCKERFILE,
        )
        self.assertEqual((gname, group), (name, name))
        runners = tuple(
            L.User(
                f"{name}{i}",
                int(top) - i,
                f"{name}{i}",
                int(gtop) - i,
                tuple(other.split(",")),
            )
            for i in range(1, int(count) + 1)
        )
        self.assertEqual(runners, L.PR_RUNNERS)
        self.assertEqual(home, f"{L.SLOTS_DIR}/{L.PR_GROUP}")
        # The sudo rules: the monitor's user runs each helper as the user it needs, and no other user has one.
        self.assertIn(f"'Defaults:{L.MONITOR.name} !requiretty'", DOCKERFILE)
        self.assertIn(
            f"'{L.MONITOR.name} ALL=({L.WRITER.name}) NOPASSWD: {L.IMAGE_CODE}/docker/rk_write.py'",
            DOCKERFILE,
        )
        # The start's root step (docker/rk_root.py) runs in the entry point's root phase only: no rule names it, and
        # no rule runs anything as root.
        rules = re.findall(r"ALL=\(([^)]*)\) NOPASSWD: (\S+)", DOCKERFILE)
        self.assertEqual(len(rules), 2, rules)
        self.assertNotIn("root", [who for who, _ in rules])
        self.assertFalse([p for _, p in rules if "rk_root" in p])
        self.assertEqual(
            set(re.findall(r"(\S+) ALL=\(", DOCKERFILE)), {f"'{L.MONITOR.name}"}
        )
        # The launcher's rule: as any runner, listed one by one (the monitor is in their group, so not `%rkpr`).
        self.assertIn(
            f"printf '{L.MONITOR.name} ALL=(%s) NOPASSWD: /opt/rocke_monitor/docker/rk_prcode.py\\n' "
            f"\"$(seq -s, -f '{L.PR_GROUP}%g' 1 {L.SLOTS})\"",
            DOCKERFILE,
        )

    def test_the_image_makes_the_volumes_and_works_in_the_first(self):
        (volume,) = re.findall(r"(?m)^VOLUME (\[.*\])$", DOCKERFILE)
        self.assertEqual(tuple(json.loads(volume)), (*L.VOLUMES, L.CODE_DIR))
        self.assertIn("mkdir -p " + " ".join(L.VOLUMES), DOCKERFILE)
        self.assertIn(
            f"mkdir -p -m 755 {L.CODE_DIR}", DOCKERFILE
        )  # root's: no user writes the code
        self.assertEqual(re.findall(r"(?m)^WORKDIR (\S+)$", DOCKERFILE), [L.WORK_DIR])
        self.assertIn(
            f"chown {L.WRITER.name}:{L.MONITOR.group} {L.WRITE_SECRETS_DIR}", DOCKERFILE
        )

    def test_the_image_makes_its_checks_venv(self):
        self.assertTrue(
            [t for t in instructions("RUN") if f" python3 -m venv {L.IMAGE_VENV} " in t]
        )

    def test_the_image_writes_its_rocm_record_where_the_layout_reads_it(self):
        """The image's build (docker/install_rocm.sh) writes its ROCm record at ROCM_RECORD, with the folder of the
        release's LLVM tools as `llvm_bin`, after it found clang there: what the container's paths.rocm_llvm_bin is.
        """
        script = (ROOT / "docker" / "install_rocm.sh").read_text(encoding="utf-8")
        self.assertIn(
            f'with open("{L.ROCM_RECORD}", "w", encoding="utf-8") as f:', script
        )
        self.assertIn('"llvm_bin": llvm_bin', script)
        self.assertIn(
            '[ -x "$llvm_bin/clang" ] || { echo "install_rocm.sh: no clang at $llvm_bin" >&2; exit 3; }',
            script,
        )


class TheRocmRecordTest(unittest.TestCase):
    def test_the_record_names_the_llvm_tools(self):
        """rocm_record: the record at a path, ROCM_RECORD by default, {} where there is none or it is no JSON object;
        rocm_llvm_bin: its `llvm_bin`, "" without one."""
        llvm = "/opt/rocm-9.9.0/lib/llvm/bin"
        with tempfile.TemporaryDirectory() as d:
            rec = pathlib.Path(d) / "rocm.json"
            self.assertEqual(
                (L.rocm_record(str(rec)), L.rocm_llvm_bin(str(rec))), ({}, "")
            )
            rec.write_text(json.dumps({"version": "9.9.0", "llvm_bin": llvm}))
            self.assertEqual(L.rocm_record(str(rec))["version"], "9.9.0")
            self.assertEqual(L.rocm_llvm_bin(str(rec)), llvm)
            with mock.patch.object(L, "ROCM_RECORD", str(rec)):
                self.assertEqual(L.rocm_llvm_bin(), llvm)
            for text in ("{", "[1]", json.dumps({"version": "9.9.0"})):
                with self.subTest(text=text):
                    rec.write_text(text)
                    self.assertEqual(L.rocm_llvm_bin(str(rec)), "")
            rec.write_text("[1]")
            self.assertEqual(L.rocm_record(str(rec)), {})

    def test_the_images_environment(self):
        """The monitor's home and its folders: the layout's; the write login's too, which the image names by no variable
        of its own: rk takes the layout's in the container, whatever the configuration names. Every folder it names is
        the image's code or in a volume."""
        from rkmon_lib import secrets_registry as sr

        # The final image's: the one that sets HOME.
        (final,) = [t for t in instructions("ENV") if t.startswith("HOME=")]
        env = dict(kv.split("=", 1) for kv in final.split())
        self.assertEqual(env["HOME"], f"/home/{L.MONITOR.name}")
        self.assertEqual(env["RKMON_HOME"], L.MONITOR_HOME)
        cfg = {"gh": {"write_config_dir": "/elsewhere"}}
        # No secret's place by a variable of the image's: those are the suite's (secrets_registry: `variable`).
        self.assertFalse([e.name for e in sr.entries(cfg) if e.variable in env])
        self.assertEqual(
            sr.path(cfg, sr.entry(cfg, "github_write"), env), L.WRITE_LOGIN_DIR
        )
        for k, v in env.items():
            if v.startswith("/") and k != "HOME":
                with self.subTest(var=k):
                    self.assertTrue(
                        v.startswith(L.IMAGE_CODE + "/")
                        or any(v.startswith(d + "/") for d in L.VOLUMES),
                        v,
                    )

    def test_the_counter_is_the_first_gh_on_the_images_path(self):
        """The GitHub request cap's counter, docker/rk_gh.py, is linked as gh in GH_COUNTER_DIR, which the image's
        PATH names first (the writer's too: ThePythonCodeReadsItTest)."""
        self.assertIn(
            f"ln -s {L.IMAGE_CODE}/docker/rk_gh.py {L.GH_COUNTER_DIR}/gh", DOCKERFILE
        )
        self.assertIn(f"PATH={L.GH_COUNTER_DIR}:$PATH", instructions("ENV"))

    def test_the_image_exposes_the_ports(self):
        (exposed,) = re.findall(r"(?m)^EXPOSE (.+)$", DOCKERFILE)
        self.assertEqual([int(p) for p in exposed.split()], [L.ADMIN_PORT, L.TEAM_PORT])

    def test_the_compose_file_publishes_the_admin_port_and_mounts_the_volumes(self):
        """The compose file publishes the admin dashboard's port; compose.override.yaml publishes the team
        dashboard's while the container serves it (rkmon_lib/deploy_plan.py). It mounts the secrets volume
        (SECRETS_VOLUME) at SECRETS_DIR, and gives it no name of its own: docker names it after the compose project.
        """
        inside = [
            int(p)
            for p in re.findall(
                r'(?m)^\s+- "\$\{RKMON_\w+:[-?][^}]*\}:\$\{RKMON_\w+:[-?][^}]*\}:(\d+)"$',
                COMPOSE,
            )
        ]
        self.assertEqual(inside, [L.ADMIN_PORT])
        self.assertEqual(
            tuple(
                re.findall(r"(?m)^\s+- [\w-]+:(/\S+)$", service(deploy_plan.SERVICE))
            ),
            (*L.VOLUMES, L.CODE_DIR, f"{L.SETTINGS_DIR}:ro"),
        )
        self.assertIn(f"- {L.CODE_VOLUME}:{L.CODE_DIR}\n", service(deploy_plan.SERVICE))
        self.assertRegex(COMPOSE, rf"(?m)^  {L.CODE_VOLUME}:")
        main = service(deploy_plan.SERVICE)
        self.assertIn(f"- {L.SECRETS_VOLUME}:{L.SECRETS_DIR}\n", main)
        self.assertRegex(COMPOSE, rf"(?m)^  {L.SECRETS_VOLUME}:\n(?!    )")

    def test_the_compose_file_mounts_the_settings_volume_read_only_and_writes_it_from_a_one_off(
        self,
    ):
        """The container mounts the settings volume read-only. Deliver's one-off service (deliver.LAYER_SERVICE),
        which `docker compose up` never starts, mounts it alone, where the program it runs writes (LAYER_WRITE_DIR),
        with no network and only its capabilities, on the container's image. The file declares every volume its
        services mount, so each deployment's compose project has its own."""
        main, one_off = service(deploy_plan.SERVICE), service(deliver.LAYER_SERVICE)
        self.assertIn(f"\n      - {L.SETTINGS_VOLUME}:{L.SETTINGS_DIR}:ro\n", main)
        self.assertEqual(
            re.findall(r"(?m)^      - (.+)$", one_off),
            [f"{L.SETTINGS_VOLUME}:{L.LAYER_WRITE_DIR}"],
        )
        self.assertEqual(deliver.LAYER_WRITE_DIR, L.LAYER_WRITE_DIR)
        self.assertEqual(
            re.findall(r"(?m)^    network_mode: (\S+)$", one_off), ["none"]
        )
        self.assertEqual(re.findall(r"(?m)^    cap_drop: \[(.*)\]$", one_off), ["ALL"])
        (added,) = re.findall(r"(?m)^    cap_add: \[(.*)\]$", one_off)
        self.assertEqual(tuple(added.split(", ")), L.LAYER_WRITE_CAPABILITIES)
        self.assertRegex(one_off, r"(?m)^    profiles: \[[\w-]+\]$")
        (anchor,) = re.findall(r"(?m)^    image: &(\w+) ", main)
        self.assertEqual(re.findall(r"(?m)^    image: (.+)$", one_off), [f"*{anchor}"])
        self.assertIn("\n    pull_policy: never\n", one_off)
        for key in ("ports", "environment", "restart"):
            self.assertNotRegex(one_off, rf"(?m)^    {key}:", key)
        (declared,) = re.findall(r"(?ms)^volumes:\n(.*)", COMPOSE)
        mounted = re.findall(r"(?m)^      - ([\w-]+):/", COMPOSE)
        self.assertEqual(
            re.findall(r"(?m)^  ([\w-]+):", declared), list(dict.fromkeys(mounted))
        )
        self.assertIn(L.SETTINGS_VOLUME, mounted)

    def test_the_compose_file_keeps_the_layouts_capabilities_only(self):
        """Every capability dropped but the layout's, and its cap on the processes; the image test's own containers
        get the same as docker run flags."""
        main = service(deploy_plan.SERVICE)
        self.assertEqual(re.findall(r"(?m)^    cap_drop: \[(.*)\]$", main), ["ALL"])
        (added,) = re.findall(r"(?m)^    cap_add: \[(.*)\]$", main)
        self.assertEqual(tuple(added.split(", ")), L.CAPABILITIES)
        self.assertEqual(
            re.findall(r"(?m)^    pids_limit: (\d+)$", main), [str(L.PIDS_LIMIT)]
        )
        self.assertNotIn("privileged", COMPOSE)
        flags = list(L.RUN_LIMITS)
        self.assertEqual(flags[:2], ["--cap-drop", "ALL"])
        self.assertEqual(flags[-2:], ["--pids-limit", str(L.PIDS_LIMIT)])
        self.assertEqual(tuple(flags[3:-2:2]), L.CAPABILITIES)
        self.assertEqual(set(flags[2:-2:2]), {"--cap-add"})

    def test_the_handed_layer_s_folder_is_the_one_off_container_s_own(self):
        """The folder of the layer handed to a one-off container of the logins' walk: in the container's own file
        system, apart from the volumes, the settings volume's folders, the code volume's, the RAM disk and the system's
        temporary folders. Neither the compose file nor the image names it, or the variable that hands it, so neither
        the running container nor a start's one-off container has either; the entry point takes the variable out of
        its environment first."""
        d = L.HANDED_LAYER_DIR
        self.assertTrue(d.startswith("/") and not d.endswith("/"), d)
        for other in (
            *L.VOLUMES,
            L.SETTINGS_DIR,
            L.LAYER_WRITE_DIR,
            L.CODE_DIR,
            L.RAMDISK,
            L.SHM_DIR,
            *L.SYSTEM_TMP,
        ):
            with self.subTest(other=other):
                self.assertFalse(
                    d == other or d.startswith(other + "/") or other.startswith(d + "/")
                )
        for text in (COMPOSE, DOCKERFILE):
            self.assertNotIn(L.HANDED_LAYER_VARIABLE, text)
            self.assertNotIn(d, text)
        self.assertIn(f"\nhanded=${{{L.HANDED_LAYER_VARIABLE}:-}}  ", ENTRY)
        self.assertIn(f"\nunset {L.HANDED_LAYER_VARIABLE}\n", ENTRY)

    def test_the_entry_point_names_the_layouts_users_and_folders(self):
        for part in (
            f"as=${{RKMON_AS:-{L.MONITOR.name}}}",
            f"{L.MONITOR.name}|{L.WRITER.name}|root) ;;",
            f"RKMON_AS is {L.MONITOR.name}, {L.WRITER.name} or root",
            f'[ "$(id -un)" != {L.WRITER.name} ]',
            f"WRITE_LOGIN={L.WRITE_LOGIN_DIR} ",
            f"${{RKMON_SECRETS_DIR:-{L.SECRETS_DIR}}}",
            f"LOGS=${{RKMON_LOGS:-{L.WORK_DIR}/{L.LOGS_FOLDER}}}",
            f"PKG={L.IMAGE_CODE} ",
        ):
            with self.subTest(part=part):
                self.assertIn(part, ENTRY)
        # The monitor's home is RKMON_HOME, which the image's environment names (test_the_images_environment):
        # the entry point has no copy of it.
        self.assertNotIn(L.MONITOR_HOME, ENTRY)

    def test_the_start_maps_the_same_users_and_volumes(self):
        ids = deployment.rk_ids()
        self.assertEqual(ids.OURS, tuple(u.name for u in L.USERS))
        self.assertEqual(ids.VOLUMES, L.VOLUMES)
        self.assertEqual(ids.RUNNER_IDS, {u.uid for u in L.PR_RUNNERS})


class TheSettingsAgreeTest(unittest.TestCase):
    """The settings' defaults that are the layout's values: the dashboards' ports (webserve's defaults), the team
    dashboard's host port (its port inside by default), the writer's id and the id the PR runner's key checks.
    """

    def test_the_configuration_defaults(self):
        """A port no configuration sets is the layout's (rkmon_lib/derived.py), and its description (the registry's)
        names it in its closing sentence."""
        unset = derived.Layout()
        for key, port in (
            ("admin_dashboard.port", L.ADMIN_PORT),
            ("team_dashboard.port", L.TEAM_PORT),
        ):
            self.assertEqual(derived.FUNCTIONS[key]({}, unset), port, key)
        docs = config.doc()
        for key, port, listens in (
            ("admin_dashboard.port", L.ADMIN_PORT, "the service listens on it inside"),
            ("team_dashboard.port", L.TEAM_PORT, "the page listens on it inside"),
        ):
            with self.subTest(key=key):
                self.assertIn(listens, docs[key])
                self.assertTrue(docs[key].endswith(f" Derived: {port}."), docs[key])

    def test_the_ramdisk_is_mounted_where_the_slots_are(self):
        """The RAM disk's default folder, the compose override's mount of it (rkmon_lib/deploy_plan.py) and the
        example's are RAMDISK, the folder that holds the slots."""
        self.assertEqual(
            derived.FUNCTIONS["ramdisk.path"]({}, derived.Layout()), L.RAMDISK
        )
        example = (ROOT / "docker" / "compose.local.example.yaml").read_text(
            encoding="utf-8"
        )
        self.assertRegex(example, rf"(?m)^\s+- {re.escape(L.RAMDISK)}:size=")
        cfg = {"ramdisk": {"enabled": True, "size_gb": 4}}
        mount = f"      - {L.RAMDISK}:size=4g,mode=1777,exec"
        self.assertEqual(deployment.resources(cfg)["ramdisk_path"], L.RAMDISK)
        override = deploy_plan_of({deploy_plan.RAMDISK_GIB: 4}).override
        self.assertIn(mount, override.splitlines())

    def test_the_deploy_settings_schema(self):
        """Layout 1's schema holds the writer's id and the PR runner's key; layout 2's sets neither: there the writer's
        and the PR runners' ids are the package's own."""

        def users_of(*where):
            schema = json.loads(
                (ROOT / "schema").joinpath(*where).read_text(encoding="utf-8")
            )
            return (
                schema,
                schema["properties"]["host"]["properties"]["users"]["properties"],
            )

        schema, users = users_of("layout1", "deploy_settings.schema.json")
        self.assertEqual(users["github_writer_uid"]["default"], L.WRITER.uid)
        self.assertEqual(users["pr_runner_uid"]["default"], L.PR_RUNNER.uid)
        dashboards = schema["properties"]["features"]["properties"]["dashboards"]
        team = dashboards["properties"]["team"]["properties"]
        self.assertEqual(team["port"]["default"], L.TEAM_PORT)
        _schema, users = users_of("deploy_settings.schema.json")
        self.assertEqual(
            [k for k in users if "writer" in k or "runner" in k or "pr_" in k], []
        )


def script_with_layout(script: str, base: pathlib.Path, extra: str):
    """docker/<script> loaded from a copy of the package in `base` whose container_layout.py ends with `extra` (the
    worker tools, code_volume.py, whole_file.py and the broker copied too: the scripts read worker/perms.py, the
    launcher worker/pr_env.py and pr_results.py, the root steps code_volume.py, whole_file.py and the broker's check).
    The files copied are read-only, as the image's package is: each is copied once.
    """
    (base / "docker").mkdir(parents=True)
    (base / "rkmon_lib").mkdir()
    docker = [f"docker/{name}" for name in dict.fromkeys((script, "rk_write.py"))]
    for rel in [*docker, "rkmon_lib/code_volume.py", "rkmon_lib/whole_file.py"]:
        shutil.copy2(ROOT / rel, base / rel)
        (base / rel).chmod((base / rel).stat().st_mode & 0o555)
    shutil.copytree(ROOT / "worker", base / "worker")
    text = (ROOT / "rkmon_lib" / "container_layout.py").read_text(encoding="utf-8")
    (base / "rkmon_lib" / "container_layout.py").write_text(text + extra)
    spec = importlib.util.spec_from_file_location(
        f"{script[:-3]}_copy", base / "docker" / script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The writer's run's values, in a fresh interpreter that imports it from the tree in its first argument.
WRITER_RUN = """
import json, sys
sys.path.insert(0, sys.argv[1])
from rkmon_lib import writer_run as w
print(json.dumps([w.WRITE_DIR, w.LOG, w.BASE_ENV["RKMON_HOME"], w.BASE_ENV["PATH"].split(":")[0]]))
"""


def writer_run_with_layout(base: pathlib.Path, extra: str) -> list:
    """rkmon_lib/writer_run.py's write login folder, log, monitor home and first PATH folder, as a fresh interpreter
    imports it from a copy of rkmon_lib in `base` whose container_layout.py ends with `extra` (the broker starts the
    writer's run from the tree it checked, `python3 -I -B`). The copy keeps the tree's modes, read-only where the tree
    is, as the image's code is: its container_layout.py is made writable for its end."""
    shutil.copytree(
        ROOT / "rkmon_lib",
        base / "rkmon_lib",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    layout = base / "rkmon_lib" / "container_layout.py"
    layout.chmod(layout.stat().st_mode | 0o200)
    layout.write_text(layout.read_text(encoding="utf-8") + extra, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-I", "-B", "-c", WRITER_RUN, str(base)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    return json.loads(out)


# Another layout: each script must follow it, so none keeps a value of its own.
OTHER_LAYOUT = """
WORK_DIR, SECRETS_DIR, WRITE_SECRETS_DIR = "/w9", "/s9", "/sw9"
WRITE_LOGIN_DIR, MONITOR_HOME = "/sw9/login9", "/w9/home9"
LOGS_FOLDER = "logs9"
CODE_DIR, IMAGE_ID_FILE = "/c9", "/e9/code_id9"
GH_COUNTER_DIR = "/g9"
MONITOR = User("m9", 2001, "m9", 2001)
WRITER = User("w9", 2002, "m9", 2001)
"""


class ThePythonCodeReadsItTest(unittest.TestCase):
    def test_the_scripts_the_image_runs_as_another_user_read_it(self):
        """The root steps and the write broker read the layout, which they load by its path from their own folder; the
        writer's run imports it from the tree it is in, the counter's folder first on its PATH.
        """
        with tempfile.TemporaryDirectory() as d:
            root = script_with_layout("rk_root.py", pathlib.Path(d, "a"), OTHER_LAYOUT)
            volume = root.Volume()
            write = script_with_layout(
                "rk_write.py", pathlib.Path(d, "b"), OTHER_LAYOUT
            )
            run = writer_run_with_layout(pathlib.Path(d, "c"), OTHER_LAYOUT)
        self.assertEqual(
            (root.WRITE_USER, root.WRITE_VOLUME, root.NEW, root.OLD),
            ("w9", "/sw9", "/sw9/login9", "/s9/login9"),
        )
        self.assertEqual((volume.folder, volume.id_file), ("/c9", "/e9/code_id9"))
        self.assertEqual(
            (write.WRITE_USER, write.LOG, write.CODE_DIR, write.IMAGE_ID_FILE),
            ("w9", "/w9/logs9/rk-write.log", "/c9", "/e9/code_id9"),
        )
        self.assertEqual(
            run, ["/sw9/login9", "/w9/logs9/rk-write.log", "/w9/home9", "/g9"]
        )

    def test_the_readers(self):
        self.assertEqual(
            (webserve.INNER_PORT, webserve.TEAM_INNER_PORT), (L.ADMIN_PORT, L.TEAM_PORT)
        )
        # The slot commands: the PR runners' group for the trees PR code reads, and a slot's folders.
        self.assertEqual(prcode.tree_command("/t")[-3:], ["--group", L.PR_GROUP, "/t"])
        self.assertIn(
            "mkdir " + " ".join((*L.SLOT_FOLDERS, prcode.INPUTS)), prcode.SHELL
        )
        self.assertEqual(
            pr_user.values(),
            {"group": L.PR_GROUP, "home": L.SLOT_HOME, "rk": str(ROOT / "rk")},
        )
        self.assertEqual(pr_user.group(), L.PR_GROUP)

    # The code files whose own strings hold the loopback address, each with why.
    LOOPBACK_HELD = {
        "rkmon_lib/container_layout.py": "its one name, LOOPBACK",
        "rkmon_lib/from_layout1.py": "migration code: layout 1's values, frozen",
        "tools/webhost_remote.py": "it runs on the team web host, standard library only, without the package",
    }

    def test_the_loopback_has_one_name(self):
        """The loopback address, which this machine alone reaches, has one name, LOOPBACK, and the names a URL gives
        this machine (LOCAL_HOSTS) are built on it: no string of the code holds the address, docstrings apart, but in
        the files LOOPBACK_HELD names, each of which does. So the plan and docker/.env's values publish the admin
        dashboard on it by that name, and every message, help line and default names it so.
        """
        held = {}
        for f in code_files():
            for line, v in literals(f):
                if isinstance(v, str) and L.LOOPBACK in v:
                    held.setdefault(str(f.relative_to(ROOT)), []).append(line)
        self.assertEqual(sorted(held), sorted(self.LOOPBACK_HELD), held)
        self.assertEqual(L.LOCAL_HOSTS[0], L.LOOPBACK)

    def test_the_dashboards_listen_where_the_layout_says(self):
        """In the image both dashboards listen on LISTEN_ADDRESS, read each time they start: another address there
        moves both. On a host the configuration says where."""
        image = {"RKMON_CONTAINER": "1"}
        cfg = {
            "admin_dashboard": {"bind": "127.0.0.1"},
            "team_dashboard": {"bind": "127.0.0.1", "port": 8080},
        }
        for address in (L.LISTEN_ADDRESS, "192.0.2.7"):
            with self.subTest(address=address), mock.patch.object(
                L, "LISTEN_ADDRESS", address
            ):
                self.assertEqual(webserve.where(cfg, environ=image)[0], address)
                self.assertEqual(
                    webserve.team_where(cfg, environ=image), (address, L.TEAM_PORT)
                )
        self.assertEqual(webserve.where(cfg, environ={})[0], "127.0.0.1")
        self.assertEqual(webserve.team_where(cfg, environ={}), ("127.0.0.1", 8080))

    def test_an_installed_copy_of_the_worker_tools_runs_no_file_it_finds(self):
        """An installed copy (deploy.install_worker_tools) takes the PR runners' group, a slot's home and rk from the
        values its install wrote beside it and from nothing else: a container_layout.py planted in the folder above it
        (which worker/drafter.py puts first on the import path) or in the package of the image's launcher never runs,
        and without the values it refuses. Telling a launched command line loads nothing. The package's own worker/
        imports its package's layout."""
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            cfg = {"paths": {"working_dir": str(base / "work")}}
            deploy.install_worker_tools(cfg, package=ROOT)
            tools = pathlib.Path(paths.get(cfg, "worker_tools"))
            launcher = base / "image" / "docker" / "rk_prcode.py"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("")
            tops = (tools.parent, launcher.parent.parent)
            for top in tops:
                (top / "rkmon_lib").mkdir()
                (top / "rkmon_lib" / "__init__.py").write_text("")
                (top / "rkmon_lib" / "container_layout.py").write_text(PLANTED)

            def probe(tools_dir: pathlib.Path, top: pathlib.Path):
                return subprocess.run(
                    [sys.executable, "-I", "-B", "-c", PROBE]
                    + [str(tools_dir), str(top), str(launcher)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            values = {"group": L.PR_GROUP, "home": L.SLOT_HOME, "rk": str(ROOT / "rk")}
            p = probe(tools, tools.parent)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = json.loads(p.stdout)
            self.assertEqual(
                out["argv"],
                [str(ROOT / "rk"), "pr-run", "--env", "/dev/stdin", "-C", L.SLOT_HOME]
                + ["--", "a"],
            )
            self.assertEqual(out["data"], "X=1\0")
            self.assertEqual(out["launched"], [True, False, False, False])
            # Without the launcher (a host) nothing needs the values: the command runs through the copy's pr_env.py.
            self.assertEqual(
                out["host"][0][1:], ["-I", str((tools / "pr_env.py").resolve()), "a"]
            )
            self.assertIsNone(out["host"][1])
            self.assertEqual((out["values"], out["layouts"]), (values, []))
            (tools / pr_user.VALUES_FILE).unlink()
            p = probe(tools, tools.parent)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("does not name the PR runners' group", p.stderr)
            p = probe(ROOT / "worker", ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = json.loads(p.stdout)
            self.assertEqual(
                (out["values"], out["layouts"]),
                (values, ["rkmon_lib.container_layout"]),
            )
            for top in tops:
                self.assertFalse((top / "rkmon_lib" / "RAN").exists(), top)


def code_files() -> list:
    """The Python files of rkmon_lib/, tools/, worker/, rkmon.py and docker/rkmon."""
    return [ROOT / "rkmon.py", ROOT / "docker" / "rkmon"] + [
        f
        for top in ("rkmon_lib", "tools", "worker")
        for f in sorted((ROOT / top).rglob("*.py"))
    ]


def literals(path: pathlib.Path):
    """(line, value) of each constant in the code of `path`, docstrings left out."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in docs:
            yield node.lineno, node.value


def _held(values, before: str, after: str) -> re.Pattern:
    """Any of the values as a whole word of a text, not part of a longer one: not after BEFORE, not before AFTER."""
    return re.compile(
        "|".join(
            f"(?<!{before}){re.escape(str(v))}(?!{after})"
            for v in dict.fromkeys(values)
        )
    )


# What a folder held is not after, and not before: it is no part of a longer path or name.
FOLDER_EDGES = (r"[\w.~/-]", r"[\w.-]")
# The layout's values as a text may hold them: the users' names and the PR runners' group, the ports, every address,
# the volumes' folders, the RAM disk's, the settings volume's two, the handed layer's, the image's own code, its checks
# venv, its ROCm record and the counter's folder.
HELD = (
    _held(
        (L.WRITER.name, L.PR_GROUP, *(u.name for u in L.PR_RUNNERS)),
        r"[\w.-]",
        r"[\w-]",
    ),
    _held((L.ADMIN_PORT, L.TEAM_PORT), r"\d", r"\d"),
    _held((L.EVERY_ADDRESS, L.LISTEN_ADDRESS), r"[\d.]", r"[\d.]"),
    _held(
        (
            *L.VOLUMES,
            L.RAMDISK,
            L.SETTINGS_DIR,
            L.LAYER_WRITE_DIR,
            L.HANDED_LAYER_DIR,
            L.IMAGE_CODE,
            L.IMAGE_VENV,
            L.ROCM_RECORD,
            L.GH_COUNTER_DIR,
        ),
        *FOLDER_EDGES,
    ),
)
# The layout's values the image's test (tests/container/) takes from the layout too: the image's ROCm record.
IMAGE_TEST_HELD = _held((L.ROCM_RECORD,), *FOLDER_EDGES)


def holds_the_layout(text: str) -> bool:
    return any(p.search(text) for p in HELD)


class OneHomeTest(unittest.TestCase):
    def test_a_text_that_holds_a_value_is_found(self):
        for text in (
            "RKMON_AS=" + L.WRITER.name,
            f"sudo -n -u {L.PR_RUNNER.name} x",
            f"sudo -n -u {L.PR_RUNNERS[-1].name} x",
            f"chgrp {L.PR_GROUP} f",
            f"{L.WORK_DIR}/monitor/config.json",
            f"--volume v:{L.WRITE_SECRETS_DIR}",
            f"{L.RAMDISK}/slots",
            f"http://{L.LISTEN_ADDRESS}:{L.TEAM_PORT}/",
            f"127.0.0.1:{L.ADMIN_PORT}",
            f"{L.SETTINGS_DIR}/{L.LAYER_FILE}",
            f"-v rkmon-settings:{L.LAYER_WRITE_DIR}",
            f"{L.IMAGE_VENV}/bin/python",
            f"open('{L.ROCM_RECORD}')",
            f"{L.IMAGE_CODE}/docker/entrypoint.sh",
        ):
            self.assertTrue(holds_the_layout(text), text)
        for text in (
            f"~{L.WORK_DIR}",
            f"data{L.WORK_DIR}",
            f"{L.WORK_DIR}ers",
            f"rkmon_lib{L.RAMDISK}.py",
            f"{L.PR_RUNNER.name}obe",
            f"1{L.ADMIN_PORT}",
            f"1{L.EVERY_ADDRESS}",
            f"{L.IMAGE_VENV}-claude/bin/python",
            f"{L.IMAGE_CODE}-1.2/docker/rkmon",
        ):
            self.assertFalse(holds_the_layout(text), text)

    def test_no_python_file_keeps_a_copy(self):
        """The folders, the image's own code among them, the writer's and the PR runners' names, their group's, their
        ids, the dashboards' ports and every address appear in no other Python file of rkmon_lib/, tools/, worker/,
        rkmon.py or docker/rkmon, not even inside a text; a docstring may name them. (The monitor's name, rkmon, is
        the wrapper's and the default project's too: it is left out.)"""
        numbers = {L.WRITER.uid, L.PR_GID, L.ADMIN_PORT, L.TEAM_PORT}
        numbers |= {u.uid for u in L.PR_RUNNERS} | {u.gid for u in L.PR_RUNNERS}
        found = []
        for f in code_files():
            if f.name == "container_layout.py":
                continue
            for line, v in literals(f):
                if (isinstance(v, str) and holds_the_layout(v)) or (
                    isinstance(v, int) and not isinstance(v, bool) and v in numbers
                ):
                    found.append(f"{f.relative_to(ROOT)}:{line}: {v!r}")
        self.assertEqual(found, [])

    def test_the_image_s_test_takes_the_rocm_record_from_the_layout(self):
        """The image's ROCm record's path appears in no Python file of tests/container/ either, not even inside a text:
        the hip and rocm stages take it from the layout; a docstring may name it."""
        found = [
            f"{f.relative_to(ROOT)}:{line}: {v!r}"
            for f in sorted((ROOT / "tests" / "container").rglob("*.py"))
            for line, v in literals(f)
            if isinstance(v, str) and IMAGE_TEST_HELD.search(v)
        ]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
