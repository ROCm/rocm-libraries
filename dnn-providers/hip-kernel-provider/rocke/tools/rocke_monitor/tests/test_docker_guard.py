# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deploy-path stage's docker guard (tests/fakes/docker_guard), offline: each verdict (pass, narrowed, placed,
refused), what keeps every image on this host (no pull, push or build, by docker or by compose, and no --pull but
never), prune refused, and every object of another deployment on the same host refused: its container, its compose
project, its volumes and its images. No engine is asked: the guard's lists of the scope's container and image IDs are
given.
"""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True


def load_guard():
    """tests/fakes/docker_guard, a script with no .py, as a module."""
    path = ROOT / "tests" / "fakes" / "docker_guard"
    loader = importlib.machinery.SourceFileLoader("docker_guard", str(path))
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("docker_guard", loader)
    )
    loader.exec_module(module)
    return module


G = load_guard()

SCOPE = "l2test-unit-deploy-path"
IMAGE = "rocke-monitor:l2test-unit"
NETWORK = f"{SCOPE}_default"
CONTAINER = f"{SCOPE}-rkmon-1"
VOLUME = f"{SCOPE}_rkmon-work"
OUR_ID = "a" * 64  # the ID of the scope's container
OUR_IMAGE_ID = "sha256:" + "b" * 64
# Another deployment on the same host, under the names an older monitor's compose project gives its objects.
OTHER = {
    "container": "rkmon-rkmon-1",
    "project": "rkmon",
    "volumes": (
        "rkmon_rkmon-work",
        "rkmon_rkmon-secrets",
        "rkmon_rkmon-secrets-write",
        "rocke_pr_monitor_secrets",
        "ollama",
    ),
    "images": ("rocke-pr-monitor:latest", "rocke-pr-monitor:2026.10.1"),
    "id": "c" * 64,
}


class Base(unittest.TestCase):
    @staticmethod
    def make(log: str = ""):
        guard = G.Guard(
            {
                "engine": "/nonexistent/docker",
                "scope": SCOPE,
                "images": [IMAGE],
                "network": NETWORK,
                "log": log,
            }
        )
        # No engine to ask: the scope's containers and images by ID.
        guard._ids = [OUR_ID]
        guard._image_ids = [OUR_IMAGE_ID]
        return guard

    def check(self, *args: str, environ=None) -> tuple:
        return self.make().check(list(args), environ or {})

    def refused(self, *args: str, environ=None, why: str = "") -> None:
        with self.assertRaises(G.Refused, msg=args) as caught:
            self.check(*args, environ=environ)
        self.assertIn(why, str(caught.exception), args)


class TestVerdicts(Base):
    def test_the_engine_s_questions_and_the_scope_s_objects_pass_as_they_are(self):
        for args in (
            ("version",),
            ("info",),
            ("buildx", "version"),
            ("compose", "version"),
            ("exec", CONTAINER, "rk", "status"),
            ("exec", OUR_ID[:12], "true"),  # a container of the scope's by its ID
            ("logs", "--tail", "5", CONTAINER),
            ("inspect", "--type", "volume", VOLUME),
            ("volume", "inspect", VOLUME),
            ("network", "inspect", NETWORK),
            ("image", "inspect", IMAGE),
            ("rm", "-f", CONTAINER),
            ("run", "--rm", "--network", "none", IMAGE, "true"),
            ("run", "--rm", "--pull", "never", "--network", NETWORK, IMAGE, "true"),
            ("create", "--pull=never", "--network", "none", IMAGE),
            ("compose", "-p", SCOPE, "ps"),
            ("compose", "-p", SCOPE, "up", "-d", "--pull", "never"),
            ("compose", "-p", SCOPE, "down"),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.check(*args), (list(args), "pass"))

    def test_a_listing_is_narrowed_to_the_scope_by_an_anchored_name(self):
        anchored = ["--filter", f"name=^{SCOPE}"]
        label = f"label=com.docker.compose.project={SCOPE}"
        for args, passed in (
            (("ps", "-a"), ["ps", "-a", *anchored]),
            (("container", "ls", "--all"), ["ps", "--all", *anchored]),
            (("volume", "ls"), ["volume", "ls", *anchored]),
            (("network", "ls"), ["network", "ls", *anchored]),
            (("compose", "ls"), ["compose", "ls", *anchored]),
            (("images",), ["images", "--filter", f"reference={IMAGE}"]),
            (("image", "ls"), ["image", "ls", "--filter", f"reference={IMAGE}"]),
            # a filter on another key, which docker joins to the guard's with and
            (
                ("volume", "ls", "--filter", label),
                ["volume", "ls", "--filter", label, *anchored],
            ),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.check(*args), (passed, "narrowed"))

    def test_a_listing_filtered_on_an_exact_name_of_the_scope_passes_as_it_is(self):
        for value in (f"^{VOLUME}$", f"^/{CONTAINER}$", f"^{SCOPE}", f"^{SCOPE}\\.x$"):
            with self.subTest(value=value):
                args = ("volume", "ls", "--filter", f"name={value}")
                self.assertEqual(self.check(*args), (list(args), "narrowed"))

    def test_a_one_off_container_that_names_no_network_is_placed_on_the_stage_s(
        self,
    ):
        self.assertEqual(
            self.check("run", "--rm", IMAGE, "true"),
            (["run", "--network", NETWORK, "--rm", IMAGE, "true"], "placed"),
        )


class TestImagesStayOnThisHost(Base):
    def test_no_pull_push_or_build_by_docker(self):
        for args in (
            ("pull", IMAGE),
            ("push", IMAGE),
            ("build", "-t", IMAGE, "."),
            ("buildx", "build", "-t", IMAGE, "."),
            ("login", "registry.example.com"),
            ("tag", IMAGE, f"{IMAGE}-copy"),
            ("image", "pull", IMAGE),
            ("image", "push", IMAGE),
            ("image", "tag", IMAGE, f"{IMAGE}-copy"),
            ("image", "build", "."),
        ):
            with self.subTest(args=args):
                self.refused(*args)

    def test_no_pull_push_or_build_by_compose_even_in_the_scope(self):
        for sub in ("pull", "push", "build"):
            with self.subTest(sub=sub):
                self.refused("compose", "-p", SCOPE, sub, why=f"compose {sub}")
        for words in (
            ("up", "-d", "--build"),
            ("up", "-d", "--pull", "always"),
            ("up", "-d", "--pull=missing"),
            ("create", "--pull", "build"),
            ("run", "--rm", "--build", "rkmon", "true"),
            ("up", "-d", "--pull"),  # no value: none to read
        ):
            with self.subTest(words=words):
                self.refused("compose", "-p", SCOPE, *words)

    def test_no_pull_but_never_on_run_or_create(self):
        for args in (
            ("run", "--rm", "--pull", "always", "--network", "none", IMAGE, "true"),
            ("run", "--rm", "--pull=missing", IMAGE, "true"),
            ("create", "--pull", "always", "--network", "none", IMAGE),
        ):
            with self.subTest(args=args):
                self.refused(*args, why="--pull")

    def test_no_prune(self):
        for args in (
            ("system", "prune", "-f"),
            ("volume", "prune", "-f"),
            ("image", "prune", "-a", "-f"),
            ("builder", "prune", "-f"),
            ("container", "prune", "-f"),
            ("network", "prune", "-f"),
            ("buildx", "prune", "-f"),
        ):
            with self.subTest(args=args):
                self.refused(*args)


class TestAnotherDeployment(Base):
    """Another deployment on the same host: none of its objects is acted on or read but in a narrowed listing."""

    def test_its_container(self):
        name = OTHER["container"]
        for args in (
            ("exec", name, "true"),
            ("exec", OTHER["id"][:12], "true"),  # by its ID
            ("logs", name),
            ("inspect", name),
            ("inspect", "--type", "container", name),
            ("cp", f"{name}:/work/monitor/config.json", "/tmp/x"),
            ("stop", name),
            ("start", name),
            ("restart", name),
            ("kill", name),
            ("rm", "-f", name),
            ("container", "rm", "-f", name),
            ("top", name),
            ("network", "connect", NETWORK, name),
            ("ps", "-a", "--filter", f"name=^{name}$"),
        ):
            with self.subTest(args=args):
                self.refused(*args)

    def test_its_compose_project_however_compose_finds_it(self):
        project = OTHER["project"]
        for args, environ in (
            (("compose", "-p", project, "down"), {}),
            (("compose", "--project-name", project, "ps"), {}),
            (("compose", "ps"), {"COMPOSE_PROJECT_NAME": project}),
            (("compose", "-p", project, "exec", "rkmon", "rk", "status"), {}),
            (("compose", "-p", project, "logs"), {}),
        ):
            with self.subTest(args=args):
                self.refused(*args, environ=environ, why=repr(project))
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "compose.yaml").write_text(
                f"name: {project}\nservices: {{}}\n", encoding="utf-8"
            )
            self.refused(
                "compose", "--project-directory", str(folder), "down", why=repr(project)
            )
            self.refused("compose", "-f", str(folder / "compose.yaml"), "stop")
            (folder / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            (folder / ".env").write_text(
                f"COMPOSE_PROJECT_NAME={project}\n", encoding="utf-8"
            )
            self.refused("compose", "--project-directory", str(folder), "down")

    def test_its_volumes(self):
        for volume in OTHER["volumes"]:
            for args in (
                ("volume", "rm", volume),
                ("volume", "inspect", volume),
                ("inspect", "--type", "volume", volume),
                ("inspect", volume),
                ("run", "--rm", "--network", "none", "-v", f"{volume}:/data", IMAGE),
                (
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--mount",
                    f"type=volume,src={volume},dst=/data",
                    IMAGE,
                ),
                ("volume", "ls", "--filter", f"name=^{volume}$"),
                ("volume", "ls", "--filter", f"name={volume}"),
            ):
                with self.subTest(args=args):
                    self.refused(*args)

    def test_its_images(self):
        for image in OTHER["images"]:
            for args in (
                ("run", "--rm", "--network", "none", image, "true"),
                ("create", "--network", "none", image),
                ("rmi", image),
                ("image", "rm", image),
                ("image", "inspect", image),
                ("inspect", "--type", "image", image),
                ("images", "--filter", f"reference={image}"),
            ):
                with self.subTest(args=args):
                    self.refused(*args)

    def test_a_name_filter_that_may_match_a_part_of_its_names(self):
        """docker reads a name filter as a pattern that matches any part of a name: one not anchored at the
        name's start, or with a dot left as a pattern's, may list another deployment's names.
        """
        for value in (SCOPE, VOLUME, f"^{SCOPE}.x", f".*{SCOPE}", f"^rkmon|^{SCOPE}"):
            with self.subTest(value=value):
                self.refused("volume", "ls", "--filter", f"name={value}")


class TestAnchors(unittest.TestCase):
    def test_a_name_s_anchored_filter_and_the_name_it_reads_back(self):
        self.assertEqual(G.anchored("a.b-c_d"), "^a\\.b-c_d")
        for value, name in (
            ("^a\\.b-c_d", "a.b-c_d"),
            ("^a\\.b-c_d$", "a.b-c_d"),
            ("^/a-b$", "a-b"),
            ("a-b", ""),
            ("^a.b", ""),
            ("^a|b", ""),
            ("^", ""),
        ):
            with self.subTest(value=value):
                self.assertEqual(G.anchored_name(value), name)


class TestMain(unittest.TestCase):
    def test_a_refused_command_exits_125_is_logged_and_never_reaches_the_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            log = folder / "calls.jsonl"
            config = folder / "guard.json"
            config.write_text(
                json.dumps(
                    {
                        "engine": str(folder / "no-engine"),
                        "scope": SCOPE,
                        "images": [IMAGE],
                        "network": NETWORK,
                        "log": str(log),
                    }
                ),
                encoding="utf-8",
            )
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = G.main([str(config), "volume", "rm", OTHER["volumes"][0]])
            self.assertEqual(code, G.REFUSED)
            self.assertIn("docker guard: refused: a volume outside", err.getvalue())
            (line,) = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual(line["verdict"], "refused")
            self.assertEqual(line["argv"], ["volume", "rm", OTHER["volumes"][0]])
            self.assertEqual(line["command"], "volume rm")
            self.assertEqual(line["names"], [["volume", OTHER["volumes"][0]]])

    def test_each_line_names_the_command_the_guard_read(self):
        calls = (
            (("ps", "-a"), "ps"),
            (("container", "ls", "--all"), "ps"),
            (("volume", "ls"), "volume ls"),
            (("exec", CONTAINER, "true"), "exec"),
            (
                ("compose", "-p", SCOPE, "--progress", "plain", "config"),
                "compose config",
            ),
            (
                ("compose", "-p", SCOPE, "exec", "rkmon", "ls", "/secrets"),
                "compose exec",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "calls.jsonl"
            for args, _command in calls:
                guard = Base.make(str(log))
                guard.log(list(args), *guard.check(list(args), {}))
            lines = [json.loads(x) for x in log.read_text().splitlines()]
        self.assertEqual([x["command"] for x in lines], [c for _a, c in calls])


if __name__ == "__main__":
    unittest.main()
