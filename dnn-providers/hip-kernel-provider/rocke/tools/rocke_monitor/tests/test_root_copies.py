# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one-time move's copies, which root makes at each start (docker/rk_root.py `copies`): its table of
the secrets registry's facts, held equal to rkmon_lib/secrets_registry.py, their one home, and its reading of the
bindings' keys, held equal to the registry's need of them over the same layers; each copy, its owner and its
mode, and none when the new copy exists; the claude login's key alone, read by the one reader of the older Claude
settings' shape, and nothing from a shape the settings cannot carry; a model binding's key while a pass of review
automation runs on its binding, else a line that says why not; the atlassian login and the GPU host's key never
copied; an older copy that is not a plain file, or two that differ, not copied; each copy made once, by its mark in
the code volume, so that a file written later at an older name is never copied, and no mark for a copy left for a
later start; the start's exit code when a copy cannot be made; and no value in any line.
"""

import contextlib
import errno
import importlib.util
import io
import json
import os
import pathlib
import shutil
import socket
import stat
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import deploy_folder  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    confirm,
    container_layout,
    deploy_plan,
    deploy_tree,
    file_types,
    image,
    runtime_config,
    secrets_registry as sr,
    settings_file,
)


def _script():
    """docker/rk_root.py, loaded as a module of its own, as the image runs it."""
    spec = importlib.util.spec_from_file_location(
        "rk_root_of_the_copies_tests", ROOT / "docker" / "rk_root.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rk_root = _script()
# Invented values, made as the tests run, so that no file of the package holds one.
KEY = "test-" + "key-not-real"
OTHER_KEY = "test-" + "other-key-not-real"
HOOK = "https://hooks.example.com/" + "hook-not-real"
SSH_KEY = (
    "-----BEGIN"
    + " OPENSSH PRIVATE KEY-----\nssh-not-real\n-----END"
    + " OPENSSH PRIVATE KEY-----\n"
)
VALUES = (KEY, OTHER_KEY, HOOK, "ssh-not-real", "sub-not-real")
# The keepers' users, as the stand-in for root's chown records them.
OWNERS = {"monitor": (4242, 4343), "writer": (1999, 4343)}
GLM = {
    "provider": "openai_compatible",
    "base_url": "https://models.example.com/v1",
    "model_id": "GLM-5.3-Flash",
    "api_key_secret_name": "glm_api_key",
}
STAYS = "the older copy stays until the move ends"
# What a start says of a copy an earlier start dealt with, while its new copy is missing: the older copy, the new copy
# and the command that sets it.
AGAIN = (
    "the one-time move did not copy {} to {}: the move makes each copy once, and an earlier start dealt with this one; "
    "the fix: rkmon {}, on the host"
)


def layer(path: pathlib.Path, features=None, bindings=None) -> dict:
    """The deploy layer that deploy settings with FEATURES and BINDINGS make, as the first start delivers it, written at
    PATH."""
    settings = path.parent / "deploy.yaml"
    deploy_folder.write_settings(settings, 8765, features=features, bindings=bindings)
    doc = settings_file.read(settings, file_types.DEPLOY)
    made = deploy_plan.plan(
        doc,
        image.package_version(),
        image.code_sha256(),
        deploy_plan.Host(os.getuid(), os.getgid(), "2026-01-02T03:04:05Z"),
    )
    path.write_text(deploy_plan.layer_text(made.layer), encoding="utf-8")
    return made.layer


def first_pass_on(binding: str) -> dict:
    """Review automation's features with its first review pass on the model binding BINDING."""
    return {"review_automation": {"passes": {"first_review_pass": {"model": binding}}}}


class TableTest(unittest.TestCase):
    """rk_root's table is the secrets registry's facts, as the container's paths name them."""

    # with the places the load derives
    CFG = config.normalize({"paths": {"secrets_dir": container_layout.SECRETS_DIR}})
    ENV = {"RKMON_CONTAINER": "1"}

    @staticmethod
    def place(keeper: str, name: str) -> str:
        return os.path.join(rk_root.KEEPERS[keeper][0], name)

    def test_the_keepers_folders_and_users(self):
        L = container_layout
        self.assertEqual(
            rk_root.KEEPERS,
            {
                sr.MONITOR: (L.SECRETS_DIR, L.MONITOR.name),
                sr.WRITER: (L.WRITE_SECRETS_DIR, L.WRITER.name),
            },
        )
        self.assertEqual(sorted(rk_root.WHOSE), sorted(sr.KEEPERS))
        self.assertEqual(rk_root.LAYER, os.path.join(L.SETTINGS_DIR, L.LAYER_FILE))

    def test_the_copies_are_the_registry_s(self):
        """Each older copy the registry says the move copies, from where to where, with the command that sets it: the
        write login's is move_write_login's, the claude login's CLAUDE, and the others COPIES, each a secret kept in a
        file of its own."""
        found, kinds = [], {}
        for e in sr.entries({}):
            for o, older in sr.older_places(self.CFG, e, self.ENV):
                if o.copied:
                    new = sr.home(self.CFG, e, self.ENV)
                    found.append((older, new, " ".join(sr.set_command(e))))
                    kinds[new] = (e.shape == sr.LOGIN, sr.held_files(e))
        ours = [
            (self.place(o, old), self.place(k, n), command)
            for o, old, k, n, command in (*rk_root.COPIES, rk_root.CLAUDE)
        ]
        self.assertEqual(
            sorted(found),
            sorted([*ours, (rk_root.OLD, rk_root.NEW, "login github_write")]),
        )
        claude = sr.entry({}, sr.CLAUDE_LOGIN)
        self.assertEqual(
            ours[-1][1:], (sr.home(self.CFG, claude, self.ENV), "login claude")
        )
        for _old, new, _command in ours[:-1]:
            self.assertEqual(kinds[new], (False, None), new)
        self.assertEqual(kinds[ours[-1][1]], (True, None))

    def test_the_names_of_a_binding_s_key_are_the_registry_s(self):
        """A binding's key's name, the names it is copied from, and the names the fixed entries keep."""
        self.assertEqual(rk_root.KEY_SUFFIX, sr.BINDING_KEY_SUFFIX)
        self.assertEqual(rk_root.OLDER_KEY_NAMES, sr.OLDER_KEY_NAMES)
        self.assertEqual(rk_root.SECRET_NAME.pattern, sr.NAME.pattern)
        fixed = sr.entries({})
        self.assertEqual(
            rk_root.FIXED_NAMES,
            {e.name for e in fixed}
            | {e.place for e in fixed if e.place}
            | {o.name for e in fixed for o in e.older},
        )
        for name in (
            "worker_api_key",
            "glm_key",
            "glm",
            "glm_api_key",
            "gw_token",
            "Not-A-Name",
            "",
        ):
            with self.subTest(name=name):
                self.assertEqual(rk_root._key_name(name), sr.binding_key_name(name))
        for keys in (
            [sr.FIRST_PASS_KEY],
            ["glm_api_key"],
            ["claude_api_key"],
            ["gpu_ssh_api_key"],
            ["teams_webhook_api_key"],
            ["x_api_key", "x_api_key_api_key"],
        ):
            cfg = {
                "model_bindings": {
                    f"b{n}": {"api_key_secret": k} for n, k in enumerate(keys)
                }
            }
            entries = {e.name: e for e in sr.entries(cfg)}
            for k in keys:
                with self.subTest(key=k):
                    e = entries[k]
                    self.assertEqual(
                        rk_root.older_key_names(k, rk_root.FIXED_NAMES | set(keys)),
                        [o.name for o in e.older],
                    )
                    self.assertEqual(
                        {(o.keeper, o.copied, o.file) for o in e.older}
                        | {(e.keeper, True, False)},
                        {(sr.MONITOR, True, False)},
                    )
                    self.assertEqual(sr.set_command(e), ("secret", k))

    def test_the_passes_are_deploy_tree_s(self):
        """Each pass's model key in a delivered layer: those rkmon_lib/deploy_tree.py finds in the schema."""
        with tempfile.TemporaryDirectory() as d:
            made = layer(pathlib.Path(d) / "deploy.json")
        self.assertEqual(
            sorted(rk_root.pass_models(made["settings"])),
            sorted(path for path, _node in deploy_tree._model_keys()),
        )

    def test_a_binding_s_key_is_used_as_the_registry_needs_it(self):
        """Over the same delivered layers, root's reading of the bindings' keys (layer_keys) is the registry's: the
        keys the bindings name are its entries, and root copies a key exactly when something that is on needs it
        (secrets_registry.needed, confirm.logins_needed): while review automation is on and a pass runs on a binding
        that names it. With review automation on, the login walk lists those keys alone.
        """
        drafter = {**GLM, "api_key_secret_name": sr.FIRST_PASS_KEY}
        worker = {**GLM, "api_key_secret_name": "local_api_key"}

        def passes(**models) -> dict:
            return {
                "review_automation": {
                    "passes": {p: {"model": m} for p, m in models.items()}
                }
            }

        off = passes(first_review_pass="drafter")
        off["review_automation"]["enabled"] = False
        for case, features, bindings in (
            (
                "the first review pass on the binding",
                first_pass_on("glm"),
                {"glm": GLM},
            ),
            ("no pass on the binding", None, {"drafter": drafter}),
            ("review automation off", off, {"drafter": drafter}),
            ("the local worker drafts", first_pass_on("worker"), {"worker": worker}),
            ("no pass on the local worker", None, {"worker": worker}),
            ("CI triage on the binding", passes(ci_triage_pass="glm"), {"glm": GLM}),
            (
                "the first code quality pass on the binding",
                passes(first_code_quality_pass="glm"),
                {"glm": GLM},
            ),
            (
                "the notices on the binding",
                passes(notice_draft_pass="glm"),
                {"glm": GLM},
            ),
            ("two bindings name one key", first_pass_on("a"), {"a": GLM, "b": GLM}),
            ("two keys", first_pass_on("glm"), {"glm": GLM, "drafter": drafter}),
            ("no binding", None, {}),
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                path = pathlib.Path(d) / "deploy.json"
                made = layer(path, features, bindings)
                named, on, why = rk_root.layer_keys(str(path))
                self.assertEqual(why, "")
                values = runtime_config.from_settings(made["settings"]).values
                fixed = {e.name for e in sr.entries({})}
                keys = {e.name for e in sr.entries(values)} - fixed
                self.assertEqual(set(named), keys)
                used = {k for k, u in named.items() if u}
                self.assertEqual({e.name for e in sr.needed(values)} - fixed, used)
                self.assertEqual(set(confirm.logins_needed(values)) - fixed, used)
                if on:
                    walked = {e.name for e in sr.walk_entries(values)} - fixed
                    self.assertEqual(walked, used)
                else:
                    self.assertEqual(used, set())


class CopiesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = pathlib.Path(tmp.name)
        self.folders = {
            "monitor": self.base / "secrets",
            "writer": self.base / "secrets-write",
        }
        for folder in self.folders.values():
            folder.mkdir(mode=0o700)
        self.layer = self.base / "settings" / "deploy.json"
        self.layer.parent.mkdir()
        self.marks = self.base / "code" / "moved"
        self.marks.parent.mkdir()
        self.chowned = {}

    def fchown(self, fd: int, uid: int, gid: int) -> None:
        self.chowned[os.fstat(fd).st_ino] = (uid, gid)

    def no_marks(self) -> None:
        """The code volume as no start has marked it: each copy is due again, as at a deployment's first start."""
        shutil.rmtree(self.marks, ignore_errors=True)

    def at(self, keeper: str, name: str) -> pathlib.Path:
        return self.folders[keeper] / name

    def older(self, name: str, text: str) -> pathlib.Path:
        p = self.at("monitor", name)
        p.write_text(text, encoding="utf-8")
        p.chmod(0o600)
        return p

    def copies(self, failed: bool = False) -> list:
        """What the copies say, once no line holds a value, and no file is left but the copies."""
        lines, failing = rk_root.copies(
            {k: str(v) for k, v in self.folders.items()},
            OWNERS,
            str(self.layer),
            self.fchown,
            str(self.marks),
        )
        said = "\n".join(lines)
        for value in VALUES:
            self.assertNotIn(value, said)
        self.assertEqual(failing, failed, lines)
        for folder in self.folders.values():
            if folder.is_dir():
                self.assertEqual(
                    [n for n in os.listdir(folder) if n.startswith(".")], []
                )
        return lines

    def copied(self, keeper: str, name: str) -> str:
        """The new copy NAME in KEEPER's folder, its text: a file of its own, the keeper's user's, mode 600."""
        s = self.at(keeper, name).lstat()
        self.assertTrue(stat.S_ISREG(s.st_mode))
        self.assertEqual(s.st_nlink, 1)
        self.assertEqual(stat.S_IMODE(s.st_mode), 0o600)
        self.assertEqual(self.chowned.get(s.st_ino), OWNERS[keeper])
        return self.at(keeper, name).read_text(encoding="utf-8")

    def line(self, old: str, keeper: str, name: str, what: str = "") -> str:
        whose = rk_root.WHOSE[keeper]
        return (
            f"the one-time move copied {what}{self.at('monitor', old)} to {self.at(keeper, name)} ({whose}, mode "
            f"600); {STAYS}"
        )

    def test_the_writer_s_secrets_go_to_its_volume(self):
        """The team chat's webhook and the web host's key, from the monitor's volume to the writer's, under their new
        names, byte for byte; the older copies stay, and the next start copies nothing.
        """
        hook = self.older("teams_webhook", HOOK)
        web = self.older("webhost_ed25519", SSH_KEY)
        self.assertEqual(
            self.copies(),
            [
                self.line("teams_webhook", "writer", "teams_webhook"),
                self.line("webhost_ed25519", "writer", "web_host_ssh_key"),
            ],
        )
        self.assertEqual(self.copied("writer", "teams_webhook"), HOOK)
        self.assertEqual(self.copied("writer", "web_host_ssh_key"), SSH_KEY)
        self.assertEqual((hook.read_text(), web.read_text()), (HOOK, SSH_KEY))
        self.assertEqual(self.copies(), [])

    def test_none_when_the_new_copy_exists(self):
        """A new copy that exists, whatever it is, is left as it is, and nothing is said."""
        self.older("teams_webhook", HOOK)
        self.older("webhost_ed25519", SSH_KEY)
        self.older(rk_root.CLAUDE[1], json.dumps({"ANTHROPIC_API_KEY": KEY}))
        self.at("writer", "teams_webhook").write_text("theirs", encoding="utf-8")
        self.at("writer", "web_host_ssh_key").symlink_to(self.base / "nowhere")
        self.at("monitor", rk_root.CLAUDE[3]).write_text("theirs", encoding="utf-8")
        self.assertEqual(self.copies(), [])
        self.assertEqual(self.at("writer", "teams_webhook").read_text(), "theirs")
        self.assertEqual(self.at("monitor", rk_root.CLAUDE[3]).read_text(), "theirs")
        self.assertEqual(self.chowned, {})

    def test_an_older_copy_that_holds_nothing_is_not_copied(self):
        self.older("teams_webhook", " \n")
        self.assertEqual(self.copies(), [])
        self.assertFalse(self.at("writer", "teams_webhook").exists())

    def test_the_claude_login_s_key_alone(self):
        """Each shape of claude_env that the settings carry: the key alone is copied to the claude login, one
        line, and nothing else of the older settings."""
        old, new = rk_root.CLAUDE[1], rk_root.CLAUDE[3]
        gateway = "https://gateway.example.com"
        for env in (
            {"ANTHROPIC_API_KEY": KEY},
            {"ANTHROPIC_API_KEY": f" {KEY}\n"},
            {"ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_BASE_URL": gateway},
            # an address that only check 16's judgement flags: root reads with none, and check 16 says it copies
            {
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_BASE_URL": "https://user:pw-not-real@gateway.example.com",
            },
            # blank space is no key
            {"ANTHROPIC_API_KEY": " \n", "ANTHROPIC_AUTH_TOKEN": KEY},
            {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY},
            {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": f"X-Api-Key: {KEY}"},
            {
                "ANTHROPIC_AUTH_TOKEN": KEY,
                "ANTHROPIC_BASE_URL": gateway,
                "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Bearer {KEY}",
                "ANTHROPIC_MODEL": "opus",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
        ):
            with self.subTest(env=sorted(env)):
                self.at("monitor", new).unlink(missing_ok=True)
                self.no_marks()
                self.older(old, json.dumps(env))
                self.assertEqual(
                    self.copies(),
                    [self.line(old, "monitor", new, "the key alone in ")],
                )
                self.assertEqual(self.copied("monitor", new), f"{KEY}\n")
                self.assertEqual(json.loads(self.at("monitor", old).read_text()), env)

    def test_a_shape_the_settings_cannot_carry_copies_nothing(self):
        """A shape the settings cannot carry, an older copy with no key, or one the reader refuses: no key is
        copied, and the line names what stops it, never a header's name, and the fix. What stops it is the reader's
        `move`, which check 16 shows (rkmon_lib/confirm.py)."""
        old, new = rk_root.CLAUDE[1], rk_root.CLAUDE[3]
        p = self.at("monitor", old)
        fix = "the fix: rkmon login claude, on the host"
        cannot = f"the one-time move did not copy the key in {p}: "
        reader = rk_root.runpy.run_path(str(ROOT / "rkmon_lib" / "claude_env_shape.py"))
        for env, said in (
            (
                {
                    "ANTHROPIC_API_KEY": " \n",
                    "ANTHROPIC_BASE_URL": "https://gw.example.com",
                },
                f"the one-time move copied nothing from {p}: it has no key; {fix}",
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY},
                f"{cannot}the deploy settings cannot carry everything in it: ANTHROPIC_API_KEY and "
                f"ANTHROPIC_AUTH_TOKEN are two different keys, and the claude login is one key; {fix}",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": "X-Gateway-Sub: sub-not-real",
                },
                f"{cannot}the deploy settings cannot carry everything in it: a custom header sends a value that is "
                f"not the key, and no setting sends a header of fixed text; {fix}",
            ),
            (
                {"ANTHROPIC_BASE_URL": "https://gateway.example.com"},
                f"the one-time move copied nothing from {p}: it has no key; {fix}",
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "OTHER_VARIABLE": OTHER_KEY},
                f"{cannot}{p} names 1 variable the session does not take; {fix}",
            ),
        ):
            with self.subTest(env=sorted(env)):
                self.no_marks()
                self.older(old, json.dumps(env))
                self.assertEqual(self.copies(), [said])
                self.assertNotIn("X-Gateway-Sub", said)
                self.assertFalse(os.path.lexists(self.at("monitor", new)))
                if "OTHER_VARIABLE" not in env:
                    move = reader["move"](reader["read"](str(p)))
                    self.assertTrue(move and f": {move}; {fix}" in said, move)
        p.unlink()
        p.symlink_to(self.older("elsewhere", json.dumps({"ANTHROPIC_API_KEY": KEY})))
        self.no_marks()
        self.assertEqual(self.copies(), [f"{cannot}{p} is a link, not a file; {fix}"])
        self.assertFalse(os.path.lexists(self.at("monitor", new)))

    def test_a_binding_s_key_while_a_pass_runs_on_its_binding(self):
        """A binding's key, copied from its older names to the name its binding's api_key_secret_name gives it, while
        review automation is on and a pass runs on that binding."""
        layer(self.layer, first_pass_on("glm"), {"glm": GLM})
        self.older("glm_key", KEY)
        self.assertEqual(
            self.copies(), [self.line("glm_key", "monitor", "glm_api_key")]
        )
        self.assertEqual(self.copied("monitor", "glm_api_key"), KEY)
        self.assertEqual(self.copies(), [])

    def test_the_first_review_pass_s_key_with_and_without_a_drafting_model(self):
        """The local worker's key goes to the first review pass's while a pass drafts on a binding that names it;
        else it is reported, and not copied: no feature uses it."""
        worker, new = "worker_api_key", sr.FIRST_PASS_KEY
        drafter = {**GLM, "api_key_secret_name": new}
        unused = f"the one-time move did not copy {self.at('monitor', worker)} to {self.at('monitor', new)}: "
        for case, features, bindings, said in (
            (
                "no binding names the key",
                None,
                {"glm": GLM},
                f"{unused}no model binding names {new}, so no feature uses it",
            ),
            (
                "no pass runs on the binding",
                None,
                {"drafter": drafter},
                f"{unused}no pass of review automation runs on a model binding that names {new}, so no feature "
                "uses it",
            ),
            (
                "review automation is off",
                {"review_automation": {"enabled": False}},
                {"drafter": drafter},
                f"{unused}review automation is off, so no feature uses it",
            ),
            (
                "no layer",
                None,
                None,
                f"{unused}your confirmed settings are not in the container yet",
            ),
        ):
            with self.subTest(case=case):
                if bindings is None:
                    self.layer.unlink(missing_ok=True)
                else:
                    layer(self.layer, features, bindings)
                self.older(worker, KEY)
                self.assertEqual(self.copies(), [said])
                self.assertFalse(os.path.lexists(self.at("monitor", new)))
        self.layer.write_text("not json", encoding="utf-8")
        self.assertEqual(
            self.copies(),
            [
                f"{unused}your confirmed settings cannot be read (Expecting value: line 1 column 1 (char 0))"
            ],
        )
        layer(self.layer, first_pass_on("drafter"), {"drafter": drafter})
        self.assertEqual(self.copies(), [self.line(worker, "monitor", new)])
        self.assertEqual(self.copied("monitor", new), KEY)

    def test_two_older_copies_of_a_key_that_differ_are_not_copied(self):
        """Two older copies of a key that differ are not copied, at that start or a later one, whatever they hold by
        then; two that hold one value are."""
        layer(self.layer, first_pass_on("glm"), {"glm": GLM})
        self.older("glm_key", KEY)
        self.older("glm", OTHER_KEY)
        both = f"{self.at('monitor', 'glm_key')} or {self.at('monitor', 'glm')}"
        self.assertEqual(
            self.copies(),
            [
                f"the one-time move did not copy {both}: they hold different values; the fix: rkmon secret "
                "glm_api_key, on the host"
            ],
        )
        self.assertFalse(os.path.lexists(self.at("monitor", "glm_api_key")))
        self.older("glm", f"{KEY}\n")
        self.assertEqual(
            self.copies(),
            [
                AGAIN.format(
                    both, self.at("monitor", "glm_api_key"), "secret glm_api_key"
                )
            ],
        )
        self.assertFalse(os.path.lexists(self.at("monitor", "glm_api_key")))
        self.no_marks()
        self.assertEqual(
            self.copies(), [self.line("glm_key", "monitor", "glm_api_key")]
        )
        self.assertEqual(self.copied("monitor", "glm_api_key"), KEY)

    def test_the_atlassian_login_and_the_gpu_host_s_key_are_never_copied(self):
        """The atlassian login, the GPU host's key and the older copies the move only removes as it ends stay as they
        are, and nothing is said of them."""
        login = self.at("monitor", "atlassian")
        login.mkdir()
        (login / "credentials").write_text(KEY, encoding="utf-8")
        for name in ("gpu_ssh_key", "admin_token", "gh_read_token"):
            self.older(name, KEY)
        before = sorted(os.listdir(self.folders["monitor"]))
        self.assertEqual(self.copies(), [])
        self.assertEqual(sorted(os.listdir(self.folders["monitor"])), before)
        self.assertEqual(os.listdir(self.folders["writer"]), [])

    def test_an_older_copy_that_is_not_a_plain_file_is_not_copied(self):
        """A link, a folder, a pipe, a socket or a file larger than COPY_LIMIT is not read, nor copied: the line says
        why, and the fix; a refusal, never an ERROR."""
        web = self.at("monitor", "webhost_ed25519")
        web.symlink_to(self.older("elsewhere", SSH_KEY))
        hook = self.at("monitor", "teams_webhook")
        os.mkfifo(hook)
        fix = "the fix: rkmon secret {}, on the host"
        self.assertEqual(
            self.copies(),
            [
                f"the one-time move did not copy {hook}: {hook} is not a file; {fix.format('teams_webhook')}",
                f"the one-time move did not copy {web}: {web} is a link, not a file; "
                f"{fix.format('web_host_ssh_key')}",
            ],
        )
        hook.unlink()
        hook.write_text("x" * (rk_root.COPY_LIMIT + 1), encoding="utf-8")
        self.no_marks()
        self.assertEqual(
            self.copies()[0],
            f"the one-time move did not copy {hook}: {hook} is larger than 64 KiB; {fix.format('teams_webhook')}",
        )
        # a socket does not open (ENXIO)
        hook.unlink()
        claude = self.at("monitor", rk_root.CLAUDE[1])
        for p in (hook, claude):
            with socket.socket(socket.AF_UNIX) as s:
                s.bind(str(p))
        self.no_marks()
        said = self.copies()
        self.assertEqual(
            [said[0], said[-1]],
            [
                f"the one-time move did not copy {hook}: {hook} is not a file; {fix.format('teams_webhook')}",
                f"the one-time move did not copy the key in {claude}: {claude} is not a file; the fix: rkmon login "
                "claude, on the host",
            ],
        )
        self.assertEqual(os.listdir(self.folders["writer"]), [])

    def test_a_copy_that_cannot_be_written_is_an_error(self):
        self.older("teams_webhook", HOOK)
        self.folders["writer"].rmdir()
        new = self.at("writer", "teams_webhook")
        self.assertEqual(
            self.copies(failed=True),
            [
                f"ERROR: the one-time move could not write {new}: No such file or directory; the fix: rkmon secret "
                "teams_webhook, on the host"
            ],
        )

    @unittest.skipIf(os.geteuid() == 0, "root reads a file of mode 000")
    def test_an_older_copy_that_cannot_be_read_is_an_error(self):
        """An older copy that cannot be read is not copied: an ERROR, with the fix."""
        hook = self.older("teams_webhook", HOOK)
        claude = self.older(rk_root.CLAUDE[1], json.dumps({"ANTHROPIC_API_KEY": KEY}))
        for p in (hook, claude):
            p.chmod(0)
        cannot = "ERROR: the one-time move could not read {}: Permission denied; the fix: rkmon {}, on the host"
        self.assertEqual(
            self.copies(failed=True),
            [
                cannot.format(hook, "secret teams_webhook"),
                cannot.format(claude, "login claude"),
            ],
        )
        self.assertEqual(os.listdir(self.folders["writer"]), [])
        self.assertFalse(self.at("monitor", rk_root.CLAUDE[3]).exists())
        for mark in ("writer.teams_webhook", f"monitor.{rk_root.CLAUDE[3]}"):
            self.assertFalse((self.marks / mark).exists(), mark)

    def test_a_name_taken_by_then_is_left_as_it_is(self):
        """The copy is linked in place only while its name is free: a file made there meanwhile stays."""
        theirs = self.at("writer", "teams_webhook")
        theirs.write_text("theirs", encoding="utf-8")
        self.assertFalse(
            rk_root._put(
                str(self.folders["writer"]),
                "teams_webhook",
                HOOK.encode(),
                OWNERS["writer"],
                self.fchown,
            )
        )
        self.assertEqual(theirs.read_text(), "theirs")
        self.assertEqual(os.listdir(self.folders["writer"]), ["teams_webhook"])

    def test_a_copy_made_is_never_made_again(self):
        """A copy a start made is not made again: when the writer's copy has gone and the monitor's user has written
        the older name since, the line names it and the fix, and the writer's volume stays as it is.
        """
        hook = self.older("teams_webhook", HOOK)
        new = self.at("writer", "teams_webhook")
        self.assertEqual(
            self.copies(), [self.line("teams_webhook", "writer", "teams_webhook")]
        )
        new.unlink()
        hook.write_text(OTHER_KEY, encoding="utf-8")
        self.assertEqual(
            self.copies(), [AGAIN.format(hook, new, "secret teams_webhook")]
        )
        self.assertEqual(os.listdir(self.folders["writer"]), [])

    def test_a_file_written_later_at_an_older_name_is_never_copied(self):
        """A first start that finds no older copy deals with each copy: an older copy written later, as the monitor's
        user may write one, is never copied, to the writer's volume least of all."""
        self.assertEqual(self.copies(), [])
        claude = rk_root.CLAUDE
        written = (
            ("teams_webhook", "writer", "teams_webhook", "secret teams_webhook", HOOK),
            (
                "webhost_ed25519",
                "writer",
                "web_host_ssh_key",
                "secret web_host_ssh_key",
                SSH_KEY,
            ),
            (
                claude[1],
                claude[2],
                claude[3],
                claude[4],
                json.dumps({"ANTHROPIC_API_KEY": KEY}),
            ),
        )
        for old, _keeper, _new, _command, text in written:
            self.older(old, text)
        self.assertEqual(
            self.copies(),
            [
                AGAIN.format(self.at("monitor", old), self.at(keeper, new), command)
                for old, keeper, new, command, _text in written
            ],
        )
        self.assertEqual(os.listdir(self.folders["writer"]), [])
        self.assertFalse(self.at("monitor", claude[3]).exists())

    def test_the_marks(self):
        """A mark for each copy a start dealt with, named after the new copy's keeper and name: an empty file, mode
        644, in a folder of mode 755, whatever the umask (the entry point's is 027), so that the doctor's check of the
        code volume reads them."""
        old = os.umask(0o027)
        self.addCleanup(os.umask, old)
        self.older("teams_webhook", HOOK)
        self.copies()
        self.assertEqual(
            sorted(os.listdir(self.marks)),
            sorted(
                [f"{keeper}.{name}" for _w, _o, keeper, name, _c in rk_root.COPIES]
                + [f"{rk_root.CLAUDE[2]}.{rk_root.CLAUDE[3]}"]
                + [f"monitor.{name}" for name in rk_root.OLDER_KEY_NAMES.values()]
            ),
        )
        self.assertEqual(stat.S_IMODE(self.marks.lstat().st_mode), 0o755)
        for mark in self.marks.iterdir():
            s = mark.lstat()
            self.assertEqual((stat.S_IMODE(s.st_mode), s.st_size), (0o644, 0), mark)

    def test_a_copy_left_for_later_has_no_mark(self):
        """A copy that cannot be written, and one no feature uses, have no mark: a later start makes them."""
        self.older("teams_webhook", HOOK)
        self.folders["writer"].rmdir()
        self.assertTrue(self.copies(failed=True)[0].startswith("ERROR: "))
        self.assertFalse((self.marks / "writer.teams_webhook").exists())
        self.folders["writer"].mkdir(mode=0o700)
        self.assertEqual(
            self.copies(), [self.line("teams_webhook", "writer", "teams_webhook")]
        )
        self.assertEqual(self.copied("writer", "teams_webhook"), HOOK)
        layer(self.layer, {"review_automation": {"enabled": False}}, {"glm": GLM})
        key = self.older("glm_key", KEY)
        self.assertEqual(
            self.copies(),
            [
                f"the one-time move did not copy {key} to {self.at('monitor', 'glm_api_key')}: review automation is "
                "off, so no feature uses it"
            ],
        )
        self.assertFalse((self.marks / "monitor.glm_api_key").exists())
        layer(self.layer, first_pass_on("glm"), {"glm": GLM})
        self.assertEqual(
            self.copies(), [self.line("glm_key", "monitor", "glm_api_key")]
        )
        self.assertTrue((self.marks / "monitor.glm_api_key").exists())

    def test_marks_another_user_may_write_make_no_copy(self):
        """Marks that are not root's alone to write, or not a folder, cannot be trusted: no copy is made, an ERROR."""
        self.older("teams_webhook", HOOK)
        cannot = f"ERROR: the one-time move made no copy: its marks cannot be kept in {self.marks}: "
        self.marks.mkdir()
        self.marks.chmod(0o777)
        self.assertEqual(
            self.copies(failed=True),
            [f"{cannot}another user may write it (uid {os.getuid()}, mode 0777)"],
        )
        self.marks.chmod(0o755)
        with mock.patch.object(rk_root.os, "geteuid", lambda: os.getuid() + 1):
            self.assertEqual(
                self.copies(failed=True),
                [f"{cannot}another user may write it (uid {os.getuid()}, mode 0755)"],
            )
        self.marks.rmdir()
        self.marks.symlink_to(self.base)
        self.assertEqual(self.copies(failed=True), [f"{cannot}it is not a folder"])
        self.marks.unlink()
        self.marks.write_text("", encoding="utf-8")
        self.assertEqual(self.copies(failed=True), [f"{cannot}it is not a folder"])
        self.assertEqual(os.listdir(self.folders["writer"]), [])

    def test_a_mark_that_cannot_be_made_is_an_error(self):
        """A copy whose mark cannot be made is an ERROR, and the start waits: else a later start might make it."""
        self.older("teams_webhook", HOOK)
        full = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(rk_root, "_mark", side_effect=full):
            said = self.copies(failed=True)
        new = self.at("writer", "teams_webhook")
        self.assertEqual(
            said[:2],
            [
                self.line("teams_webhook", "writer", "teams_webhook"),
                f"ERROR: the one-time move could not mark its copy to {new} as dealt with, in {self.marks}: No "
                "space left on device",
            ],
        )
        self.assertTrue(
            all(
                s.startswith("ERROR: the one-time move could not mark ")
                for s in said[2:]
            ),
            said,
        )


class StartTest(unittest.TestCase):
    """The start makes the copies after the write login's move and before the code volume's check, which runs whatever
    they did; a copy that cannot be made ends the start with exit 1."""

    class Volume:
        def start(self) -> list:
            return ["the code volume"]

    def start(self, copies, owner=lambda: (os.getuid(), os.getgid())) -> tuple:
        out = io.StringIO()
        with mock.patch.object(rk_root.os, "geteuid", lambda: 0), mock.patch.object(
            rk_root, "_owner", owner
        ), mock.patch.object(rk_root, "write_volume", lambda: []), mock.patch.object(
            rk_root, "move_write_login", lambda: ["the write login"]
        ), mock.patch.object(
            rk_root, "copies", copies
        ), contextlib.redirect_stdout(
            out
        ):
            code = rk_root.main(["start"], volume=self.Volume())
        return code, out.getvalue().splitlines()

    def test_the_copies_at_the_start(self):
        self.assertEqual(
            self.start(lambda: (["a copy"], False)),
            (rk_root.DONE, ["the write login", "a copy", "the code volume"]),
        )
        self.assertEqual(
            self.start(lambda: (["ERROR: a copy"], True)),
            (rk_root.ERROR, ["the write login", "ERROR: a copy", "the code volume"]),
        )

        def broken():
            raise KeyError("no such user")

        self.assertEqual(
            self.start(broken),
            (
                rk_root.ERROR,
                [
                    "the write login",
                    "ERROR: the one-time move's copies: 'no such user'",
                    "the code volume",
                ],
            ),
        )

    def test_no_copies_when_the_image_has_no_write_user(self):
        def no_user():
            raise KeyError("no write user")

        def copies():
            raise AssertionError("no copies")

        code, said = self.start(copies, no_user)
        self.assertEqual(
            (code, said),
            (
                rk_root.NO_USER,
                [
                    f"ERROR: no user {rk_root.WRITE_USER} in this image",
                    "the code volume",
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
