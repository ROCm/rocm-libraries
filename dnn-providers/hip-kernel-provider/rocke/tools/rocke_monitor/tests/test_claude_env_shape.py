# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The older Claude secret's one reader (rkmon_lib/claude_env_shape.py): its shape, names and settings
only, and integrations.claude_code's settings made from it, for each shape they can carry, and each shape the settings
cannot carry; the shape with no judgement of what looks like a secret, as the one-time move reads it, and the key it
copies; the files it refuses; the shape as JSON and back; and the module loaded by its path, with nothing but the
standard library, as root loads it. No credential reaches the shape, its sentences or a message.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rkmon_lib import claude_env_shape as ces  # noqa: E402
from rkmon_lib.settings_file import secret_shape  # noqa: E402

KEY = "test-" + "key-not-real"
OTHER_KEY = "test-" + "other-key-not-real"
PASSWORD = "pw-" + "not-real"
HEADERS = "X-Gateway-Sub: " + "sub-not-real"
TOKEN = "sk-" + "ant-" + "notreal0123456789abcdef"


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.path = self.dir / "claude_env"

    def write(self, env) -> None:
        self.path.write_text(json.dumps(env), encoding="utf-8")

    def refused(self, looks_secret=None) -> str:
        """The message `read` refuses the file with, which holds no credential."""
        with self.assertRaises(ValueError) as cm:
            ces.read(self.path, looks_secret)
        said = str(cm.exception)
        for secret in (KEY, OTHER_KEY, PASSWORD, TOKEN):
            self.assertNotIn(secret, said)
        return said


class TestShape(Base):
    """claude_env's shape, judged as the translation and check 16 judge it (settings_file.secret_shape), and
    integrations.claude_code's settings made from it: each shape they can carry, and each shape the settings
    cannot carry, with its fix, an edit of the file's variables."""

    def shape(self, **env):
        """The shape of a claude_env of ENV, the settings made from it, and what they cannot carry, each by what was
        found: each fix's edits are of the session's variables, the sentence with no header's name names none, and no
        text holds a credential."""
        self.write(env)
        shape = ces.read(self.path, secret_shape)
        made, cannot = ces.settings(shape)
        for c in cannot:
            self.assertTrue(c.fix)
            for part in c.fix:
                self.assertTrue(
                    isinstance(part, str)
                    or (
                        isinstance(part, ces.Edit) and part.variable in ces.CLAUDE_VARS
                    ),
                    part,
                )
            for h in shape.headers:
                self.assertFalse(h.name and h.name in c.unnamed, c.unnamed)
        said = json.dumps([made, [[c.found, c.unnamed, repr(c.fix)] for c in cannot]])
        for secret in (KEY, OTHER_KEY, PASSWORD):
            self.assertNotIn(secret, repr(shape) + said)
        return shape, made, [c.found for c in cannot]

    def test_each_one_says_how_to_fix_it(self):
        """Each shape the settings cannot carry has its fix: the edit of claude_env that lets them carry it (as the
        translation shows it: tests/test_translate_production.py)."""
        cases = [
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY},
                (
                    ces.Edit("ANTHROPIC_API_KEY"),
                    ", or",
                    ces.Edit("ANTHROPIC_AUTH_TOKEN"),
                    ", whichever the gateway does not take",
                ),
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": HEADERS},
                (
                    ces.Edit(
                        "ANTHROPIC_CUSTOM_HEADERS",
                        "its lines without the header X-Gateway-Sub, when the gateway takes requests without it",
                    ),
                ),
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": KEY},
                (
                    ces.Edit(
                        "ANTHROPIC_CUSTOM_HEADERS",
                        "one header a line: a name, a colon and a value",
                    ),
                ),
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_BASE_URL": f"https://gateway.example.com/{KEY}",
                },
                (
                    ces.Edit(
                        "ANTHROPIC_BASE_URL",
                        "the gateway's address, with no credential in it",
                    ),
                ),
            ),
        ]
        for env, fix in cases:
            with self.subTest(fix=fix):
                self.write(env)
                _made, cannot = ces.settings(ces.read(self.path, secret_shape))
                self.assertIn(fix, [c.fix for c in cannot])

    def test_each_row_of_the_table(self):
        rows = [
            ({"ANTHROPIC_API_KEY": KEY}, ("header", "", "")),
            ({"ANTHROPIC_AUTH_TOKEN": KEY}, ("bearer", "", "")),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY},
                ("header_and_bearer", "", ""),
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
                },
                ("header", "X-Gateway-Key", ""),
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}\n",
                },
                ("header", "X-Gateway-Key", "Token"),
            ),
            (
                {
                    "ANTHROPIC_AUTH_TOKEN": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Bearer {KEY}",
                },
                ("header_and_bearer", "X-Gateway-Key", "Bearer"),
            ),
        ]
        for env, want in rows:
            with self.subTest(env=sorted(env)):
                _shape, made, cannot = self.shape(**env)
                self.assertEqual(cannot, [])
                self.assertEqual(
                    (
                        made["api_key_sent_as"],
                        made["api_key_header"],
                        made["api_key_scheme"],
                    ),
                    want,
                )

    def test_the_address_the_model_variables_and_the_traffic_switch(self):
        shape, made, cannot = self.shape(
            ANTHROPIC_DEFAULT_OPUS_MODEL="example-opus",
            ANTHROPIC_API_KEY=KEY,
            ANTHROPIC_BASE_URL="https://llm-gateway.example.com/anthropic",
            ANTHROPIC_MODEL="example-sonnet",
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="0",
        )
        self.assertEqual(
            shape.variables,
            (
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            ),
        )
        self.assertEqual(
            shape.models, ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL")
        )
        self.assertEqual(shape.nonessential_traffic, "0")
        self.assertEqual(
            (made["base_url"], cannot),
            ("https://llm-gateway.example.com/anthropic", []),
        )
        shape, _made, _cannot = self.shape(
            ANTHROPIC_API_KEY=KEY, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=KEY
        )
        self.assertEqual(shape.nonessential_traffic, ces.ANOTHER_VALUE)
        # A variable set to "", or to blank space alone, is not set.
        for blank in ("", " \n\t"):
            with self.subTest(blank=blank):
                shape, _made, _cannot = self.shape(
                    ANTHROPIC_API_KEY=KEY,
                    ANTHROPIC_MODEL=blank,
                    ANTHROPIC_AUTH_TOKEN=blank,
                    ANTHROPIC_BASE_URL=blank,
                )
                self.assertEqual(shape, ces.Shape(variables=("ANTHROPIC_API_KEY",)))

    def test_no_secret_is_an_empty_shape(self):
        shape = ces.read(self.path, secret_shape)
        self.assertEqual(shape, ces.Shape())
        self.assertEqual(ces.read_key(self.path), (ces.Shape(), ""))
        self.assertEqual(
            ces.settings(shape),
            (
                {
                    "base_url": "",
                    "api_key_sent_as": "header",
                    "api_key_header": "",
                    "api_key_scheme": "",
                },
                [],
            ),
        )

    def test_what_the_settings_cannot_carry(self):
        cases = [
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": HEADERS},
                "the custom header X-Gateway-Sub sends a value that is not the key",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-One: {KEY}\nX-Two: {KEY}",
                },
                "ANTHROPIC_CUSTOM_HEADERS has 2 headers",
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY},
                "are two different keys, and the claude login is one key",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_AUTH_TOKEN": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
                },
                "and a custom header all send the key",
            ),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": KEY},
                "has 1 line that does not read as a header: a name, a colon and a value",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"{KEY}: {KEY}\nX-Gateway-Key: {KEY} {KEY}",
                },
                "has 2 lines that do not read as a header",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_BASE_URL": f"https://user:{PASSWORD}@gateway.example.com",
                },
                "ANTHROPIC_BASE_URL looks like an address with a user or password in it, and a settings file has no "
                "credential",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_BASE_URL": f"https://gateway.example.com/{KEY}",
                },
                "ANTHROPIC_BASE_URL looks like an address with the key in it",
            ),
        ]
        for env, said in cases:
            with self.subTest(said=said):
                shape, made, cannot = self.shape(**env)
                self.assertTrue(any(said in c for c in cannot), cannot)
                self.assertEqual(made["base_url"], "")

    def test_a_name_or_a_word_that_may_be_a_credential(self):
        """A header's name that looks like a secret, and a word before the key that is no scheme (letters and "-", 20
        at most), each give a line that does not read as a header: the shape holds neither.
        """
        self.assertTrue(secret_shape(TOKEN))
        for headers in (
            f"{TOKEN}: {KEY}",
            f"X-Gateway-Key: {TOKEN} {KEY}",
            f"X-Gateway-Key: Tok3n {KEY}",
        ):
            with self.subTest(headers=headers):
                shape, made, cannot = self.shape(
                    ANTHROPIC_API_KEY=KEY, ANTHROPIC_CUSTOM_HEADERS=headers
                )
                self.assertEqual(shape.headers, (ces.Header("", False, ""),))
                self.assertNotIn(TOKEN, repr(shape) + json.dumps([made, cannot]))
                self.assertEqual(
                    cannot,
                    [
                        "ANTHROPIC_CUSTOM_HEADERS has 1 line that does not read as a header: a name, a colon and a "
                        "value"
                    ],
                )

    def test_the_sentence_with_no_headers_name(self):
        """Each Cannot's UNNAMED is its FOUND, but for a header named in it: the same sentence without the name."""
        self.write({"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": HEADERS})
        (c,) = ces.settings(ces.read(self.path, secret_shape))[1]
        self.assertIn("X-Gateway-Sub", c.found)
        self.assertEqual(
            c.unnamed,
            "a custom header sends a value that is not the key, and no setting sends a header of fixed text",
        )
        self.write({"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY})
        (c,) = ces.settings(ces.read(self.path, secret_shape))[1]
        self.assertEqual(c.unnamed, c.found)


class TestNoJudgement(Base):
    """The shape with no judgement of what looks like a secret, as the one-time move reads it (`read_key`): a variable
    the session does not take is counted, never named; the address is taken to hold a credential when the key is in
    it; and the key, which only the move reads."""

    def test_a_variable_the_session_does_not_take(self):
        self.write({"ANTHROPIC_API_KEY": KEY, "OTHER_VARIABLE": "x", TOKEN: "y"})
        said = self.refused()
        self.assertTrue(
            said.endswith("names 2 variables the session does not take"), said
        )
        self.assertNotIn("OTHER_VARIABLE", said)
        self.assertTrue(
            self.refused(secret_shape).endswith(
                "names variables the session does not take: OTHER_VARIABLE and 1 more"
            )
        )
        self.write({"ANTHROPIC_API_KEY": KEY, "OTHER_VARIABLE": "x"})
        self.assertTrue(
            self.refused().endswith("names 1 variable the session does not take")
        )
        # A name that does not read as a variable's is counted, judged or not.
        self.write({"ANTHROPIC_API_KEY": KEY, "other variable": "x"})
        self.assertTrue(
            self.refused(secret_shape).endswith(
                "names 1 variable the session does not take"
            )
        )

    def test_the_address(self):
        self.write(
            {
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_BASE_URL": f"https://gateway.example.com/{KEY}",
            }
        )
        shape = ces.read(self.path)
        self.assertEqual(
            (shape.base_url, shape.base_url_credential), ("", ces.KEY_IN_ADDRESS)
        )
        self.write(
            {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_BASE_URL": "https://gw.example.com"}
        )
        self.assertEqual(ces.read(self.path).base_url, "https://gw.example.com")

    def test_the_key(self):
        cases = [
            ({"ANTHROPIC_API_KEY": f" {KEY}\n"}, KEY),
            ({"ANTHROPIC_AUTH_TOKEN": KEY}, KEY),
            ({"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY}, KEY),
            ({"ANTHROPIC_API_KEY": "  ", "ANTHROPIC_AUTH_TOKEN": KEY}, KEY),
            ({"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY}, KEY),
            ({"ANTHROPIC_BASE_URL": "https://gw.example.com"}, ""),
        ]
        for env, key in cases:
            with self.subTest(env=sorted(env)):
                self.write(env)
                shape, got = ces.read_key(self.path)
                self.assertEqual(got, key)
                self.assertEqual(shape, ces.read(self.path))
                self.assertNotIn(KEY, repr(shape))

    def test_why_the_move_copies_no_key(self):
        """`move`, of the shape with no judgement: what the settings cannot carry, by its sentences with no header's
        name; else NO_KEY when the file holds no key, blank space being none; else "", and the move has a key to copy.
        An address that only the judgement flags is no reason."""
        cases = [
            ({"ANTHROPIC_API_KEY": KEY}, ""),
            ({"ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_MODEL": "example-sonnet"}, ""),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_BASE_URL": f"https://user:{PASSWORD}@gateway.example.com",
                },
                "",
            ),
            ({"ANTHROPIC_BASE_URL": "https://gw.example.com"}, ces.NO_KEY),
            (
                {"ANTHROPIC_API_KEY": " \n", "ANTHROPIC_MODEL": "example-sonnet"},
                ces.NO_KEY,
            ),
            ({}, ces.NO_KEY),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": OTHER_KEY},
                "the deploy settings cannot carry everything in it: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are two "
                "different keys, and the claude login is one key",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"{HEADERS}\nX-Gateway-Key: {KEY}",
                },
                "the deploy settings cannot carry everything in it: a custom header sends a value that is not the key, "
                "and no setting sends a header of fixed text; ANTHROPIC_CUSTOM_HEADERS has 2 headers, and the "
                "settings send the key in one header at most",
            ),
        ]
        for env, why in cases:
            with self.subTest(env=sorted(env)):
                self.write(env)
                shape, key = ces.read_key(self.path)
                self.assertEqual(ces.move(shape), why)
                self.assertEqual(bool(key), why != ces.NO_KEY)
                self.assertNotIn("X-Gateway", why)

    def test_a_header_is_judged_only_when_asked(self):
        """With no judgement a header's name is kept as a header's name; the sentences the move prints (UNNAMED) name
        no header."""
        self.write(
            {
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_CUSTOM_HEADERS": f"{TOKEN}: {KEY}\n{HEADERS}",
            }
        )
        shape = ces.read(self.path)
        self.assertEqual([h.name for h in shape.headers], [TOKEN, "X-Gateway-Sub"])
        for c in ces.settings(shape)[1]:
            self.assertNotIn(TOKEN, c.unnamed)
            self.assertNotIn("X-Gateway-Sub", c.unnamed)
        self.assertEqual(
            [h.name for h in ces.read(self.path, secret_shape).headers],
            ["", "X-Gateway-Sub"],
        )


class TestRefused(Base):
    """The files it refuses, each with a message that holds no value: a link, anything but a regular file, one too
    large, one that is not UTF-8 text, and one that is not a JSON object of strings."""

    def test_a_link_and_what_is_not_a_file(self):
        target = self.dir / "elsewhere"
        target.write_text(json.dumps({"ANTHROPIC_API_KEY": KEY}), encoding="utf-8")
        self.path.symlink_to(target)
        self.assertEqual(self.refused(), f"{self.path} is a link, not a file")
        with self.assertRaises(ValueError):
            ces.read_key(self.path)
        self.path.unlink()
        self.path.symlink_to(self.dir / "nothing")
        self.assertEqual(self.refused(), f"{self.path} is a link, not a file")
        self.path.unlink()
        self.path.mkdir()
        self.assertEqual(self.refused(), f"{self.path} is not a file")
        self.path.rmdir()
        if hasattr(os, "mkfifo"):
            # a pipe with no writer: refused at once, never waited on
            os.mkfifo(self.path)
            self.assertEqual(self.refused(), f"{self.path} is not a file")
            self.path.unlink()
        if hasattr(socket, "AF_UNIX"):
            # a socket does not open (ENXIO): refused, as root's move refuses it
            with socket.socket(socket.AF_UNIX) as s:
                s.bind(str(self.path))
            self.assertEqual(self.refused(), f"{self.path} is not a file")
            with self.assertRaisesRegex(ValueError, "is not a file"):
                ces.read_key(self.path)

    def test_its_size_and_its_text(self):
        self.path.write_text(
            json.dumps({"ANTHROPIC_API_KEY": "k" * ces.LIMIT_BYTES}), encoding="utf-8"
        )
        self.assertEqual(self.refused(), f"{self.path} is larger than 64 KiB")
        self.path.write_bytes(b'{"ANTHROPIC_API_KEY": "\xff"}')
        self.assertEqual(self.refused(), f"{self.path} is not UTF-8 text")

    def test_what_is_not_a_json_object_of_strings(self):
        for text, said in (
            (f"ANTHROPIC_API_KEY={KEY}", "is not JSON: Expecting value"),
            (json.dumps([KEY]), "must be a JSON object of strings"),
            (json.dumps({"ANTHROPIC_API_KEY": 1}), "must be a JSON object of strings"),
        ):
            with self.subTest(said=said):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(self.refused(), f"{self.path} {said}")
        # Nesting too deep for Python's reader, and a number too long for it, are errors of their own, which say
        # nothing of the file: which one a Python gives, if any, is its own.
        self.path.write_text("[" * 50000, encoding="utf-8")
        self.assertTrue(self.refused().startswith(f"{self.path} is not JSON: "))
        with mock.patch.object(ces.json, "loads", side_effect=RecursionError(KEY)):
            self.assertEqual(
                self.refused(), f"{self.path} is not JSON: it nests too deep"
            )
        with mock.patch.object(ces.json, "loads", side_effect=ValueError(KEY)):
            self.assertEqual(
                self.refused(), f"{self.path} is not JSON: it cannot be read"
            )
        self.path.write_text(
            '{"ANTHROPIC_API_KEY": ' + "9" * 5000 + "}", encoding="utf-8"
        )
        self.assertTrue(self.refused().startswith(f"{self.path} "))


class TestJson(Base):
    """A shape as one JSON object and back, as check 16's one-off container prints it: each shape `read` makes comes
    back, and anything else is refused."""

    def shapes(self) -> list:
        out = []
        for env in (
            {},
            {
                "ANTHROPIC_BASE_URL": "https://llm-gateway.example.com/anthropic",
                "ANTHROPIC_AUTH_TOKEN": KEY,
                "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Bearer {KEY}\n{KEY}",
                "ANTHROPIC_MODEL": "model-one",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "True",
            },
            {
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_AUTH_TOKEN": OTHER_KEY,
                "ANTHROPIC_BASE_URL": f"https://user:{PASSWORD}@gateway.example.com",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "yes",
            },
        ):
            self.write(env)
            out.append(ces.read(self.path, secret_shape))
        return out

    def test_each_shape_comes_back(self):
        for shape in self.shapes():
            with self.subTest(shape=shape):
                data = json.loads(json.dumps(ces.to_json(shape)))
                self.assertEqual(ces.from_json(data, secret_shape), shape)
                self.assertEqual(ces.from_json(data), shape)
        fixture = json.loads(
            (HERE / "data" / "production_claude_env_shape.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(ces.to_json(ces.from_json(fixture, secret_shape)), fixture)

    def test_what_it_refuses(self):
        good = ces.to_json(self.shapes()[1])
        header = good["headers"][0]
        cases = [
            [],
            dict(good, more=1),
            {k: v for k, v in good.items() if k != "keys_differ"},
            dict(good, variables=["ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL"]),
            dict(good, variables=["ANTHROPIC_MODEL", "ANTHROPIC_MODEL"]),
            dict(good, variables=["OTHER_VARIABLE"]),
            dict(good, variables=("ANTHROPIC_MODEL",)),
            dict(good, headers={}),
            dict(good, headers=[dict(header, more=1)]),
            dict(good, headers=[dict(header, name="X Gateway")]),
            dict(good, headers=[dict(header, name="", value_is_key=True)]),
            dict(good, headers=[dict(header, word="Tok3n")]),
            dict(good, headers=[dict(header, value_is_key=1)]),
            dict(good, keys_differ=0),
            dict(good, base_url=["https://gw.example.com"]),
            dict(good, base_url="https://gw.example.com\nX"),
            dict(good, base_url_credential="a token or key"),
            dict(good, base_url="", base_url_credential="a token or key!"),
            dict(good, nonessential_traffic="yes"),
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    ces.from_json(data)
        # Judged: a header's name or an address that looks like a secret.
        for data in (
            dict(good, headers=[dict(header, name=TOKEN)]),
            dict(good, base_url=f"https://user:{PASSWORD}@gateway.example.com"),
        ):
            with self.subTest(data=data):
                ces.from_json(data)
                with self.assertRaises(ValueError) as cm:
                    ces.from_json(data, secret_shape)
                self.assertNotIn(TOKEN, str(cm.exception))
                self.assertNotIn(PASSWORD, str(cm.exception))


class TestLoadedByItsPath(Base):
    def test_root_loads_it_by_its_path(self):
        """Loaded by its path in a Python of its own (-I: no folder of the package on its path), as root's one-time
        move loads it from the image's own code: it reads the file with rkmon_lib/custom_headers.py beside it.
        """
        self.write(
            {
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}",
            }
        )
        probe = (
            "import json, runpy, sys\n"
            "m = runpy.run_path(sys.argv[1])\n"
            "shape, key = m['read_key'](sys.argv[2])\n"
            "print(json.dumps([m['to_json'](shape), key == sys.argv[3]]))\n"
        )
        p = subprocess.run(
            [sys.executable, "-I", "-c", probe, ces.__file__, str(self.path), KEY],
            cwd=self.dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual((p.returncode, p.stderr), (0, ""))
        self.assertEqual(json.loads(p.stdout), [ces.to_json(ces.read(self.path)), True])
        self.assertEqual(
            ces.read(self.path).headers, (ces.Header("X-Gateway-Key", True, "Token"),)
        )


if __name__ == "__main__":
    unittest.main()
