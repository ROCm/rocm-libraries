# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tests/fakes/fake_sshd.py, the test SSH server of the image test's web-host stage: its arithmetic against the test
vectors of the RFCs it follows (X25519, Ed25519, ChaCha20, Poly1305); then, where OpenSSH's client is installed, the
server as ssh, ssh-keyscan and rsync reach it: a command runs as the login, in its home, with its input, output, errors
and exit status, more than a window of data each way; ssh-keyscan reads its one host key; and it lets in only the one
login by the one key, while the client trusts it only by the host key the client has recorded.
"""

import json
import os
import pathlib
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = ROOT / "tests" / "fakes" / "fake_sshd.py"
sys.path.insert(0, str(SERVER.parent))
sys.dont_write_bytecode = True

import fake_sshd  # noqa: E402

h = bytes.fromhex
LOGIN = "op"


class TestThePrimitives(unittest.TestCase):
    """The arithmetic, against the RFCs' own test vectors."""

    def test_x25519(self):
        # RFC 7748, 5.2: the first vector
        self.assertEqual(
            fake_sshd.x25519(
                h("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"),
                h("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"),
            ).hex(),
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552",
        )
        # RFC 7748, 6.1: Alice's and Bob's public keys, and the secret both reach
        a = h("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
        b = h("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
        big_a = fake_sshd.x25519(a, fake_sshd.BASE_POINT)
        big_b = fake_sshd.x25519(b, fake_sshd.BASE_POINT)
        self.assertEqual(
            big_a.hex(),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
        )
        self.assertEqual(
            big_b.hex(),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
        )
        shared = "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"
        self.assertEqual(fake_sshd.x25519(a, big_b).hex(), shared)
        self.assertEqual(fake_sshd.x25519(b, big_a).hex(), shared)

    def test_ed25519(self):
        # RFC 8032, 7.1, TEST 1: the key pair, and the signature of the empty message
        seed = h("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        public = fake_sshd.ed25519_public(seed)
        self.assertEqual(
            public.hex(),
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        )
        signature = fake_sshd.ed25519_sign(seed, b"")
        self.assertEqual(
            signature.hex(),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        )
        self.assertTrue(fake_sshd.ed25519_verify(public, b"", signature))
        # another message, or a signature changed in one bit, does not verify
        self.assertFalse(fake_sshd.ed25519_verify(public, b"x", signature))
        bent = bytes([signature[0] ^ 1]) + signature[1:]
        self.assertFalse(fake_sshd.ed25519_verify(public, b"", bent))

    def test_chacha20_block(self):
        # RFC 8439, 2.3.2: the key 00..1f, the block counter 1, the nonce 000000090000004a00000000
        block = fake_sshd.chacha20_block(
            bytes(range(32)), struct.pack("<I", 1) + h("000000090000004a00000000")
        )
        self.assertEqual(
            block.hex(),
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
            "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e",
        )

    def test_poly1305(self):
        # RFC 8439, 2.5.2
        key = h("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
        self.assertEqual(
            fake_sshd.poly1305(key, b"Cryptographic Forum Research Group").hex(),
            "a8061dc1305136c6c22b8baf0c0127a9",
        )


@unittest.skipUnless(
    shutil.which("ssh") and shutil.which("ssh-keygen") and shutil.which("ssh-keyscan"),
    "OpenSSH's client is not installed here",
)
class TestTheServer(unittest.TestCase):
    """The server on a free port of 127.0.0.1, as OpenSSH's own client reaches it, in a folder of the test's own: the
    client's home is there too, so it reads none of this machine's ssh files and no agent.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)
        self.home = self.d / "home"  # the login's, on the server
        self.home.mkdir()
        self.host_key = self.keygen("host_key")
        self.login_key = self.keygen("login_key")
        self.log = self.d / "sshd.log"
        (self.d / "client").mkdir()
        self.env = {
            "HOME": str(self.d / "client"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
        }
        ready = self.d / "ready"
        with open(self.d / "sshd.err", "wb") as err:
            self.server = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(SERVER),
                    "--host-key",
                    str(self.host_key),
                    "--authorized-key",
                    f"{self.login_key}.pub",
                    "--user",
                    LOGIN,
                    "--home",
                    str(self.home),
                    "--address",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--log",
                    str(self.log),
                    "--ready",
                    str(ready),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
            )
        self.addCleanup(self.stop)
        end = time.monotonic() + 30
        while not self.listens(ready):
            if self.server.poll() is not None or time.monotonic() > end:
                self.fail(f"the server did not start: {self.errors()}")
            time.sleep(0.05)
        self.port = self.listens(ready)
        self.known = self.d / "known_hosts"
        self.known.write_text(f"[127.0.0.1]:{self.port} {self.public(self.host_key)}\n")

    def stop(self):
        self.server.terminate()
        try:
            self.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait()

    @staticmethod
    def listens(ready: pathlib.Path) -> int:
        """The port the server's --ready file READY gives once written whole, else 0."""
        try:
            text = ready.read_text(encoding="ascii")
        except OSError:
            return 0
        return int(text) if text.endswith("\n") else 0

    def errors(self) -> str:
        return (self.d / "sshd.err").read_text(errors="replace")[-2000:]

    def keygen(self, name: str) -> pathlib.Path:
        """A new ssh-ed25519 key pair, NAME and NAME.pub in the test's folder: the private key's path."""
        path = self.d / name
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "", "-f", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        return path

    @staticmethod
    def public(key: pathlib.Path) -> str:
        """KEY's public key, its type and its base64, as known_hosts and ssh-keyscan give it."""
        return " ".join(pathlib.Path(f"{key}.pub").read_text().split()[:2])

    def ssh_words(self, key: pathlib.Path) -> list:
        """ssh to the server with KEY alone, trusting it by the recorded host key alone, never asking."""
        return [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            str(key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.known}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "ControlPath=none",
            "-p",
            str(self.port),
        ]

    def ssh(self, command: str, data: bytes = b"", key=None, login: str = LOGIN):
        """COMMAND run on the server over ssh, DATA its input: the finished process, its output and errors as bytes."""
        return subprocess.run(
            [*self.ssh_words(key or self.login_key), f"{login}@127.0.0.1", command],
            input=data,
            capture_output=True,
            env=self.env,
            timeout=120,
        )

    def events(self, what: str = "") -> list:
        """The server's log, each line's fields; only the WHAT events when given."""
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [e for e in map(json.loads, lines) if not what or e["event"] == what]

    @staticmethod
    def said(p) -> str:
        return p.stderr.decode(errors="replace")

    def test_a_command_runs_as_the_login_in_its_home(self):
        p = self.ssh('echo "$HOME|$USER|$(pwd)"; echo err >&2; exit 3')
        self.assertEqual(p.returncode, 3, self.said(p))
        home = str(self.home)
        self.assertEqual(p.stdout.decode(), f"{home}|{LOGIN}|{home}\n")
        self.assertIn("err", self.said(p).splitlines())
        fingerprint = subprocess.run(
            ["ssh-keygen", "-l", "-f", f"{self.host_key}.pub"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[1]
        (kex,) = self.events("kex")
        self.assertIn(kex["kex"], fake_sshd.KEX)
        self.assertEqual(
            (kex["cipher"], kex["host_key"]), (fake_sshd.CIPHER, fingerprint)
        )
        # the client asks whether the key would do, then signs: let in once signed
        self.assertEqual(
            [(e["user"], e["signed"], e["ok"]) for e in self.events("auth")][-1],
            (LOGIN, True, True),
        )
        self.assertEqual(
            [e["command"] for e in self.events("exec")],
            ['echo "$HOME|$USER|$(pwd)"; echo err >&2; exit 3'],
        )
        self.assertEqual([e["status"] for e in self.events("exit")], [3])

    def test_more_than_a_window_of_data_each_way(self):
        # Input past the server's window (fake_sshd.WINDOW), and output past OpenSSH's client's (2 MB).
        data = os.urandom(3 * fake_sshd.WINDOW)
        p = self.ssh("cat", data)
        self.assertEqual((p.returncode, p.stdout == data), (0, True), self.said(p))
        size = 2_500_000
        p = self.ssh(f"yes 0123456789abcdef | head -c {size}")
        self.assertEqual(p.returncode, 0, self.said(p))
        self.assertEqual(p.stdout, (b"0123456789abcdef\n" * (size // 17 + 1))[:size])

    def test_ssh_keyscan_reads_its_one_host_key(self):
        p = subprocess.run(
            [
                "ssh-keyscan",
                "-T",
                "10",
                "-t",
                "ed25519,ecdsa,rsa",
                "-p",
                str(self.port),
                "127.0.0.1",
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=60,
        )
        found = [x for x in p.stdout.splitlines() if x and not x.startswith("#")]
        self.assertEqual(
            found, [f"[127.0.0.1]:{self.port} {self.public(self.host_key)}"], p.stderr
        )
        self.assertEqual(self.events("auth"), [])  # it never logs in

    @unittest.skipUnless(shutil.which("rsync"), "rsync is not installed here")
    def test_rsync_copies_into_the_login_s_home(self):
        src = self.d / "page.html"
        src.write_text("<p>the page</p>\n")
        p = subprocess.run(
            [
                "rsync",
                "-a",
                "--chmod=F644",
                "-e",
                shlex.join(self.ssh_words(self.login_key)),
                str(src),
                f"{LOGIN}@127.0.0.1:site/",
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        copy = self.home / "site" / "page.html"
        self.assertEqual(copy.read_text(), "<p>the page</p>\n")
        self.assertEqual(copy.stat().st_mode & 0o777, 0o644)

    def test_another_host_key_is_refused_before_any_login(self):
        # the server reads its host key at each connection: another one takes its place
        other = self.keygen("other_host_key")
        os.replace(other, self.host_key)
        os.replace(f"{other}.pub", f"{self.host_key}.pub")
        p = self.ssh("echo ran")
        self.assertEqual((p.returncode, p.stdout), (255, b""))
        self.assertIn("Host key verification failed", self.said(p))
        self.assertEqual(len(self.events("kex")), 1)
        self.assertEqual((self.events("auth"), self.events("exec")), ([], []))

    def test_another_key_or_login_is_refused(self):
        other = self.keygen("other_login_key")
        for p in (self.ssh("echo ran", key=other), self.ssh("echo ran", login="root")):
            self.assertEqual((p.returncode, p.stdout), (255, b""))
            self.assertIn("Permission denied (publickey)", self.said(p))
        auths = self.events("auth")
        self.assertTrue(auths)
        self.assertFalse(any(e["ok"] for e in auths), auths)
        self.assertEqual({e["user"] for e in auths}, {LOGIN, "root"})
        self.assertEqual(self.events("exec"), [])


if __name__ == "__main__":
    unittest.main()
