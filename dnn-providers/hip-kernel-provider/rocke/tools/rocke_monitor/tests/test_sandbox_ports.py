# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""sandbox.free_port: a port for a server under test that no other test process gives out and no client connection
takes meanwhile, whatever else runs on the machine (two suites side by side, each in several processes).
"""

import pathlib
import socket
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import sandbox  # noqa: E402

ANOTHER_PROCESS = (
    "import sys; sys.path.insert(0, sys.argv[1]); import sandbox; "
    "b = sandbox.port_block(); print(b[2], sandbox.free_port())"
)


class TestFreePort(unittest.TestCase):
    def test_a_free_port_is_outside_the_ephemeral_range_in_this_process_s_block_and_free(
        self,
    ):
        """free_port gives a port outside the kernel's ephemeral range, in this process's block past its first port
        (the one kept bound to claim it), and nothing has bound it."""
        port = sandbox.free_port()
        low, high = sandbox.ephemeral_ports()
        self.assertFalse(low <= port <= high, (port, low, high))
        block = sandbox.port_block()
        first = block[2]
        self.assertTrue(first < port < first + sandbox.PORT_BLOCK, (first, port))
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))

    def test_another_test_process_gets_a_block_of_its_own(self):
        """While this process keeps its block, another process's free_port gives a port of another block."""
        first = sandbox.port_block()[2]
        out = subprocess.run(
            [sys.executable, "-B", "-c", ANOTHER_PROCESS, str(ROOT / "tests")],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        other_first, other_port = map(int, out.split())
        self.assertNotEqual(other_first, first)
        self.assertFalse(
            first <= other_port < first + sandbox.PORT_BLOCK, (first, other_port)
        )
        self.assertTrue(other_first < other_port < other_first + sandbox.PORT_BLOCK)

    def test_a_port_in_use_is_passed_over_and_the_ports_go_round_the_block(self):
        """A port of the block that something has bound is passed over; once the block's end is reached, its ports
        are given again from its start, each checked free again. The test binds the block's first port that nothing
        has bound (a server of an earlier test may still hold one), and skips when every one is.
        """
        block = sandbox.port_block()
        first, end = block[2], block[4]
        with socket.socket() as s:
            for taken in range(first + 1, end):
                try:
                    s.bind(("127.0.0.1", taken))
                    break
                except OSError:
                    continue
            else:
                self.skipTest(
                    f"every port of this process's block {first + 1}-{end - 1} is in use"
                )
            given = [sandbox.free_port() for _ in range(sandbox.PORT_BLOCK - 1)]
        self.assertNotIn(taken, given)
        self.assertTrue(
            all(first < p < first + sandbox.PORT_BLOCK for p in given), given
        )
        self.assertLess(len(set(given)), len(given))


if __name__ == "__main__":
    unittest.main()
