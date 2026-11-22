# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``Runtime.stream_wait_event``: one stream waits, on the device, for an event
recorded on another. Runs on a GPU and skips without one.
"""

from __future__ import annotations

import time

from rocke.runtime.hip_module import Runtime
from rocke.runtime.stall_gate import StallGate
from rocke.tests.gpu import current_device_arch


def test_gpu_the_waiting_stream_holds_until_the_event_completes():
    current_device_arch()
    runtime = Runtime()
    held, waiting = runtime.create_stream(), runtime.create_stream()
    fork, join = runtime.event(), runtime.event()
    try:
        with StallGate() as gate:
            gate.arm(held)
            fork.record(held)
            runtime.stream_wait_event(waiting, fork)
            join.record(waiting)
            # The host does not block, and nothing behind the wait runs while
            # the stream the event was recorded on is held.
            time.sleep(0.05)
            assert not join.query()
            gate.release()
            join.synchronize()
            assert fork.query()
    finally:
        for event in (fork, join):
            event.destroy()
        for stream in (held, waiting):
            runtime.destroy_stream(stream)
