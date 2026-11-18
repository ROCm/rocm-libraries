# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runs superviseClient in its own process so that tests can signal the supervisor.

Usage: supervisor_driver.py <json file>. The file holds the command, cwd, db and grace_seconds
of the superviseClient call. The watchdog is off, so only a signal ends the supervision. The exit
code is 128 + signal for SIGTERM and SIGHUP; SIGINT ends it by KeyboardInterrupt.
"""

import json
import signal
import sys

from tensilelite.ClientSupervisor import Watchdog, superviseClient

DRIVER_POLL_SECONDS = 0.1


def main():
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    # Python leaves SIGINT ignored if the parent did, which would hide the KeyboardInterrupt.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    result = superviseClient(
        config["command"],
        config["cwd"],
        config["db"],
        Watchdog(0, DRIVER_POLL_SECONDS, config["grace_seconds"]),
        False,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
