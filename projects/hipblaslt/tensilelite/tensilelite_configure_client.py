# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure the native client used by the active TensileLite installation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from _tensilelite_client_binding import (
    ClientBindingError,
    binding_path,
    current_installation,
    read_binding,
    validate_client,
)


def configure(client: Path, *, ensure: bool = False) -> Path:
    installation = current_installation()
    client = Path(os.path.abspath(client.expanduser()))
    if ensure and read_binding(installation) == client:
        return client
    validate_client(client, installation.version)
    destination = binding_path(installation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as temporary:
        json.dump(str(client), temporary)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return client


def reset() -> None:
    binding_path(current_installation()).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--client", type=Path, help="absolute tensilelite-client executable")
    action.add_argument("--ensure-client", type=Path, help=argparse.SUPPRESS)
    action.add_argument("--reset", action="store_true", help="remove this installation's binding")
    args = parser.parse_args(argv)
    try:
        if args.reset:
            reset()
        else:
            configure(args.client or args.ensure_client, ensure=args.ensure_client is not None)
    except ClientBindingError as exc:
        parser.exit(1, f"tensilelite-configure-client: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
