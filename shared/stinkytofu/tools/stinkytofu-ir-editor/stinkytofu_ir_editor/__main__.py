# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""python -m stinkytofu_ir_editor --bundles <dumpDir> --load-dir <loadDir>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .server import Workspace, WorkspaceError, make_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stinkytofu_ir_editor",
        description="Reorder the instructions of StinkyTofu IR checkpoint bundles in a browser.",
    )
    parser.add_argument(
        "--bundles",
        required=True,
        type=Path,
        help="the dump directory (--dump-st-ir / IRCheckpointDumpDir)",
    )
    parser.add_argument(
        "--load-dir",
        required=True,
        type=Path,
        help="where saved edits go; build again with --load-st-ir on it",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: 127.0.0.1, this machine only)",
    )
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument(
        "--stinkytofu-path",
        type=Path,
        help="directory holding the stinkytofu Python package, e.g. <build>/lib",
    )
    args = parser.parse_args(argv)

    if args.stinkytofu_path:
        sys.path.insert(0, str(args.stinkytofu_path.resolve()))
    try:
        workspace = Workspace(args.bundles, args.load_dir)
        _ = workspace.ir  # fail now, not on the first request
    except WorkspaceError as error:
        parser.error(str(error))

    server = make_server(workspace, args.host, args.port)
    print(
        f"stinkytofu IR editor: http://{args.host}:{server.server_address[1]}/  "
        f"({len(workspace.list_bundles())} bundles in {workspace.bundles_dir}, "
        f"saving to {workspace.load_dir})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
