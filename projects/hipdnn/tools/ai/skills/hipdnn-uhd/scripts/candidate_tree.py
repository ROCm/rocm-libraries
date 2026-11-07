#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A descriptor tree with candidate models installed, and a command run against it.

Standard library only, plus `python3 -m uhd_gen promote` for a descriptor engine.

Exercising a model the build does not hold -- collecting an engine's L1 under the L2 it will
be installed with, validating a candidate -- needs the build's descriptor tree with those
models in place of the incumbent ones, as the ONLY tree the runtime reads. `HIPDNN_DESCRIPTOR_DIR`
replaces the search roots rather than adding one, so the tree must be the whole per-arch root
that owns `kpack/`: a copy of one engine's folder leaves every kernel unresolvable (every
configuration then reports `config_not_applicable`).

So: a real copy of the root, symlinks dereferenced. The ingestor resolves a kernel's kpack
library and refuses one that lands outside the descriptor tree (IngestorKernelCode.hpp),
which a symlink back into the build always does. Then per model

* **opaque** (an engine binding its model by an id declared in provider code): every document
  carrying the declared id is removed and the candidate dropped in. A second document under
  an id the build already defines is refused by the loader, and the build's one stands --
  which is also why `HIPDNN_DESCRIPTOR_RUNTIME_DIR` cannot test a candidate;
* **descriptor** (an engine binding through its UED's role map): `uhd_gen promote` installs
  the model and rebinds the role map -- only the entry of the model's own metric is replaced.

Then the command after `--` runs with `HIPDNN_DESCRIPTOR_DIR` on the copy, and no other
descriptor root (engine_run.py). With `--cache-dir` it also gets its own `HIPDNN_CACHE_DIR`: a
benchmark winner record there takes precedence over any heuristic
(KernelIngestorStateManager.hpp), so a scratch cache is what makes a catalog ranker the thing
actually exercised.

    candidate_tree.py --root <per-arch descriptor root> --farm <new dir>
        [--opaque <declared id> <model dir>] [--promote <engine> <role> <arch> <model dir>]
        [--cache-dir <new dir>] [--feature-evaluator <hipdnn_uhd_features>]
        [--record installed.json] [-- <command> ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_run  # noqa: E402  (beside this script)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(path: Path) -> dict:
    try:
        return _load(path)
    except (OSError, ValueError):
        return {}


def artifact_of(document: dict) -> str | None:
    """The file a UHD's adapter loads beside it, if any."""
    adapter = document.get("adapter")
    body = document.get(adapter) if isinstance(adapter, str) else None
    if not isinstance(body, dict):
        return None
    return body.get("library" if adapter == "custom_library" else "artifact")


def build_farm(root: Path, farm: Path) -> None:
    if farm.exists():
        raise SystemExit(f"{farm} exists; a candidate tree is built once")
    farm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root.resolve(), farm, symlinks=False)


def carrying(farm: Path, uhd_id: str) -> list[Path]:
    """Every UHD document in the tree whose id is `uhd_id`."""
    return [
        p
        for p in sorted(farm.rglob("*.uhd.json"))
        if str(_safe(p).get("id", "")).lower() == uhd_id
    ]


def install_opaque(farm: Path, candidate_dir: Path, declared_id: str) -> dict:
    """The candidate as the only document carrying `declared_id`."""
    removed = []
    for path in carrying(farm, declared_id):
        artifact = artifact_of(_load(path))
        path.unlink()
        removed.append(str(path.relative_to(farm)))
        if artifact:
            installed = path.parent / artifact
            if farm.resolve() not in installed.resolve().parents:
                raise SystemExit(
                    f"descriptor artifact escapes the candidate tree: {artifact}"
                )
            if installed.is_file():
                installed.unlink()
                removed.append(str(installed.relative_to(farm)))
    descriptors = sorted(candidate_dir.glob("*.uhd.json"))
    if len(descriptors) != 1:
        raise SystemExit(
            f"{candidate_dir} must hold exactly one *.uhd.json, found {len(descriptors)}"
        )
    destination = farm / f"candidate_{declared_id[:8]}"
    destination.mkdir()
    shutil.copy2(descriptors[0], destination / descriptors[0].name)
    artifact = artifact_of(_load(descriptors[0]))
    if artifact:
        shutil.copy2(candidate_dir / artifact, destination / artifact)
    return {
        "kind": "opaque",
        "declared_id": declared_id,
        "removed": removed,
        "installed": str((destination / descriptors[0].name).relative_to(farm)),
    }


def install_promoted(
    farm: Path,
    candidate_dir: Path,
    engine: str,
    role: str,
    arch: str,
    feature_evaluator: str | None = None,
) -> dict:
    """`uhd_gen promote` the candidate into the copy: installed, and bound in the engine's UED.

    `feature_evaluator` is the build's `hipdnn_uhd_features`; without it promote looks the
    evaluator up itself (`HIPDNN_UHD_FEATURE_EVALUATOR`, a build under the checkout, PATH).
    """
    ueds = [p for p in farm.rglob("*.ued.json") if _safe(p).get("name") == engine]
    if len(ueds) != 1:
        raise SystemExit(
            f"{len(ueds)} UED(s) named {engine} under {farm}; need exactly one"
        )
    command = [
        sys.executable,
        "-m",
        "uhd_gen",
        "promote",
        "--model-dir",
        str(candidate_dir),
        "--descriptor-tree",
        str(farm),
        "--engine",
        engine,
        "--role",
        role,
        "--arch",
        arch,
    ]
    if feature_evaluator:
        command += ["--feature-evaluator", feature_evaluator]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"promote into the candidate tree failed:\n{proc.stderr[-3000:]}"
        )
    return {
        "kind": "descriptor",
        "engine": engine,
        "role": role,
        "arch": arch,
        "ued": str(ueds[0].relative_to(farm)),
        "roles_after": {
            k: v
            for k, v in _load(ueds[0]).items()
            if k in ("sort_kernel_catalog", "predict_engine")
        },
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", required=True, help="The build's per-arch descriptor root"
    )
    parser.add_argument(
        "--farm", required=True, help="Where to build the candidate tree"
    )
    parser.add_argument(
        "--opaque",
        nargs=2,
        action="append",
        default=[],
        metavar=("ID", "DIR"),
        help="Install DIR's model as the only document carrying declared ID",
    )
    parser.add_argument(
        "--promote",
        nargs=4,
        action="append",
        default=[],
        metavar=("ENGINE", "ROLE", "ARCH", "DIR"),
        help="Promote DIR's model into ENGINE's UED under ROLE/ARCH",
    )
    parser.add_argument(
        "--cache-dir", help="A scratch HIPDNN_CACHE_DIR, so no winner record decides"
    )
    parser.add_argument(
        "--feature-evaluator",
        help="The build's hipdnn_uhd_features, for --promote",
    )
    parser.add_argument("--record", help="Write what was installed here (JSON)")
    args = parser.parse_args(argv)

    farm = Path(args.farm)
    build_farm(Path(args.root), farm)
    installed = [install_opaque(farm, Path(d), i.lower()) for i, d in args.opaque]
    installed += [
        install_promoted(farm, Path(d), e, r, a, args.feature_evaluator)
        for e, r, a, d in args.promote
    ]
    record = {"root": args.root, "farm": str(farm), "installed": installed}
    if args.record:
        Path(args.record).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(record), flush=True)  # before the command's own output
    if not command:
        return 0
    env = engine_run.isolated_env(farm)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        env["HIPDNN_CACHE_DIR"] = args.cache_dir
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
