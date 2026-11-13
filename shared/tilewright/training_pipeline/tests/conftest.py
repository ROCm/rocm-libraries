# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pytest

PIPELINE_DIR = Path(__file__).resolve().parent.parent
for _p in (PIPELINE_DIR, PIPELINE_DIR / "stages", PIPELINE_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_PROBE = {"allowed": False}
# The device lists of the HIP runtime, which hide the GPUs from every test
# unless its world is a fake GPU host (`strict_sysfs`).
DEVICE_LIST_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def _set_private_gpu_state(mp: pytest.MonkeyPatch, base: Path) -> None:
    """Leases, an empty sysfs and the CPU and process readings of the
    quiet-host gate and the ttm check in `base`, no amd-smi, no run's guard
    and a node fault store of its own."""
    (base / "sys").mkdir()
    (base / "proc").mkdir()
    mp.setenv("TILEWRIGHT_GPU_LEASE_DIR", str(base / "leases"))
    mp.setenv("TILEWRIGHT_NODE_FAULTS", str(base / "node_faults"))
    mp.setenv("TILEWRIGHT_SYSFS_ROOT", str(base / "sys"))
    mp.setenv("TILEWRIGHT_AMD_SMI", "")
    mp.setenv("TILEWRIGHT_PROC_ROOT", str(base / "proc"))
    mp.delenv("TILEWRIGHT_GUARD_DIR", raising=False)
    mp.delenv("TILEWRIGHT_GUARD_SESSION", raising=False)


@pytest.fixture(autouse=True, scope="session")
def private_host(tmp_path_factory):
    """Leases, sysfs, amd-smi, the CPU and process readings of the
    quiet-host gate and the ttm check, and the run guard of the tests' own
    for the whole session, no HIP read of the device values
    (TILEWRIGHT_DEVICE_PROBE empty) and the GPUs hidden from the runtime
    (DEVICE_LIST_VARS at -1, which a test's fake GPU host undoes), whatever
    the environment the suite starts in: module-scoped worlds, set up before
    any test's own fixtures,
    run the orchestrator, whose preflight would otherwise read the host's
    KFD topology, build the device probe and run it on a GPU of the host
    under a lease in the host's lease directory, and run amd-smi. Yields the
    directory of that state."""
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("host_state")
    _set_private_gpu_state(mp, base)
    mp.setenv("TILEWRIGHT_DEVICE_PROBE", "")
    for var in DEVICE_LIST_VARS:
        mp.setenv(var, "-1")
    mp.delenv("TILEWRIGHT_GUARD_ROOT", raising=False)
    yield base
    mp.undo()


def _compiled(path: Path) -> bool:
    """Whether `path` is a compiled program, as the probe built from
    tools/device_props.cpp is; the tests' fake probes are scripts."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


@pytest.fixture(autouse=True, scope="session")
def no_device_probe():
    """Fail the test, or the fixture of any scope, that would build the
    device probe (tools/device_props.cpp) or run a compiled one in the test
    process, unless the test uses `allow_device_probe`: the probe runs on a
    GPU of the host."""
    from lib import device_values as dv

    build, run = dv.build_command, dv._run_probe

    def guarded_build(*args, **kwargs):
        if not _PROBE["allowed"]:
            pytest.fail(
                f"the test would build the device probe (tools/"
                f"{dv.PROBE_SOURCE.name}), which runs on a GPU of the host: name "
                f"a fake probe in {dv.PROBE_VAR}, leave a fake sysfs in "
                f"TILEWRIGHT_SYSFS_ROOT, or use the allow_device_probe fixture "
                f"with a compiler of the test's own"
            )
        return build(*args, **kwargs)

    def guarded_run(probe, gpu, env):
        if not _PROBE["allowed"] and _compiled(probe):
            pytest.fail(
                f"the test would run the compiled device probe {probe} on GPU "
                f"{gpu} of the host: name a fake probe in {dv.PROBE_VAR}, or use "
                f"the allow_device_probe fixture"
            )
        return run(probe, gpu, env)

    mp = pytest.MonkeyPatch()
    mp.setattr(dv, "build_command", guarded_build)
    mp.setattr(dv, "_run_probe", guarded_run)
    yield
    mp.undo()


@pytest.fixture
def allow_device_probe(monkeypatch):
    """Let the test build the device probe and run a compiled one, with a
    compiler and a probe of its own: the probe runs on a GPU of the host."""
    monkeypatch.setitem(_PROBE, "allowed", True)


@pytest.fixture(autouse=True)
def private_gpu_state(monkeypatch, tmp_path_factory):
    """Leases, sysfs, amd-smi, the CPU and process readings of the
    quiet-host gate and the ttm check, and the run guard of the tests' own:
    no test takes or sees a lease of the host, reads its GPUs or load, runs
    amd-smi, or writes into another run's guard."""
    _set_private_gpu_state(monkeypatch, tmp_path_factory.mktemp("gpu_state"))


@pytest.fixture
def strict_sysfs(private_gpu_state, monkeypatch, tmp_path_factory) -> Path:
    """The test's own fake sysfs of GPUs that load gfx1250-strict
    (tests/gpu_fakes.py `write_strict_sysfs`) instead of the empty one, for
    the tests that run the orchestrator for that arch, as the audit worlds
    do: its preflight reads from the KFD topology which library subtree
    each of the run's devices loads, so the tree's GPUs are not hidden
    from the runtime (ROCR_VISIBLE_DEVICES unset). Returns the tree's root."""
    import gpu_fakes as gf

    root = gf.write_strict_sysfs(tmp_path_factory.mktemp("strict_sys"))
    monkeypatch.setenv("TILEWRIGHT_SYSFS_ROOT", str(root))
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    return root


@pytest.fixture
def pipeline_dir() -> Path:
    return PIPELINE_DIR


@pytest.fixture
def config_env(monkeypatch, tmp_path):
    """The environment variables the shipped configs reference."""
    for name in (
        "TILEWRIGHT_BENCH_ROOT",
        "TILEWRIGHT_DEPLOY_ROOT",
        "TILEWRIGHT_DATASETS",
        "TILEWRIGHT_AUDIT_ROOT",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return tmp_path


def write_bundle(
    train_dir: Path,
    labels: Iterable[str],
    *,
    seed: int = 0,
    dims: Sequence[int] = (4, 6, 5),
    signatures: Optional[Dict[str, list]] = None,
    names: Optional[Dict[str, list]] = None,
) -> Path:
    """A small trained-bundle `models.pt` plus `cells.json` with random
    weights for `labels`."""
    import torch

    from lib import features as fs
    from lib import mlrec

    q = fs.query_feature_names()
    i = fs.item_feature_names()
    x = fs.interaction_feature_names()
    embed, hidden, inter = dims
    models = {}
    for n, label in enumerate(sorted(labels)):
        g = torch.Generator().manual_seed(seed * 1000 + n)
        shapes = mlrec.tensor_shapes(len(q), len(i), len(x), embed, hidden, inter)
        sd = {k: torch.randn(*s, generator=g) for k, s in shapes.items()}
        sd["temperature"] = torch.tensor([1.0])
        models[label] = {
            "state_dict": sd,
            "embed_dim": embed,
            "hidden_dim": hidden,
            "inter_hidden": inter,
            "feature_norms": {
                "q_mean": [0.0] * len(q),
                "q_std": [1.0] * len(q),
                "i_mean": [0.0] * len(i),
                "i_std": [1.0] * len(i),
                "x_mean": [0.0] * len(x),
                "x_std": [1.0] * len(x),
            },
            "smart_k_signatures": (signatures or {}).get(
                label, [[128, 128, 64, 16, 16, 128, 0, 0]]
            ),
        }
    bundle = {"q_names": q, "i_names": i, "x_names": x, "models": models}
    bundle.update(names or {})
    train_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, train_dir / "models.pt")
    (train_dir / "cells.json").write_text(
        json.dumps({"feature_names_hash": fs.feature_names_hash(), "cells": []})
    )
    return train_dir / "models.pt"


@pytest.fixture
def make_bundle():
    return write_bundle


@pytest.fixture
def tdm_rule(monkeypatch):
    """The gfx1250 fakes under the identity rule `tdm_th_v1`, which every
    record written before `cd_th_v1` follows: the TDM masking and the alias
    groups the tests that use it pin."""
    from lib import dat

    monkeypatch.setattr(dat, "CLUSTER_RULE_ARCHS", ())
