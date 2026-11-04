# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Repro directories and ``aotriton/replay.py``, on CPU with a fake ``hipdnn_frontend``."""

import json
import re
import signal
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import hipdnn_sdpa
from hipdnn_sdpa import Dropout, Mask, Varlen, _problem, _repro, _runtime

contract = _runtime.contract
REPLAY = Path(__file__).resolve().parent.parent / "aotriton" / "replay.py"
GRAPH = {"fwd": '{"direction": "fwd"}', "bwd": '{"direction": "bwd"}'}
ENGINE = "hipkernel:flydsl_sdpa_cdna"


def record(prob, kernel="", ranked=("ENG_A", ENGINE), engine=ENGINE):
    return _runtime._Call(
        prob.direction,
        engine,
        kernel,
        ranked,
        GRAPH[prob.direction],
        _repro.problem_meta(prob),
        prob.tensors,
    )


def varlen_forward(layout):
    """A forward problem with dropout, a mask and varlen, padded or packed."""
    heads, dim = 2, 8
    seed = torch.tensor([7], dtype=torch.int64)
    offset = torch.tensor([9], dtype=torch.int64)
    dropout = Dropout(0.25, seed, offset)
    mask = Mask("tl", None, 0)
    if layout == "padded":
        q, k, v, o = (torch.randn(2, heads, 4, dim) for _ in range(4))
        stats = torch.zeros(2, heads, 4, 1)
        varlen = Varlen(
            torch.tensor([3, 2], dtype=torch.int32),
            torch.tensor([3, 2], dtype=torch.int32),
            None,
            None,
            4,
            4,
        )
    else:

        def packed(last):
            base = torch.randn(6, heads, last)
            return base.permute(1, 0, 2).unsqueeze(
                0
            )  # [1, H, T, last] over [T, H, last]

        q, k, v, o = (packed(dim) for _ in range(4))
        stats = torch.zeros(6, heads).permute(1, 0).unsqueeze(0).unsqueeze(-1)
        pos = torch.tensor([0, 3, 6], dtype=torch.int32)
        both = torch.tensor([3, 3], dtype=torch.int32)
        varlen = Varlen(both, both.clone(), pos, pos.clone(), 3, 3)
    return _problem.forward_problem(
        q, k, v, o, stats, 0.5, torch.randn(2, heads, 4, 4) if layout == "padded" else None,
        mask, dropout, varlen,
    )  # fmt: skip


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_save_repro_writes_every_role_and_the_scalars(tmp_path, layout):
    prob = varlen_forward(layout)
    _repro.save_repro(tmp_path, {"fwd": record(prob, kernel="k1")})
    saved = torch.load(tmp_path / "tensors.pt")
    outputs = torch.load(tmp_path / "outputs.pt")
    assert set(saved) | set(outputs) == set(prob.tensors)
    assert set(outputs) == {"o", "stats"}
    assert {"seed", "offset", "seq_len_q", "seq_len_kv"} <= set(saved)
    for role, tensor in {**saved, **outputs}.items():
        assert torch.equal(tensor, prob.tensors[role].detach().contiguous()), role
    if layout == "packed":
        assert any(r.startswith("ragged_offset_q_") for r in saved)
    call = json.loads((tmp_path / "meta.json").read_text())["calls"]["fwd"]
    assert (call["engine"], call["kernel"]) == (ENGINE, "k1")
    assert (call["scale"], call["dropout_p"]) == (0.5, 0.25)
    assert (call["mask_alignment"], call["mask_left"], call["mask_right"]) == (
        "tl",
        None,
        0,
    )
    assert (call["max_sq"], call["max_sk"]) == (
        (4, 4) if layout == "padded" else (3, 3)
    )
    assert call["uids"] == prob.uids()
    assert (tmp_path / "fwd.graph.json").read_text() == GRAPH["fwd"]


def test_a_backward_repro_keeps_the_forward_outputs_and_do_as_inputs(tmp_path):
    shapes = [(1, 2, 4, 8)] * 3 + [(1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 1)]
    q, k, v, o, do, stats = (torch.randn(*s) for s in shapes)
    dq, dk, dv = (torch.zeros(1, 2, 4, 8) for _ in range(3))
    fwd = _problem.forward_problem(q, k, v, o, stats, 0.5, None, None, None, None)
    bwd = _problem.backward_problem(
        q, k, v, o, do, stats, dq, dk, dv, 0.5, None, None, None, None, None
    )
    dq.fill_(2.0)  # what hipDNN wrote
    _repro.save_repro(tmp_path, {"fwd": record(fwd), "bwd": record(bwd)}, {"dq": dq})
    saved = torch.load(tmp_path / "tensors.pt")
    outputs = torch.load(tmp_path / "outputs.pt")
    assert set(saved) == {"q", "k", "v", "o", "do", "stats"}
    assert set(outputs) == {"dq", "dk", "dv"} and (outputs["dq"] == 2.0).all()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["directions"] == ["fwd", "bwd"]
    assert (tmp_path / "fwd.graph.json").exists() and (
        tmp_path / "bwd.graph.json"
    ).exists()


def test_saved_tensors_do_not_carry_their_whole_storage(tmp_path):
    big = torch.randn(1000, 2, 8)
    view = big[:4].permute(1, 0, 2).unsqueeze(0)  # a small window of a big allocation
    prob = _problem.forward_problem(
        view, view, view, torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 1), 0.5,
        None, None, None, None,
    )  # fmt: skip
    _repro.save_repro(tmp_path, {"fwd": record(prob)})
    loaded = torch.load(tmp_path / "tensors.pt")["q"]
    assert loaded.untyped_storage().nbytes() == loaded.numel() * loaded.element_size()
    assert (tmp_path / "tensors.pt").stat().st_size < 50_000


# ------------------------------------------------------------------------------ replay

FAKE_FRONTEND = textwrap.dedent(
    """
    import ctypes, json, os, signal, torch


    class _Ok:
        def is_bad(self):
            return False


    class HeuristicMode:
        FALLBACK = "fallback"


    class Handle:
        def __init__(self, stream=0):
            pass

        def get(self):
            return 1


    class _Config:
        def __init__(self, engine_id, engine_name):
            self.engine_id, self.engine_name = engine_id, engine_name


    class Graph:
        def from_json(self, handle, text):
            self.direction = json.loads(text)["direction"]
            return _Ok()

        def get_engine_configs(self, modes, handle):
            return [_Config(7, os.environ["FAKE_ENGINE"])]

        def create_execution_plan_ext(self, engine_id):
            with open(os.environ["FAKE_LOG"], "a") as log:
                log.write(f"{self.direction} {engine_id}\\n")
            return _Ok()

        def check_support(self):
            return _Ok()

        def build_plans(self):
            return _Ok()

        def get_execution_plan_engine_id(self):
            return 7

        def get_plan_name(self, handle):
            return os.environ["FAKE_ENGINE"]

        def get_workspace_size(self):
            return 0

        def execute(self, handle, variant_pack, workspace):
            if os.environ.get("FAKE_CRASH"):
                os.kill(os.getpid(), signal.SIGSEGV)
            results = torch.load(os.environ["FAKE_RESULTS"])[self.direction]
            for uid, src in results.items():
                ctypes.memmove(variant_pack[uid], src.data_ptr(), src.numel() * src.element_size())
            return _Ok()
    """
)


@pytest.fixture
def replay():
    sys.path.insert(0, str(REPLAY.parent))
    try:
        import replay
    finally:
        sys.path.remove(str(REPLAY.parent))
    return replay


@pytest.fixture
def fake_frontend(tmp_path, monkeypatch):
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "hipdnn_frontend.py").write_text(FAKE_FRONTEND)
    monkeypatch.setenv("PYTHONPATH", str(fakes))
    monkeypatch.setenv("FAKE_ENGINE", ENGINE)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "pins.log"))
    monkeypatch.delenv("FAKE_CRASH", raising=False)
    return tmp_path


def attention_case():
    q, k, v, do = (torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(4))
    scores = (q @ k.transpose(-1, -2)) * 0.5
    o = torch.softmax(scores, -1) @ v
    stats = torch.logsumexp(scores, -1, keepdim=True)
    dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
    detach = lambda t: t.detach().clone()  # noqa: E731
    return [detach(t) for t in (q, k, v, o, stats, do, dq, dk, dv)]


def repro_of_attention(directory, hipdnn_offset):
    """A fwd+bwd repro whose saved hipDNN outputs are the exact results; hipDNN 'returns' them
    shifted by ``hipdnn_offset`` per direction when the fake executes (see ``results``).
    """
    q, k, v, o, stats, do, dq, dk, dv = attention_case()
    fwd = _problem.forward_problem(q, k, v, o, stats, 0.5, None, None, None, None)
    bwd = _problem.backward_problem(
        q, k, v, o, do, stats, dq, dk, dv, 0.5, None, None, None, None, None
    )
    _repro.save_repro(directory, {"fwd": record(fwd), "bwd": record(bwd)})
    uids = {"fwd": fwd.uids(), "bwd": bwd.uids()}
    results = {
        "fwd": {
            uids["fwd"]["o"]: o + hipdnn_offset["fwd"],
            uids["fwd"]["stats"]: stats,
        },
        "bwd": {
            uids["bwd"][r]: t + hipdnn_offset["bwd"]
            for r, t in (("dq", dq), ("dk", dk), ("dv", dv))
        },
    }
    return results


def errors(text):
    return {
        (m["role"], m["against"]): float(m["value"])
        for m in re.finditer(
            r"(?P<role>\w+): max \|replay - (?P<against>saved hipDNN|reference)\| *= (?P<value>\S+)",
            text,
        )
    }


def test_replay_pins_the_engine_and_reports_the_error_against_saved_and_reference(
    fake_frontend, monkeypatch, capsys, replay
):
    results = repro_of_attention(fake_frontend / "repro", {"fwd": 0.25, "bwd": 0.5})
    torch.save(results, fake_frontend / "results.pt")
    monkeypatch.setenv("FAKE_RESULTS", str(fake_frontend / "results.pt"))
    assert (
        sys.modules.get("hipdnn_frontend") is None
    )  # the child, not this process, imports it
    status = replay.main([str(fake_frontend / "repro"), "--device", "cpu"])
    shown = capsys.readouterr().out
    assert status == 0, shown
    found = errors(shown)
    assert found[("o", "saved hipDNN")] == pytest.approx(
        0.25, abs=1e-5
    )  # saved o is exact
    assert found[("o", "reference")] == pytest.approx(0.25, abs=1e-4)
    assert (
        found[("stats", "saved hipDNN")] == 0 and found[("stats", "reference")] < 1e-5
    )
    assert found[("dq", "saved hipDNN")] == pytest.approx(0.5, abs=1e-5)
    assert found[("dk", "reference")] == pytest.approx(0.5, abs=1e-4)
    pins = (fake_frontend / "pins.log").read_text().split()
    assert pins == ["fwd", "7", "bwd", "7"]


def test_a_replay_that_crashes_is_reported_as_reproduced(
    fake_frontend, monkeypatch, capsys, replay
):
    repro_of_attention(fake_frontend / "repro", {"fwd": 0, "bwd": 0})
    monkeypatch.setenv("FAKE_CRASH", "1")
    status = replay.main([str(fake_frontend / "repro"), "--device", "cpu"])
    assert status == 1
    assert (
        f"reproduced: died with signal {int(signal.SIGSEGV)}" in capsys.readouterr().out
    )


def test_a_capture_directory_replays_without_a_saved_output(
    fake_frontend, monkeypatch, capsys, replay
):
    q, k, v, o, stats, *_ = attention_case()
    prob = _problem.forward_problem(q, k, v, o, stats, 0.5, None, None, None, None)
    monkeypatch.setenv(contract.REPRO_CAPTURE_ENV, str(fake_frontend / "root"))
    monkeypatch.setenv(contract.PASS_SLUG_ENV, "p")
    _repro.capture(prob, GRAPH["fwd"], ENGINE)
    (case,) = (fake_frontend / "root").iterdir()
    assert not (case / "outputs.pt").exists()
    torch.save(
        {"fwd": {prob.uids()["o"]: o + 0.125, prob.uids()["stats"]: stats}},
        fake_frontend / "results.pt",
    )
    monkeypatch.setenv("FAKE_RESULTS", str(fake_frontend / "results.pt"))
    assert replay.main([str(case), "--device", "cpu"]) == 0
    shown = capsys.readouterr().out
    assert "o: no saved hipDNN output" in shown
    assert errors(shown)[("o", "reference")] == pytest.approx(0.125, abs=1e-4)


@pytest.mark.gpu
def test_replay_reproduces_a_saved_output(cuda_device, tmp_path):
    q, k, v = (
        torch.randn(1, 2, 128, 128, device=cuda_device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    o = torch.empty_like(q)
    stats = torch.empty(1, 2, 128, 1, device=cuda_device, dtype=torch.float32)
    hipdnn_sdpa.sdpa_fwd(q, k, v, o=o, stats=stats, scale=0.1)
    hipdnn_sdpa.save_repro(tmp_path, {"fwd": hipdnn_sdpa.last_call("fwd")})
    import subprocess

    done = subprocess.run(
        [sys.executable, str(REPLAY), str(tmp_path)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert errors(done.stdout)[("o", "saved hipDNN")] < 0.05
