# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the WP4h tail provider (``kernels.gfx950.qwen38_mono.truemono_tail``): which widths it serves,
its knobs, that every launch argument it sets is one of P9's kernel parameters, the fallbacks (nothing written,
the hybrid serves), and the step-0 forward override of ``propose`` (one call, restored). The launches themselves
are gated on hardware (ATOM's check mode + the deep check)."""

import ast
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from kernels.gfx950.qwen38_mono import truemono_tail as T
from kernels.gfx950.qwen38_mono.probes import p9_tail as P


@dataclass(frozen=True)
class Req:
    requests: int
    k: int

    @property
    def variant(self):
        return f"r{self.requests * (self.k + 1)}"


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    # never compile or allocate on the laptop GPU from a unit test
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for k in ("K3", "K0", "P9", "MUTANT", "DEEPCHECK", "STATS", "DRAFT", "P9D", "HOSTUS"):
        monkeypatch.delenv(f"{T.ENV}_{k}", raising=False)


class FakeHybrid:
    def __init__(self, runner=None):
        self.runner = runner or SimpleNamespace(model=SimpleNamespace(), drafter=None)
        self.calls = []

    def verify(self, step):
        self.calls.append("verify")
        return "hyb-verify"

    def sample(self, step):
        self.calls.append("sample")
        return "hyb-sample"

    def propose(self, step, **kw):
        self.calls.append("propose")
        return "hyb-propose"


def test_settings_defaults():
    s = T.settings()
    assert s == {
        "k3": "p9",
        "k0": "argmax",
        "p9": {},
        "draft": "hybrid",
        "p9d": {},
        "mutant": "none",
        "deepcheck": 0,
        "hostus": 0,
        "stats": 0,
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("K3", "maybe"),
        ("K0", "logits"),
        ("MUTANT", "nope"),
        ("P9", "[1]"),
        ("DEEPCHECK", "x"),
        ("DRAFT", "fast"),
        ("P9D", "3"),
    ],
)
def test_settings_refuse(monkeypatch, name, value):
    monkeypatch.setenv(f"{T.ENV}_{name}", value)
    with pytest.raises(ValueError):
        T.settings()


def test_provider_widths(monkeypatch):
    h = FakeHybrid()
    assert isinstance(T.provider(Req(1, 3), h), T.P9VerifyTail)
    assert isinstance(T.provider(Req(1, 0), h), T.ArgmaxTail)
    for r in (Req(2, 3), Req(4, 3), Req(2, 0), Req(4, 0), Req(1, 1), Req(1, 2)):
        assert T.provider(r, h) is None, r
    monkeypatch.setenv(f"{T.ENV}_K3", "off")
    monkeypatch.setenv(f"{T.ENV}_K0", "off")
    assert T.provider(Req(1, 3), h) is None
    assert T.provider(Req(1, 0), h) is None


def test_p9_spec_is_best_spec_plus_overrides(monkeypatch):
    t = T.provider(Req(1, 3), FakeHybrid())
    assert t.spec == P.best_spec("verify", **T.SERVED_VERIFY) and t.spec.hdelay == 0
    monkeypatch.setenv(f"{T.ENV}_P9", '{"hdelay": 4, "mutant": "accept_ooo"}')
    t = T.provider(Req(1, 3), FakeHybrid())
    assert t.spec == P.best_spec("verify", hdelay=4, mutant="accept_ooo")
    monkeypatch.setenv(f"{T.ENV}_P9", '{"hdelay": 99}')
    with pytest.raises(ValueError):
        T.provider(Req(1, 3), FakeHybrid())


def _set_names():
    """Every keyword of ``self.kern.set(...)`` in the module (a typo there is a KeyError at serve time)."""
    tree = ast.parse(inspect.getsource(T))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set":
            v = node.func.value
            if (isinstance(v, ast.Attribute) and v.attr == "kern") or (isinstance(v, ast.Name) and v.id == "kern"):
                names |= {kw.arg for kw in node.keywords if kw.arg}
    # the draft chain's static arguments: a dict(...) passed as kern.set(**args)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "dict":
                names |= {kw.arg for kw in node.value.keywords if kw.arg}
    return names


def test_launch_arguments_are_p9_parameters():
    names = _set_names() | {"vmap"}  # the draft chain sets vmap through args["vmap"] (vsub builds only)
    sig = {n for n, _ in P.p9_signature(P.best_spec("draft", rows_in=1, vsub=98304))}
    assert names, "no kern.set calls found"
    assert names <= sig, names - sig
    # every parameter is set somewhere (bind, the draft chain's static args, or per step)
    assert sig - names == set(), sig - names


def test_argmax_tail_falls_back_without_rocke_head():
    lm = SimpleNamespace(lm_head=SimpleNamespace(rk=None))
    runner = SimpleNamespace(model=SimpleNamespace(model=lm))
    h = FakeHybrid(runner)
    t = T.provider(Req(1, 0), h)
    assert t.owns_logits
    assert t.sample(SimpleNamespace(bs=1)) == "hyb-sample"
    assert t.off_why and "rocKE" in t.off_why
    assert t.sample(SimpleNamespace(bs=1)) == "hyb-sample"
    assert h.calls == ["sample", "sample"]


def test_p9_tail_off_serves_hybrid_and_drafter(monkeypatch):
    calls = []
    drafter = SimpleNamespace(propose=lambda **kw: calls.append(("propose", kw)) or "drafts")
    runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace()), drafter=drafter)
    h = FakeHybrid(runner)
    t = T.provider(Req(1, 3), h)
    monkeypatch.setattr(t, "_compile", lambda: None)
    assert t.verify(SimpleNamespace(bs=1)) == "hyb-verify"
    assert t.off_why and not t.pending
    assert t.propose(SimpleNamespace(), a=1) == "drafts"
    assert calls == [("propose", {"a": 1})]


class FakeDraftRunner:
    def forward(self, input_ids, hidden_states):
        return ("class-forward", input_ids.numel())


def test_propose_overrides_exactly_one_step0_forward(monkeypatch):
    dr = FakeDraftRunner()
    seen = []

    def propose(**kw):
        seen.append(dr.forward(torch.zeros(4), None))  # step 0 (4 rows): the override
        seen.append(dr.forward(torch.zeros(1), None))  # step 1
        seen.append(dr.forward(torch.zeros(4), None))  # a second 4-row call is not step 0
        return "drafts"

    runner = SimpleNamespace(drafter=SimpleNamespace(propose=propose))
    t = T.provider(Req(1, 3), FakeHybrid(runner))
    t.dr = dr
    t.pending = True
    monkeypatch.setattr(t, "_step0", lambda ids, hid, deep: ("step0", ids.numel()))
    assert t.propose(SimpleNamespace()) == "drafts"
    assert seen == [("step0", 4), ("class-forward", 1), ("class-forward", 4)]
    assert "forward" not in dr.__dict__  # restored
    assert not t.pending
    # not pending (the hybrid verified this step): the drafter runs unchanged
    seen.clear()
    assert t.propose(SimpleNamespace()) == "drafts"
    assert seen == [("class-forward", 4), ("class-forward", 1), ("class-forward", 4)]


def test_propose_restores_on_error(monkeypatch):
    dr = FakeDraftRunner()

    def propose(**kw):
        raise RuntimeError("boom")

    t = T.provider(Req(1, 3), FakeHybrid(SimpleNamespace(drafter=SimpleNamespace(propose=propose))))
    t.dr, t.pending = dr, True
    with pytest.raises(RuntimeError):
        t.propose(SimpleNamespace())
    assert "forward" not in dr.__dict__


@pytest.mark.parametrize("req", [Req(1, 3), Req(1, 0)])
def test_tails_have_the_hook_interface(req):
    # ATOM's TailHook.tail_for: an MTP width's tail needs verify, sample and propose (k = 0: sample)
    t = T.provider(req, FakeHybrid())
    for name in ("verify", "sample", "propose"):
        assert callable(getattr(t, name, None)), name
    assert t.owns_logits is True


def test_layer_passthrough_returns_the_hybrid_launcher():
    launch = object()
    assert T.layer_passthrough(SimpleNamespace(kind="qsa"), launch) is launch


def test_mutants_listed():
    assert T.MUTANTS[0] == "none" and len(set(T.MUTANTS)) == len(T.MUTANTS)


# ---- P9's default-off knobs for the served draft chain (``vsub``: N8 subset head + id map; ``pbump``: mRoPE rows)
def test_p9_draft_knobs_valid_and_named():
    a = P.best_spec("draft", rows_in=4, reject=1, vsub=98304, pbump=3)
    assert P.is_valid_p9(a)[0]
    assert "vs98304" in a.kernel_name() and "pb3" in a.kernel_name()
    assert a.head_spec().n == 98304 and P.best_spec("draft").head_spec().n == P.VOCAB
    names = [n for n, _ in P.p9_signature(a)]
    assert names[-1] == "vmap" and names[:-1] == [n for n, _ in P.p9_signature(P.best_spec("draft"))]
    assert "vmap" not in [n for n, _ in P.p9_signature(None)]


@pytest.mark.parametrize(
    "kw",
    [
        {"mode": "verify", "vsub": 98304},
        {"mode": "draft", "vsub": 98300},
        {"mode": "draft", "vsub": P.VOCAB},
        {"mode": "draft", "vsub": 98304, "fold": "fused"},
        {"mode": "verify", "pbump": 3},
        {"mode": "draft", "pbump": 4},
        {"mode": "verify", "tokonly": 1},
        {"mode": "draft", "tokonly": 1, "fold": "fused"},
    ],
)
def test_p9_draft_knobs_refused(kw):
    mode = kw.pop("mode")
    spec = P.best_spec(mode, rows_in=4, **kw)
    assert not P.is_valid_p9(spec)[0]


def test_p9_vsub_build_lowers_with_the_map():
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    sp = P.best_spec("draft", rows_in=1, vsub=98304, pbump=3, dbg=1, tokonly=1)
    assert P.is_valid_p9(sp)[0] and sp.kernel_name().endswith("_tk_pyonly")
    t = lower_kernel_to_llvm(P.build_p9(sp), arch="gfx950")
    assert "vmap" in t
