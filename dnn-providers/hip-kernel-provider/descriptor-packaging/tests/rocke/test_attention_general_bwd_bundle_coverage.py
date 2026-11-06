# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every SdpaBackward integration bundle case has a verdict from hipkernel:AttentionGeneralBwd.

The engine's verdict on a hipDNN graph is rocKE's backward predicate and plan applied to
the graph's lowering (``rocke/library/dispatch/attention_bwd/hipdnn.py``); the C++ restates
both and replays them against a corpus seeded from these same bundles. This walks every
single-node SdpaBackward bundle graph of every tier, templates and sweeps included, on
both shipped architectures, and checks:

- the lowering reads every well-formed bundle graph: the only LOWERING verdicts are on
  graphs that are malformed as hipDNN graphs (both causal flags set, or operand dims that
  disagree);
- an accepted case's plan is a function of the lowered request alone: writing the request
  back out as a graph and reading it again gives the same plan;
- the support claims agree with the verdicts: a case claimed for this engine on an
  architecture is one the engine accepts there. A lane enforces only the claims of the
  engine it pins, so a case may be claimed for both this engine and the ASM backward
  engine without changing the ASM outcome.

The rocKE sources are imported from the checkout. This is a rocKE test: it runs only in
a rocKE-enabled build, where unimportable sources fail rather than skip.
"""

import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

HKP = Path(__file__).resolve().parents[3]
TOOLS = HKP / "descriptor-packaging" / "tools"
BUNDLES = HKP.parent / "integration-tests" / "integration-test-bundles"
ENGINE = "hipkernel:AttentionGeneralBwd"
ASM_ENGINE = "ASM_SDPA_ENGINE"
ARCHES = ("gfx942", "gfx950")
FULL_CUS = {"gfx942": 304, "gfx950": 256}

# Lowering details a bundle graph may legitimately draw: graphs that are malformed as
# hipDNN graphs, which the shared bundles carry to exercise other engines' refusals.
MALFORMED = (
    re.compile(r"causal_mask and causal_mask_bottom_right are mutually exclusive"),
    re.compile(r"^\w+ dims \(.*\), want [(\[]"),
)


@pytest.fixture(scope="module")
def corpus_tool():
    sys.path.insert(0, str(TOOLS))
    try:
        return importlib.import_module("general_bwd_corpus")
    except ImportError as exc:
        pytest.fail(f"rocKE sources not importable: {exc}")
    finally:
        sys.path.remove(str(TOOLS))


@pytest.fixture(scope="module")
def bundle_cases(corpus_tool):
    skipped = Counter()
    # As the hipDNN JSON reader sees them: float32 scalars, required root keys.
    cases = [
        (cid, corpus_tool.normalize(graph))
        for cid, graph in corpus_tool.bundle_graphs(skipped)
    ]
    assert cases, f"no single-node SdpaBackward bundle graph under {BUNDLES}"
    return cases


def _verdict(tool, graph, arch):
    expect, _request = tool.outcome(graph, arch, FULL_CUS[arch])
    return expect


@pytest.fixture(scope="module")
def verdicts(corpus_tool, bundle_cases):
    """{arch: {case id: expect}} for every bundle case."""
    out = {}
    for arch in ARCHES:
        out[arch] = {
            cid: _verdict(corpus_tool, graph, arch) for cid, graph in bundle_cases
        }
    return out


def _case_key(cid):
    """(bundle directory relative to the bundle root, case id) of a corpus seed id."""
    rel = cid[len("bundle:") :]
    directory, _, case = rel.rpartition("/")
    return directory, case


def _claims():
    """[(bundle directory, engine, arch, case id)] of every SdpaBackward claim."""
    out = []
    for support in sorted(BUNDLES.glob("*/SdpaBackward/**/support.json")):
        directory = support.parent.relative_to(BUNDLES).as_posix()
        claims = json.loads(support.read_text(encoding="utf-8")).get("claims", {})
        for engine, entries in claims.items():
            for entry in entries:
                for arch in entry.get("support", {}):
                    for case in entry["cases"]:
                        out.append((directory, engine, arch, case))
    return out


@pytest.mark.parametrize("arch", ARCHES)
def test_every_bundle_case_lowers_or_is_malformed(corpus_tool, verdicts, arch):
    known = set(corpus_tool.REASON_CODES) | {corpus_tool.H.LOWERING, "OK"}
    codes = Counter()
    for cid, expect in verdicts[arch].items():
        code = expect["code"]
        codes[code] += 1
        assert code in known, (cid, code)
        if code == corpus_tool.H.LOWERING:
            assert any(
                reason.search(expect["detail"]) for reason in MALFORMED
            ), f"{cid}: the lowering refused a bundle graph: {expect['detail']}"
    assert codes["OK"] > 0, f"{arch}: the engine accepts no bundle case: {dict(codes)}"


@pytest.mark.parametrize("arch", ARCHES)
def test_accepted_cases_are_a_function_of_the_lowered_request(
    corpus_tool, bundle_cases, verdicts, arch
):
    H = corpus_tool.H
    accepted = 0
    for cid, graph in bundle_cases:
        expect = verdicts[arch][cid]
        if expect["code"] != "OK":
            continue
        request = H.from_hipdnn_graph(graph, arch, num_cus=FULL_CUS[arch])
        rewritten = corpus_tool.normalize(H.to_hipdnn_graph(request))
        again = _verdict(corpus_tool, rewritten, arch)
        assert again == expect, f"{cid}: the rewritten graph plans differently"
        assert _verdict(corpus_tool, graph, arch) == expect, f"{cid}: not deterministic"
        accepted += 1
    assert accepted > 0


def test_claims_for_the_engine_name_cases_it_accepts(verdicts):
    by_key = {
        arch: {_case_key(cid): expect for cid, expect in cases.items()}
        for arch, cases in verdicts.items()
    }
    for directory, engine, arch, case in _claims():
        if engine != ENGINE:
            continue
        assert arch in ARCHES, f"{directory}: {ENGINE} claimed on {arch}"
        expect = by_key[arch].get((directory, case))
        assert expect is not None, f"{directory}/{case}: claimed, but no such case"
        assert (
            expect["code"] == "OK"
        ), f"{directory}/{case} on {arch}: claimed for {ENGINE}, which declines it ({expect['code']})"


def test_asm_backward_claims_are_present():
    asm = [claim for claim in _claims() if claim[1] == ASM_ENGINE]
    assert asm, "no ASM backward claims found; the bundle layout moved"
