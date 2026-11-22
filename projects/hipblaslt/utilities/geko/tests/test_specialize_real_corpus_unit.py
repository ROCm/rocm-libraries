# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real model-trace corpus and the keep-beta preservation gate.

Default: portable real excerpts with pinned path/hash/line-range provenance.
Opt-in: GEKO_REAL_CORPUS_ROOT=<external corpus tree> pytest -m slow <this file>.
These are CPU-only even though the external full-corpus tests are marked slow.
Full-log identity/count expectations are frozen independent counter digests;
no parser or other executable is loaded from the supplied corpus tree.
The keep-beta golden also pins absolute baseline and stock-logic paths.
Pass: preserve every logged scalar, observation and weight and emitted logic.
Break: omit an identity field, lose observation weights, or change an exact-size
assertion / bespoke kernel / keep-beta choice. Scratch mutations prove the gate.
"""
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import pytest

from geko.specialize.calls import UNKNOWN, load_calls
from tests.specialize_test_support import invoke, outputs

DATA = Path(__file__).parent / "test_data/specialize_calls/real_corpus"
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
GOLDEN = json.loads((DATA / "stage_A_golden.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_counts(table):
    return Counter({tuple(sorted((k, v) for k, v in identity.items() if v is not UNKNOWN)): entry.call_count
                    for identity, entry in table.items()})


def identity_counts_digest(table):
    canonical = json.dumps(sorted(table.items()), separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("excerpt", PROVENANCE["excerpts"], ids=lambda e: e["configuration"])
def test_real_excerpts_preserve_complete_call_identities_observations_and_counts(excerpt):
    trace = DATA / excerpt["fixture"]
    assert sha(trace) == excerpt["excerpt_sha256"]
    expected = {tuple(sorted(record["identity"].items())): record["call_count"] for record in excerpt["expected"]}
    table = load_calls(trace)
    assert table_counts(table) == expected, "real excerpt call identity or weight changed"
    for record in excerpt["expected"]:
        key = tuple(sorted(record["identity"].items()))
        entry = next(entry for identity, entry in table.items()
                     if tuple(sorted((k, v) for k, v in identity.items() if v is not UNKNOWN)) == key)
        assert entry.observations == {tuple(record["observation"]): record["call_count"]}
        assert entry.sources == {(str(trace), excerpt["excerpt_sha256"]): record["call_count"]}


@pytest.fixture
def external_corpus():
    root = os.environ.get("GEKO_REAL_CORPUS_ROOT")
    if not root:
        pytest.skip("opt-in full corpus: set GEKO_REAL_CORPUS_ROOT to the external real-corpus tree")
    return Path(root)


# Pass: every full mask-128 trace matches independently frozen per-identity weights,
# not just the total number of calls.
# Break: narrow the file inventory, omit an epilogue, or merge equal-shaped calls.
@pytest.mark.slow
@pytest.mark.parametrize("source", PROVENANCE["full_logs"], ids=lambda s: s["path"].split("/")[1])
def test_every_full_real_mask128_log_matches_consumer_identity_counts(source, external_corpus):
    root = external_corpus
    path = root / source["path"]
    assert sha(path) == source["sha256"], "full trace changed since the fixture provenance was pinned"
    table = table_counts(load_calls(path))
    assert len(table) == source["identity_rows"]
    assert sum(table.values()) == source["call_count"]
    assert identity_counts_digest(table) == source["identity_counts_sha256"]
    for excerpt in PROVENANCE["excerpts"]:
        if root / excerpt["source_path"] == path:
            assert source["sha256"] == excerpt["source_sha256"]
            lines = path.read_bytes().splitlines(keepends=True)
            selected = b"".join(b"".join(lines[start - 1:end]) for start, end in excerpt["line_ranges"])
            assert selected == (DATA / excerpt["fixture"]).read_bytes()


# Pass: merging full process traces adds every per-identity weight.
# Break: last-file wins, path de-duplication drops distinct supplied logs, or keys
# include recipe observation metadata instead of the consumer's call identity.
@pytest.mark.slow
def test_full_real_logs_merge_with_consumer_counts(external_corpus):
    root = external_corpus
    files = [root / source["path"] for source in PROVENANCE["full_logs"]]
    table = table_counts(load_calls(files))
    assert len(table) == PROVENANCE["full_identity_rows"]
    assert sum(table.values()) == sum(source["call_count"] for source in PROVENANCE["full_logs"])
    assert identity_counts_digest(table) == PROVENANCE["full_identity_counts_sha256"]


# Pass: the same baseline and all six original model traces preserve every
# deterministic manifest value and each logic file. The historical manifest is
# normalized for public field names, the keep-beta reason and empty refusals.
# Break: alter beta policy, call counts, identity fields, exact sizes, sharing,
# recipes, kernel names, metadata, or any other manifest value.
@pytest.mark.slow
def test_stage_A_manifest_and_logic_match_frozen_golden(external_corpus, tmp_path, monkeypatch):
    root = external_corpus
    for relative, digest in {**GOLDEN["traces"], **GOLDEN["route_facts"]}.items():
        assert sha(root / relative) == digest, f"golden input changed: {relative}"
    source = root / GOLDEN["source_path"]
    assert sha(source) == GOLDEN["source_sha256"]
    out = tmp_path / "stage_A"
    code = invoke(["--specialize", "--workload-log", str(root / "corpus/traces"),
                   "--route-facts", str(root / "corpus/route_facts_producer"),
                   "--hipblaslt", str((root / GOLDEN["baseline_root"]).resolve()), "--arch", "gfx950",
                   "--workdir", str(out), "--keep-beta"], monkeypatch)
    result = outputs(out, code)
    assert code == 0
    assert result["refusals"] == []
    manifest = result["manifest"]
    assert manifest["stage"] == "keep-beta"
    assert len(manifest["calls"]) == GOLDEN["call_rows"]
    assert sum(call["call_count"] for call in manifest["calls"]) == GOLDEN["call_count"]
    assert all(call["keep_beta"] is True for call in manifest["calls"])
    # Runtime duration is the sole omitted field, checked separately.
    timings = manifest["timings_seconds"]
    assert set(timings) == {"parse_specialize", "logic_merge", "tensile_create_library", "total"}
    assert all(math.isfinite(value) and value >= 0 for value in timings.values())
    deterministic = {key: value for key, value in manifest.items() if key != "timings_seconds"}
    actual = json.dumps(deterministic, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(actual).hexdigest() == GOLDEN["normalized_manifest_sha256"], "keep-beta golden manifest changed"
    assert {path.name: sha(path) for path in (out / "logic/gfx950").glob("*.yaml")} == GOLDEN["logic_sha256"]
