# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The real writer and native host load exactly the same bias selection metadata.

Pass: serialized geko outputs preserve bias-off, bias-presence and bias-type facts.
Break: retain bias for a no-bias call, discard the marker/whitelist, or serialize
selection predicates differently. GPUBiasPresentContract.geko_output_contract
loads this same fixture and calls the real collectNarrowedFactPredicate().
The writer check here is CPU-only; the native loader needs a visible AMD GPU.
"""
import gzip
from pathlib import Path

import msgpack
import yaml

from geko.specialize.recipe import deriver
from tests.specialize_test_support import identity, recipe_key, row, specialize

CONTRACT = (Path(__file__).resolve().parents[3] / "tensilelite/tests/configs/SolutionLibraries/geko_bias_contract.dat")


def serialized_bias_contract(tmp_path, monkeypatch):
    derive = deriver("gfx950")
    merged = None
    for name, present, excluded in (("off", False, False), ("typed", True, False), ("general", True, True)):
        root = tmp_path / name
        root.mkdir(parents=True)
        record = row(bias_vector=present, bias_type="bf16_r", call_count=3,
                     M=32, N=618, K=120, lda=32, ldb=120, ldc=32, ldd=32)
        extra = []
        if excluded:
            exclusion = root / "exclude.yaml"
            exclusion.write_text(yaml.safe_dump({"exclusions": [{
                "recipe": recipe_key(), "identity": identity(record), "facts": ["UseBias"],
            }]}))
            extra = ["--exclude-controls", str(exclusion)]
        result = specialize(root, [record], monkeypatch, extra=extra)
        assert result["code"] == 0 and result["refusals"] == []
        assert len(result["manifest"]["calls"]) == 1
        assert result["manifest"]["calls"][0]["identity"] == identity(record)
        logic, = result["logic"].values()
        parsed = derive.LibraryIO.parseLibraryLogicData(
            logic, "geko_bias_contract.yaml", derive.assembler, False, False, False,
            derive.isaInfoMap, False)
        library = parsed.library
        if merged is None:
            merged = library
        else:
            merged.merge(library)
    merged.applyNaming(False)
    state = merged.state()
    assert len(state["solutions"]) == 3
    return state


def test_serialize_bias_contract_consumed_by_real_host(tmp_path, monkeypatch):
    actual = serialized_bias_contract(tmp_path, monkeypatch)
    expected = msgpack.unpackb(gzip.decompress(CONTRACT.with_suffix(".dat.gz").read_bytes()),
                               raw=False, strict_map_key=False)
    assert actual == expected, "writer output differs from the metadata exercised by the native host"
    assert yaml.safe_load(gzip.decompress(CONTRACT.with_suffix(".yaml.gz").read_bytes())) == expected, \
        "YAML and msgpack host consumers must receive the same writer metadata"
