# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""load_calls: lossless mask-128 call identities, weights and whole-input integrity.

load_calls(file | directory | list[files]) maps immutable CallIdentity keys (logged
scalar text plus a typed view) to call-weighted CallEntry counts, observations and sources.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import yaml

from geko.bench import log as blog
from geko.specialize import calls
from tests.specialize_test_support import CALL_DATA as DATA, write_rows

EXPECTED = json.loads((DATA / "producer_expected.json").read_text())
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
VALID_EXCERPTS = list(EXPECTED["fixtures"]) + ["mask128_without_extended_fields.yaml"]


@pytest.fixture
def row():
    # Literal scalar spellings from the real producer, not YAML-normalized facts.
    return dict(yaml.load((DATA / "producer_default.yaml").read_text(), Loader=yaml.BaseLoader)[0],
                solution_index=7, solution_Name="stock_solution", kernel_name="stock_kernel")


def refused(inputs):
    with pytest.raises(ValueError) as error:
        calls.load_calls(inputs)
    assert str(error.value).strip()
    return error.value


# Break: drop or coerce any logged identity fact (e.g. group by M/N/K or legacy LOG_FIELDS).
IDENTITY_CHANGES = [
    ("function", "gemm"), ("M", "129"), ("N", "129"), ("K", "129"),
    ("lda", "129"), ("ldb", "129"), ("ldc", "129"), ("ldd", "129"),
    ("stride_a", "16384"), ("stride_b", "16384"), ("stride_c", "16384"), ("stride_d", "16384"),
    ("alpha", "0.5"), ("alphai", "0.5"), ("beta", "0.0"), ("betai", "0.5"),
    ("transA", "T"), ("transB", "T"), ("batch_count", "2"), ("batch_mode", "1"),
    ("scaleA", "1"), ("scaleB", "1"), ("scaleC", "1"), ("scaleD", "1"),
    ("swizzleA", "true"), ("swizzleB", "true"), ("scaleAlpha_vector", "true"),
    ("gradient", "true"), ("use_e", "true"), ("bias_vector", "true"),
    ("bias_source", "a"), ("bias_stride", "128"),
    ("a_type", "bf16_r"), ("b_type", "bf16_r"), ("c_type", "bf16_r"), ("d_type", "bf16_r"),
    ("scale_type", "f16_r"), ("bias_type", "bf16_r"), ("aux_type", "bf16_r"),
    ("compute_type", "f32_r"), ("activation_type", "sigmoid"),
    ("flush", "true"), ("any_stride", "false"), ("rotating", "1"), ("cold_iters", "1"), ("iters", "1"),
    ("pointer_mode", "1"), ("activation_arg0", "-1.5"), ("activation_arg1", "2.25"),
]


@pytest.mark.parametrize("field,value", IDENTITY_CHANGES, ids=[p[0] for p in IDENTITY_CHANGES])
def test_different_identity_fact_keeps_separate_calls(row, tmp_path, field, value):
    path = write_rows(tmp_path / "hipblaslt_1.log", [
        dict(row, call_count=2), dict(row, call_count=3, **{field: value}),
    ])
    table = calls.load_calls(path)
    assert len(table) == 2
    assert sorted(entry.call_count for entry in table.values()) == [2, 3]
    assert sum(entry.call_count for entry in table.values()) == 5
    assert {identity[field] for identity in table} == {row[field], value}


# Break: key/hash on YAML-normalized values instead of the logged scalar text.
@pytest.mark.parametrize("field,left,right", [
    ("beta", "0.000000", "-0.000000"),
    ("alpha", "0.000000", "-0.000000"),
    ("alpha", "1.000000", "1.0"),
    ("beta", "0.000000", "0.0"),
    ("swizzleA", "false", "False"),
    ("activation_arg0", "1", "1.0"),
], ids=["beta_signed_zero", "alpha_signed_zero", "alpha_decimals", "beta_decimals", "bool_case", "arg_decimals"])
def test_equivalent_typed_facts_keep_distinct_logged_identities(row, tmp_path, field, left, right):
    path = write_rows(tmp_path / "hipblaslt_1.log", [
        dict(row, call_count=1, **{field: left}), dict(row, call_count=4, **{field: right}),
    ])
    table = calls.load_calls(path)
    assert len(table) == 2
    assert {identity[field]: entry.call_count for identity, entry in table.items()} == {left: 1, right: 4}
    assert sum(entry.call_count for entry in table.values()) == 5
    identities = list(table)
    assert identities[0].typed(field) == identities[1].typed(field)


# Break: drop a column because one row lacks it, or default it to zero/host/false/none.
MISSING_FIELDS = ["pointer_mode", "activation_arg0", "activation_arg1", "beta", "alpha",
                  "function", "scale_type", "compute_type", "bias_vector", "bias_type",
                  "scaleAlpha_vector", "scaleC", "batch_mode", "bias_stride", "M"]


@pytest.mark.parametrize("field", MISSING_FIELDS)
def test_missing_fact_is_unknown_only_in_its_record(row, tmp_path, field):
    missing = dict(row, call_count=3)
    del missing[field]
    table = calls.load_calls(write_rows(tmp_path / "hipblaslt_1.log", [dict(row, call_count=2), missing]))
    assert len(table) == 2
    by_count = {entry.call_count: identity for identity, entry in table.items()}
    assert by_count[2][field] == row[field]
    assert by_count[2].typed(field) == EXPECTED["base_identity"][field]
    assert by_count[3].typed(field) is calls.UNKNOWN
    assert by_count[3][field] is calls.UNKNOWN
    assert by_count[3][field] == calls.UNKNOWN
    assert by_count[3][field] != row[field]
    assert by_count[3][field] is not None
    assert by_count[3][field] != 0
    assert by_count[3][field] != "host"
    assert by_count[3][field] != "none"
    assert sum(entry.call_count for entry in table.values()) == 5


# Break: upgrade a log from before the extended profile by defaulting its new keys.
def test_record_without_extended_profile_keeps_new_facts_unknown():
    table = calls.load_calls(DATA / "mask128_without_extended_fields.yaml")
    assert len(table) == 1
    identity, entry = next(iter(table.items()))
    assert entry.call_count == 1
    assert (identity["M"], identity["N"], identity["K"]) == ("128", "128", "128")
    assert (identity.typed("M"), identity.typed("N"), identity.typed("K")) == (128, 128, 128)
    for field in ("pointer_mode", "activation_arg0", "activation_arg1"):
        assert identity[field] is calls.UNKNOWN
        assert identity[field] != 0
        assert identity[field] is not None


# Break: overwrite a duplicate, count rows instead of calls, or discard provenance.
@pytest.mark.parametrize("input_form", ["file_list", "reversed_file_list", "directory"])
def test_process_logs_sum_duplicate_counts_and_keep_provenance(row, tmp_path, input_form):
    first = write_rows(tmp_path / "hipblaslt_101.log", [dict(row, call_count=2), dict(row, call_count=3)])
    second = write_rows(tmp_path / "hipblaslt_202.log", [dict(row, call_count=5)])
    inputs = {"file_list": [first, second], "reversed_file_list": [second, first], "directory": tmp_path}
    table = calls.load_calls(inputs[input_form])
    assert len(table) == 1
    entry = next(iter(table.values()))
    assert entry.call_count == 10
    assert entry.observations == {(7, "stock_solution", "stock_kernel"): 10}
    assert entry.sources == {
        (str(first), hashlib.sha256(first.read_bytes()).hexdigest()): 5,
        (str(second), hashlib.sha256(second.read_bytes()).hexdigest()): 5,
    }


# Break: glob all *.log or invent a record/count for an empty process log.
def test_directory_empty_process_log_and_nonmatching_file_preserve_totals(row, tmp_path):
    good = write_rows(tmp_path / "hipblaslt_1.log", [dict(row, call_count=2)])
    (tmp_path / "hipblaslt_2.log").write_text("")
    write_rows(tmp_path / "unrelated.log", [dict(row, call_count=7)])
    table = calls.load_calls(tmp_path)
    assert len(table) == 1
    entry = next(iter(table.values()))
    assert entry.call_count == 2
    assert entry.sources == {(str(good), hashlib.sha256(good.read_bytes()).hexdigest()): 2}


# Break: put index/name/kernel into the identity, merge observations, or lose weights.
@pytest.mark.parametrize("field,value,second_observation", [
    ("solution_index", 8, (8, "stock_solution", "stock_kernel")),
    ("solution_Name", "other_solution", (7, "other_solution", "stock_kernel")),
    ("kernel_name", "other_kernel", (7, "stock_solution", "other_kernel")),
], ids=["index", "solution_name", "kernel_name"])
def test_recipe_observations_stay_side_by_side(row, tmp_path, field, value, second_observation):
    table = calls.load_calls(write_rows(tmp_path / "hipblaslt_1.log", [
        dict(row, call_count=2), dict(row, call_count=3, **{field: value}), dict(row, call_count=5),
    ]))
    assert len(table) == 1
    identity, entry = next(iter(table.items()))
    assert entry.call_count == 10
    assert entry.observations == {(7, "stock_solution", "stock_kernel"): 7, second_observation: 3}
    assert {"call_count", "solution_index", "solution_Name", "kernel_name"}.isdisjoint(identity)


# Break: key on mutable dicts or insertion-ordered tuples.
def test_identity_is_immutable_and_independent_of_record_key_order(row, tmp_path):
    del row["pointer_mode"]
    reversed_row = dict(reversed(list(row.items())))
    table = calls.load_calls(write_rows(tmp_path / "hipblaslt_1.log", [
        dict(row, call_count=2), dict(reversed_row, call_count=3),
    ]))
    assert len(table) == 1
    identity = next(iter(table))
    assert table[identity].call_count == 5
    assert identity["pointer_mode"] is calls.UNKNOWN
    previous_hash = hash(identity)
    with pytest.raises(TypeError):
        identity["beta"] = 0
    assert hash(identity) == previous_hash
    assert table[identity].call_count == 5


# Break: a writable backing slot or an unguarded __init__ on a table key.
@pytest.mark.parametrize("operation", ["logged_assignment", "typed_assignment", "reinitialization"])
def test_identity_backing_state_cannot_change_table_lookup(row, tmp_path, operation):
    table = calls.load_calls(write_rows(tmp_path / "hipblaslt_1.log", [dict(row, call_count=2)]))
    identity = next(iter(table))
    before = dict(identity)
    before_typed = {field: identity.typed(field) for field in identity}
    original_hash = hash(identity)
    replacement_logged = tuple("0.000000" if field == "beta" else before[field] for field in identity)
    replacement_typed = tuple(0.0 if field == "beta" else before_typed[field] for field in identity)
    replacement_identity = type(identity)(replacement_logged, replacement_typed)
    assert replacement_identity["beta"] == "0.000000"
    assert replacement_identity.typed("beta") == 0.0
    with pytest.raises((AttributeError, TypeError)):
        if operation == "logged_assignment":
            identity._logged = replacement_logged
        elif operation == "typed_assignment":
            identity._typed = replacement_typed
        else:
            identity.__init__(replacement_logged, replacement_typed)
    assert dict(identity) == before
    assert {field: identity.typed(field) for field in identity} == before_typed
    assert hash(identity) == original_hash
    assert table[identity].call_count == 2


# Break: recreate UNKNOWN during copy/pickle, splitting unknown identities.
@pytest.mark.parametrize("transport", ["copy", "deepcopy", "pickle"])
def test_unknown_transport_preserves_singleton_and_identity_lookup(row, tmp_path, transport):
    roundtrip = {"copy": copy.copy, "deepcopy": copy.deepcopy,
                 "pickle": lambda value: pickle.loads(pickle.dumps(value))}[transport]
    assert roundtrip(calls.UNKNOWN) is calls.UNKNOWN
    missing = dict(row, call_count=2)
    del missing["pointer_mode"]
    table = calls.load_calls(write_rows(tmp_path / "hipblaslt_1.log", [missing]))
    identity = next(iter(table))
    reconstructed = roundtrip(identity)
    assert reconstructed["pointer_mode"] is calls.UNKNOWN
    assert reconstructed == identity
    assert hash(reconstructed) == hash(identity)
    assert table[reconstructed].call_count == 2


# Break: accept a missing count, default call_count to 1, or ingest mask 64.
@pytest.mark.parametrize("defect", ["call_count", "algo_method"])
def test_incomplete_or_mask64_record_is_refused_with_reason(row, tmp_path, defect):
    if defect == "algo_method":
        row[defect] = 2
    else:
        del row[defect]
    refused(write_rows(tmp_path / "hipblaslt_1.log", [row]))


# Break: treat old mask-64 / mask-34 bench commands as aggregate calls.
@pytest.mark.parametrize("name", ["producer_mask64.yaml", "reference_qwen25_mask34.log", "reference_qwen3_mask34.log"])
def test_real_non_mask128_input_is_refused_with_reason(name):
    refused(DATA / name)


# Break: ignore or merge the mask-64 rows/files of a mixed input.
@pytest.mark.parametrize("order", ["128_first", "64_first"])
@pytest.mark.parametrize("input_form", ["one_file", "process_files"])
def test_mixed_mask64_and_mask128_is_refused(row, tmp_path, order, input_form):
    mask64 = blog.read(DATA / "producer_mask64.yaml")[0]
    rows = [row, mask64] if order == "128_first" else [mask64, row]
    if input_form == "one_file":
        inputs = write_rows(tmp_path / "hipblaslt_1.log", rows)
    else:
        inputs = [write_rows(tmp_path / f"hipblaslt_{i}.log", [record]) for i, record in enumerate(rows)]
    refused(inputs)


# Break: skip a bad process file (YAML error, mask 34, undecodable bytes) and return
# the good file's partial table; or leak UnicodeDecodeError instead of the domain error.
BAD_FILES = [(kind, order) for kind in ("truncated", "unparseable", "mask34") for order in ("good_first", "bad_first")]
BAD_FILES += [("non_utf8", order) for order in ("bad_only", "good_first", "bad_first")]


@pytest.mark.parametrize("bad_kind,order", BAD_FILES, ids=[f"{order}-{kind}" for kind, order in BAD_FILES])
def test_bad_process_file_refuses_whole_input(tmp_path, bad_kind, order):
    good = tmp_path / "hipblaslt_1.log"
    good.write_bytes((DATA / "producer_default.yaml").read_bytes())
    bad = tmp_path / "hipblaslt_2.log"
    contents = {
        "truncated": b"- { function: matmul, M: 128, call_count: 7",
        "unparseable": b"- { function: matmul, M: [ }\n",
        "mask34": (DATA / "reference_qwen25_mask34.log").read_bytes(),
        "non_utf8": b"\xff\xfe\n",
    }
    bad.write_bytes(contents[bad_kind])
    inputs = {"good_first": [good, bad], "bad_first": [bad, good], "bad_only": bad}[order]
    assert not isinstance(refused(inputs), UnicodeError)


# Break: drop the unexpected-key check and merge a record after losing a fact.
def test_unknown_identity_key_refuses_input(row, tmp_path):
    refused(write_rows(tmp_path / "hipblaslt_1.log", [dict(row, unrecognized_runtime_fact="new_fact")]))


# Break: coerce arbitrary YAML scalars (0, bool, quoted string) to a call weight.
@pytest.mark.parametrize("token", ["0", "true", '"3"'], ids=["zero", "boolean", "quoted_integer"])
def test_invalid_call_count_is_refused(row, tmp_path, token):
    refused(write_rows(tmp_path / "hipblaslt_1.log", [dict(row, call_count=token)]))


# Break: let a duplicated field silently overwrite an identity fact or the count.
@pytest.mark.parametrize("field,value", [("beta", "0.000000"), ("call_count", "9")], ids=["identity_fact", "call_weight"])
def test_duplicate_trace_field_is_refused(row, tmp_path, field, value):
    path = write_rows(tmp_path / "hipblaslt_1.log", [row])
    path.write_text(path.read_text().rstrip()[:-1] + f", {field}: {value} }}\n")
    refused(path)


# Break: accept an input with no records as zero calls.
@pytest.mark.parametrize("input_form", ["one_file", "file_list", "directory"])
def test_all_empty_trace_input_is_refused(tmp_path, input_form):
    first = tmp_path / "hipblaslt_1.log"
    second = tmp_path / "hipblaslt_2.log"
    first.write_text("")
    second.write_text("[]\n")
    refused({"one_file": first, "file_list": [first, second], "directory": tmp_path}[input_form])


# Break: reopen the file for SHA-256 after parsing. Only open/close are interposed,
# to replace the file deterministically after the single read.
def test_source_hash_identifies_exact_bytes_parsed(tmp_path, monkeypatch):
    path = tmp_path / "hipblaslt_1.log"
    original = (DATA / "producer_default.yaml").read_bytes()
    replacement = original.replace(b"call_count: 1", b"call_count: 9")
    path.write_bytes(original)
    original_open = builtins.open
    original_path_open = Path.open
    opened = []

    class ReplaceAfterRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream.__enter__()

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            if len(opened) == 1:
                with original_path_open(path, "wb") as output:
                    output.write(replacement)
            return result

    def wrap(stream, file, mode):
        if isinstance(file, (str, Path)) and Path(file) == path and mode in {"r", "rb"}:
            opened.append(mode)
            return ReplaceAfterRead(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", lambda file, mode="r", *a, **k: wrap(original_open(file, mode, *a, **k), file, mode))
        patch.setattr(Path, "open", lambda file, mode="r", *a, **k: wrap(original_path_open(file, mode, *a, **k), file, mode))
        table = calls.load_calls(path)
    assert path.read_bytes() == replacement
    assert len(table) == 1
    entry = next(iter(table.values()))
    assert entry.call_count == 1
    assert entry.sources == {(str(path), hashlib.sha256(original).hexdigest()): 1}
    assert len(opened) == 1


# Real producer excerpts. Break: lexical normalization, coercion, rounded arguments
# or changed fact types against the hand-derived expectations.
@pytest.mark.parametrize("name", list(EXPECTED["fixtures"]))
def test_real_producer_excerpts_match_hand_derived_identities(name):
    path = DATA / name
    digest = PROVENANCE[name]["excerpt_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    table = calls.load_calls(path)
    expected_rows = EXPECTED["fixtures"][name]
    assert len(table) == len(expected_rows)
    remaining = list(table.items())
    logged = EXPECTED["logged_changes"][name]
    assert len(logged) == len(expected_rows)
    for expected, changes in zip(expected_rows, logged):
        want = dict(EXPECTED["base_identity"], **expected["changes"])
        want_logged = dict(EXPECTED["base_identity_logged"], **changes)
        matches = [(identity, entry) for identity, entry in remaining if dict(identity) == want_logged]
        assert len(matches) == 1
        identity, entry = matches[0]
        for field, value in want.items():
            assert identity.typed(field) == value
            assert type(identity.typed(field)) is type(value)
        assert entry.call_count == expected["call_count"]
        assert entry.sources == {(str(path), digest): expected["call_count"]}
        remaining.remove((identity, entry))
    assert not remaining


# Break: six-digit formatting/rounding of activation arguments in the key.
@pytest.mark.parametrize("field,name,want", [
    ("activation_arg0", "producer_arg0_ulp.yaml", {"1": 2, "1.00000012": 3}),
    ("activation_arg1", "producer_arg1_ulp.yaml", {"1": 2, "1.00000012": 3}),
])
def test_one_binary32_ulp_activation_difference_does_not_merge(field, name, want):
    table = calls.load_calls(DATA / name)
    assert len(table) == 2
    assert {identity[field]: entry.call_count for identity, entry in table.items()} == want
    assert {identity.typed(field): entry.call_count for identity, entry in table.items()} == {1: 2, 1.00000012: 3}


# Expectations generated by an independent mask-128 consumer parser. Break: lose
# lexical values, keep UNKNOWN in the manifest identity, change aggregation, or
# serialize typed 1.0/False instead of the logged 1.000000/false.
@pytest.mark.parametrize("name", VALID_EXCERPTS)
def test_fixture_grouping_matches_mask128_consumer(name):
    path = DATA / "mask128_reference.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROVENANCE["mask128_reference.json"]["reference_sha256"]
    reference = json.loads(path.read_text())[name]
    want = {tuple(sorted(entry["identity"].items())): entry["call_count"] for entry in reference}
    table = calls.load_calls(DATA / name)
    # The external manifest leaves missing trace fields absent, never defaulted.
    manifest = {identity: {k: v for k, v in identity.items() if v is not calls.UNKNOWN} for identity in table}
    assert {tuple(sorted(manifest[i].items())): entry.call_count for i, entry in table.items()} == want
    assert sum(entry.call_count for entry in table.values()) == sum(want.values())
    written = json.loads(json.dumps({"calls": [{"identity": i} for i in manifest.values()]}))
    assert {tuple(sorted(call["identity"].items())) for call in written["calls"]} == set(want)
    assert all(isinstance(value, str) for call in written["calls"] for value in call["identity"].values())
