# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib

from rocke_client_aot.sidecar import (
    SIDECAR_SCHEMA,
    canonical_hash,
    canonical_json_bytes,
    make_sidecar,
)


def test_canonical_json_bytes_are_stable_and_compact():
    value = {"b": [2, 1], "a": {"z": True, "m": None}}

    assert canonical_json_bytes(value) == b'{"a":{"m":null,"z":true},"b":[2,1]}'


def test_canonical_hash_uses_canonical_json_bytes():
    value = {"b": 2, "a": 1}

    assert canonical_hash(value) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()


def test_make_sidecar_uses_common_envelope_and_operation_entries():
    sidecar = make_sidecar(
        cache_key="sdpa_fwd:fmha_fwd_mfma:example",
        artifact={"hsaco_filename": "example.hsaco"},
        selection={"dtypes": {"q": "fp16"}},
        launch={"block": [32, 1, 1]},
        args_signature=[{"name": "Q"}],
    )

    assert sidecar == {
        "schema": SIDECAR_SCHEMA,
        "cache_key": "sdpa_fwd:fmha_fwd_mfma:example",
        "artifact": {"hsaco_filename": "example.hsaco"},
        "selection": {"dtypes": {"q": "fp16"}},
        "launch": {"block": [32, 1, 1]},
        "args_signature": [{"name": "Q"}],
    }
