# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""<struct>_default() derived from a dataclass: a field whose default_factory
builds a value (TraitSpec) is set through that struct's own C _default(); with
none, zeros would not be the factory's value, so the class gets no derived
default (a hole) instead of a wrong one."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from trtest import TR

sys.path.insert(0, str(TR))
from tr.env import CFunc, CStruct  # noqa: E402
from tr.translate import derived_default  # noqa: E402

HDR = "instance_probe.h"


@dataclass
class Trait:
    k: int = 3


@dataclass
class Spec:
    shape: tuple = field(default_factory=tuple)
    trait: Trait = field(default_factory=Trait)
    wave_size: int = 64


def _run(with_trait_default: bool) -> dict:
    funcs = {
        "rocke_probe_spec_default": CFunc(
            "rocke_probe_spec_default", "rocke_probe_spec_t", [], HDR, 1
        )
    }
    if with_trait_default:
        funcs["rocke_probe_trait_default"] = CFunc(
            "rocke_probe_trait_default", "rocke_probe_trait_t", [], "other.h", 1
        )
    fields = {
        "shape": "int*",
        "n_shape": "int",
        "trait": "rocke_probe_trait_t",
        "wave_size": "int",
    }
    env = SimpleNamespace(
        headers=SimpleNamespace(
            funcs=funcs,
            structs={"rocke_probe_spec_t": CStruct("rocke_probe_spec_t", fields)},
        )
    )
    mod = SimpleNamespace(
        class_struct={"Spec": "rocke_probe_spec_t"},
        family=lambda cls: None,
        pymod=SimpleNamespace(Spec=Spec),
    )
    return derived_default(env, mod, HDR)


def test_factory_through_the_field_structs_c_default():
    body = _run(True)["rocke_probe_spec_default"]
    assert "    s.trait = rocke_probe_trait_default();" in body, body
    assert "    s.wave_size = 64;" in body, body
    assert not any("shape" in line for line in body), body


def test_factory_without_a_c_default_is_not_derived():
    assert "rocke_probe_spec_default" not in _run(False)
