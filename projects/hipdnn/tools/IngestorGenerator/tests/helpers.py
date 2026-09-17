# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Factory helpers for building minimal valid model instances in unit tests."""

from codegen.models import (
    EngineSpec,
    GraphMatchSpec,
    IngestorConfig,
    KernelSource,
    KernelSpec,
    KmdField,
    PackSpec,
)


def make_engine(**overrides) -> EngineSpec:
    defaults = dict(name="hipkernel:Test", knobs=["block_size"])
    defaults.update(overrides)
    return EngineSpec(**defaults)


def make_kmd_field(**overrides) -> KmdField:
    defaults = dict(name="block_size", type="int", default_value=64)
    defaults.update(overrides)
    return KmdField(**defaults)


def make_kernel(**overrides) -> KernelSpec:
    defaults = dict(
        name="test.f32_block64",
        kernel_source=KernelSource(
            kind="embedded_source", source_file="Test.cpp", entry_point="Test"
        ),
        metadata={"block_size": 64, "dtype": "FLOAT"},
    )
    defaults.update(overrides)
    return KernelSpec(**defaults)


def make_pack(**overrides) -> PackSpec:
    defaults = dict(name="test", kernels=[make_kernel()])
    defaults.update(overrides)
    return PackSpec(**defaults)


def make_minimal_config(**overrides) -> IngestorConfig:
    """A minimal valid single-pack IngestorConfig, for unit tests."""
    defaults = dict(
        engine=make_engine(),
        kmd_fields=[make_kmd_field(), KmdField(name="dtype", type="string")],
        packs=[make_pack()],
        graph_match=GraphMatchSpec(),
    )
    defaults.update(overrides)
    return IngestorConfig(**defaults)
