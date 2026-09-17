# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/models.py -- the derived @property surface."""

from codegen.models import EngineSpec, KmdField
from tests.helpers import make_engine, make_minimal_config, make_pack


class TestEngineSpecDerivation:
    def test_namespace_and_local_name(self):
        engine = make_engine(name="hipkernel:ConvFwd")
        assert engine.namespace == "hipkernel"
        assert engine.local_name == "ConvFwd"

    def test_slug_is_snake_case(self):
        engine = make_engine(name="hipkernel:ConvFwd")
        assert engine.slug == "conv_fwd"

    def test_pascal_name_round_trips(self):
        engine = make_engine(name="hipkernel:ConvFwd")
        assert engine.pascal_name == "ConvFwd"

    def test_camel_name(self):
        engine = make_engine(name="hipkernel:ConvFwd")
        assert engine.camel_name == "convFwd"

    def test_has_heuristic_true_by_default(self):
        engine = make_engine()
        assert engine.has_heuristic

    def test_has_heuristic_false_when_none(self):
        engine = make_engine(heuristic="none")
        assert not engine.has_heuristic


class TestKmdFieldDerivation:
    def test_mandatory_when_no_default(self):
        field = KmdField(name="dtype", type="string")
        assert field.is_mandatory

    def test_not_mandatory_with_default(self):
        field = KmdField(name="block_size", type="int", default_value=64)
        assert not field.is_mandatory

    def test_int_typed(self):
        assert KmdField(name="x", type="int").is_int_typed

    def test_non_int_not_int_typed(self):
        assert not KmdField(name="x", type="string").is_int_typed
        assert not KmdField(name="x", type="float").is_int_typed
        assert not KmdField(name="x", type="bool").is_int_typed
        assert not KmdField(name="x", type="int_list").is_int_typed


class TestIngestorConfigDerivation:
    def test_native_symbol_namespace(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.native_symbol_namespace == "hipkernel.conv_fwd"

    def test_graph_match_symbol(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.graph_match_symbol == "hipkernel.conv_fwd.graph_match"

    def test_score_symbol(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.score_symbol == "hipkernel.conv_fwd.score"

    def test_dispatch_symbol(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.dispatch_symbol == "hipkernel.conv_fwd.dispatch"

    def test_kernel_match_symbol(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.kernel_match_symbol == "hipkernel.conv_fwd.kernel_match"

    def test_operation_match_symbol(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:Pointwise"))
        pack = make_pack(name="add", discriminator="add")
        assert config.operation_match_symbol(pack) == "hipkernel.pointwise.add_match"

    def test_register_symbols_fn(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.register_symbols_fn == "registerConvFwdSymbols"

    def test_dispatch_handler_class(self):
        config = make_minimal_config(engine=make_engine(name="hipkernel:ConvFwd"))
        assert config.dispatch_handler_class == "ConvFwdDispatchHandler"

    def test_kdp_stem_single_pack_is_bare_slug(self):
        config = make_minimal_config(
            engine=make_engine(name="hipkernel:ConvFwd"), packs=[make_pack(name="main")]
        )
        assert config.kdp_stem(config.packs[0]) == "conv_fwd"

    def test_kdp_stem_multi_pack_appends_pack_name(self):
        pack_a = make_pack(name="add", discriminator="add")
        pack_b = make_pack(name="mul", discriminator="mul")
        config = make_minimal_config(
            engine=make_engine(name="hipkernel:Pointwise"), packs=[pack_a, pack_b]
        )
        assert config.kdp_stem(pack_a) == "pointwise_add"
        assert config.kdp_stem(pack_b) == "pointwise_mul"

    def test_int_typed_kmd_fields_filters(self):
        config = make_minimal_config()
        names = [f.name for f in config.int_typed_kmd_fields]
        assert "block_size" in names
        assert "dtype" not in names

    def test_is_multi_pack(self):
        single = make_minimal_config(packs=[make_pack(name="a")])
        multi = make_minimal_config(
            packs=[
                make_pack(name="a", discriminator="a"),
                make_pack(name="b", discriminator="b"),
            ]
        )
        assert not single.is_multi_pack
        assert multi.is_multi_pack
