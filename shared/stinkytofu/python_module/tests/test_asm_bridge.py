"""Tests for the thin, batch assembly-emission bridge."""

import pytest
from stinkytofu import Register, hwreg, sgpr, vgpr
from stinkytofu import emit_asm as _raw_emit_asm


def emit_asm(target, wave_size, items):
    prepared = []
    for raw in items:
        item = dict(raw)
        if item.get("kind") == "instruction" and "form" not in item and "mnemonic" in item:
            item["form"] = f"gfx1250::{item['mnemonic']}"
        prepared.append(item)
    return _raw_emit_asm(target, wave_size, prepared)


def _emit(items):
    return emit_asm("gfx1250", 32, items)


def representative_items():
    return [
        {"kind": "label", "name": "entry", "alignment": 16},
        {"kind": "instruction", "mnemonic": "s_set_vgpr_msb", "src": [Register(65)]},
        {
            "kind": "instruction",
            "mnemonic": "v_add_nc_u32",
            "dst": [vgpr(44)],
            "src": [vgpr(45), vgpr(46)],
        },
        {
            "kind": "instruction",
            "mnemonic": "v_add_nc_u64",
            "dst": [vgpr(48, 2)],
            "src": [vgpr(50, 2), vgpr(52, 2)],
        },
        {
            "kind": "instruction",
            "mnemonic": "v_wmma_f32_16x16x32_bf16",
            "dst": [vgpr(64, 8)],
            "src": [vgpr(72, 8), vgpr(80, 8), vgpr(64, 8)],
        },
        {
            "kind": "instruction",
            "mnemonic": "ds_load_b128",
            "dst": [vgpr(88, 4)],
            "src": [vgpr(4)],
            "modifiers": {"ds": {"offset": 32}},
        },
        {"kind": "instruction", "mnemonic": "s_cbranch_scc0", "target": "done"},
        {"kind": "comment", "text": "standalone comment"},
        {"kind": "blank"},
        # The three audited packed-custom shapes: FMA with SGPR-pair src1,
        # MUL with SGPR-pair src1, and MUL with VGPR-pair src1/op_sel_hi.
        # dst=v[0:1], src0=v[2:3], src1=s[4:5], src2=v[6:7]
        {"kind": "custom", "text": ".long 0xcc1f4400\n.long 0x9c180902"},
        # dst=v[8:9], src0=v[10:11], src1=s[12:13]
        {"kind": "custom", "text": ".long 0xcc280008\n.long 0x1800190a"},
        # dst=v[14:15], src0=v[16:17], src1=v[18:19], op_sel_hi=[0,1,1]
        {"kind": "custom", "text": ".long 0xcc28000e\n.long 0x18022510"},
        {"kind": "alignment", "bytes": 32},
        {"kind": "label", "name": "done"},
        {"kind": "instruction", "mnemonic": "s_endpgm"},
    ]


def test_emit_asm_mixes_typed_and_custom_records_in_order():
    assembly = _emit( representative_items())

    expected_fragments = [
        ".align 16\nentry:\n",
        "s_set_vgpr_msb 65",
        "v_add_nc_u32 v44, v45, v46",
        "v_add_nc_u64 v[48:49], v[50:51], v[52:53]",
        "v_wmma_f32_16x16x32_bf16 v[64:71], v[72:79], v[80:87], v[64:71]",
        "ds_load_b128 v[88:91], v4 offset:32",
        "s_cbranch_scc0 done",
        "// standalone comment\n\n",
        ".long 0xcc1f4400\n.long 0x9c180902\n",
        ".long 0xcc280008\n.long 0x1800190a\n",
        ".long 0xcc28000e\n.long 0x18022510\n",
        ".align 32\ndone:\n",
        "s_endpgm",
    ]
    positions = [assembly.index(fragment) for fragment in expected_fragments]
    assert positions == sorted(positions)


def test_emit_asm_repeated_calls_do_not_share_state():
    first = emit_asm(
        "gfx1250",
        32,
        [{"kind": "label", "name": "first"}, {"kind": "instruction", "mnemonic": "s_endpgm"}],
    )
    second = emit_asm(
        "gfx1250",
        32,
        [{"kind": "label", "name": "second"}, {"kind": "instruction", "mnemonic": "s_endpgm"}],
    )
    assert "first:" in first and "second:" not in first
    assert "second:" in second and "first:" not in second


@pytest.mark.parametrize(
    "bad_items, message",
    [
        (
            [{"kind": "instruction", "mnemonic": "not_an_opcode"}],
            "item\\[0\\].*unknown instruction",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [vgpr(0)],
                    "src": [vgpr(1)],
                }
            ],
            "expects 1 dst and 2 src",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [vgpr(0)],
                    "src": [vgpr(1), 2],
                }
            ],
            "field 'src'\\[1\\] must be a Register",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [Register(3)],
                    "src": [vgpr(1), vgpr(2)],
                }
            ],
            "destination must be a physical register",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [vgpr(0)],
                    "src": [Register("0\\ns_branch hidden"), vgpr(2)],
                }
            ],
            "string literals are only allowed",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u64",
                    "dst": [vgpr(0)],
                    "src": [vgpr(2, 2), vgpr(4, 2)],
                }
            ],
            "register width 1, expected 2",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [vgpr(0)],
                    "src": [vgpr(1), vgpr(2)],
                    "modifiers": {"ds": {"offset": 4}},
                }
            ],
            "modifier 'ds' is only valid",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "ds_load_b128",
                    "dst": [vgpr(0, 4)],
                    "src": [vgpr(4)],
                    "modifiers": {"ds": {"offset0": 1, "offset1": 2}},
                }
            ],
            "offset0/offset1 require na=2",
        ),
        (
            [
                {"kind": "label", "name": "duplicate"},
                {"kind": "label", "name": "duplicate"},
            ],
            "duplicate label",
        ),
        (
            [
                {"kind": "instruction", "mnemonic": "s_branch", "target": "missing"},
            ],
            "undefined branch target",
        ),
        (
            [{"kind": "label", "name": "bad_alignment", "alignment": 3}],
            "power of two",
        ),
        ([{"kind": "alignment", "bytes": 3}], "power of two"),
        ([{"kind": "comment", "text": "safe\ns_endpgm"}], "single line"),
        ([{"kind": "comment", "text": "safe\0hidden"}], "NUL bytes"),
        ([{"kind": "blank", "text": "s_endpgm"}], "unknown field 'text'"),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_nc_u32",
                    "dst": [vgpr(0)],
                    "src": [Register(), vgpr(2)],
                }
            ],
            "invalid/null Register",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "ds_load_b128",
                    "dst": [vgpr(0, 4)],
                    "src": [vgpr(4)],
                    "modifiers": {"ds": {"gds": True}},
                }
            ],
            "gds=True.*unsupported",
        ),
        (
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_add_f32",
                    "dst": [vgpr(0)],
                    "src": [vgpr(1), vgpr(2)],
                    "modifiers": {"vop3p": {"op_sel_hi": [0, 1, 1]}},
                }
            ],
            "vop3p.*only valid on VOP3P",
        ),
        ([{"kind": "custom", "text": "s_branch hidden"}], "custom line 1"),
    ],
)
def test_emit_asm_rejects_invalid_records(bad_items, message):
    with pytest.raises(ValueError, match=message):
        _emit( bad_items)


def test_emit_asm_recovers_after_error():
    with pytest.raises(ValueError):
        _emit( [{"kind": "instruction", "mnemonic": "bad"}])

    assert "s_endpgm" in emit_asm(
        "gfx1250", 32, [{"kind": "instruction", "mnemonic": "s_endpgm"}]
    )


@pytest.mark.parametrize("mnemonic", ["s_delay_alu", "buffer_load_b128"])
def test_emit_asm_rejects_unverified_typed_instruction_without_abort(mnemonic):
    with pytest.raises(ValueError, match="unsupported by the T01 typed bridge allowlist"):
        _emit( [{"kind": "instruction", "mnemonic": mnemonic}])

    assert "s_endpgm" in emit_asm(
        "gfx1250", 32, [{"kind": "instruction", "mnemonic": "s_endpgm"}]
    )


def test_emit_asm_rejects_target_and_wave_mismatch():
    with pytest.raises(ValueError, match="unsupported architecture"):
        emit_asm("gfx9999", 32, [])
    with pytest.raises(ValueError, match="does not match"):
        emit_asm("gfx1250", 64, [])


def test_emit_asm_supports_audited_negated_vfma_src0():
    source = vgpr(1)
    source.set_minus(True)
    assembly = emit_asm(
        "gfx1250",
        32,
        [
            {
                "kind": "instruction",
                "mnemonic": "v_fma_f32",
                "dst": [vgpr(0)],
                "src": [source, vgpr(2), vgpr(3)],
            }
        ],
    )
    assert "v_fma_f32 v0, -v1, v2, v3" in assembly


def test_emit_asm_supports_t00_descriptor_aliases_and_scalar_vcc_forms():
    assembly = emit_asm(
        "gfx1250",
        32,
        [
            {"kind": "instruction", "mnemonic": "s_add_co_ci_u32", "dst": [sgpr(0)],
             "src": [sgpr(1), Register(-2147483648)]},
            {"kind": "instruction", "mnemonic": "s_add_co_i32", "dst": [sgpr(2)],
             "src": [sgpr(3), Register(69632)]},
            {"kind": "instruction", "mnemonic": "s_add_co_u32", "dst": [sgpr(4)],
             "src": [Register(69632), sgpr(5)]},
            {"kind": "instruction", "mnemonic": "s_sub_co_i32", "dst": [sgpr(6)],
             "src": [Register(0), sgpr(7)]},
            {"kind": "instruction", "mnemonic": "s_code_end"},
            {"kind": "instruction", "mnemonic": "s_version", "src": [Register(0x4009)]},
            {"kind": "instruction", "mnemonic": "s_wait_idle"},
            {"kind": "instruction", "mnemonic": "v_max_num_f32", "dst": [vgpr(0)],
             "src": [vgpr(1), vgpr(2)]},
            {"kind": "instruction", "mnemonic": "v_sub_nc_i32", "dst": [vgpr(3)],
             "src": [sgpr(8), vgpr(4)]},
            {"kind": "instruction", "mnemonic": "v_subrev_nc_u32", "dst": [vgpr(5)],
             "src": [sgpr(9), vgpr(6)]},
            {"kind": "instruction", "mnemonic": "v_max3_num_f32", "dst": [vgpr(7)],
             "src": [vgpr(8), vgpr(9), vgpr(10)]},
            {"kind": "instruction", "mnemonic": "v_permlanex16_b32", "dst": [vgpr(11)],
             "src": [vgpr(12), sgpr(10), sgpr(11)]},
            {"kind": "instruction", "form": "gfx1250::v_cmp_eq_f32::vop3",
             "mnemonic": "v_cmp_eq_f32", "dst": [sgpr(12)],
             "src": [vgpr(13), Register(0)]},
            {"kind": "instruction", "mnemonic": "v_cmp_le_u32",
             "dst": [Register("vcc_lo", 0, 1)], "src": [sgpr(13), vgpr(14)]},
            {"kind": "instruction", "form": "gfx1250::v_cndmask_b32::vop3",
             "mnemonic": "v_cndmask_b32", "dst": [vgpr(15)],
             "src": [vgpr(16), vgpr(17), sgpr(14)]},
        ],
    )
    for expected in (
        "s_add_co_ci_u32 s0, s1, -2147483648",
        "s_add_co_i32 s2, s3, 69632",
        "s_add_co_u32 s4, 69632, s5",
        "s_sub_co_i32 s6, 0, s7",
        "s_code_end",
        "s_version 16393",
        "s_wait_idle",
        "v_max_num_f32 v0, v1, v2",
        "v_sub_nc_i32 v3, s8, v4",
        "v_subrev_nc_u32 v5, s9, v6",
        "v_max3_num_f32 v7, v8, v9, v10",
        "v_permlanex16_b32 v11, v12, s10, s11",
        "v_cmp_eq_f32_e64 s12, v13, 0",
        "v_cmp_le_u32 vcc_lo, s13, v14",
        "v_cndmask_b32_e64 v15, v16, v17, s14",
    ):
        assert expected in assembly


def test_emit_asm_uses_selected_primary_or_promoted_descriptor_fields():
    primary = _raw_emit_asm(
        "gfx1250",
        32,
        [{"kind": "instruction", "form": "gfx1250::v_add_nc_u32",
          "mnemonic": "v_add_nc_u32", "dst": [vgpr(0)],
          "src": [vgpr(1), vgpr(2)]}],
    )
    promoted = _raw_emit_asm(
        "gfx1250",
        32,
        [{"kind": "instruction", "form": "gfx1250::v_add_nc_u32::vop3",
          "mnemonic": "v_add_nc_u32", "dst": [vgpr(0)],
          "src": [vgpr(1), sgpr(2)]}],
    )
    assert "v_add_nc_u32 v0, v1, v2" in primary
    assert "v_add_nc_u32_e64 v0, v1, s2" in promoted


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"kind": "instruction", "mnemonic": "s_endpgm"}, "missing required field 'form'"),
        ({"kind": "instruction", "form": "gfx1250::s_endpgm::vop3",
          "mnemonic": "s_endpgm"}, "does not provide"),
        ({"kind": "instruction", "form": "gfx1250::v_add_nc_u32",
          "mnemonic": "v_add_nc_u32", "dst": [vgpr(0)],
          "src": [vgpr(1), sgpr(2)]}, "field requires a VGPR"),
        ({"kind": "instruction", "form": "gfx1250::v_add_f32::wrong",
          "mnemonic": "v_add_f32", "dst": [vgpr(0)],
          "src": [vgpr(1), vgpr(2)]}, "does not select instruction"),
    ],
)
def test_emit_asm_rejects_invalid_selected_forms(item, message):
    with pytest.raises(ValueError, match=message):
        _raw_emit_asm("gfx1250", 32, [item])


@pytest.mark.parametrize(
    ("mnemonic", "dst", "src", "negated_index"),
    [
        ("v_add_nc_u32", [vgpr(0)], [vgpr(1), vgpr(2)], 0),
        ("v_fma_f32", [vgpr(0)], [vgpr(1), vgpr(2), vgpr(3)], 1),
    ],
)
def test_emit_asm_rejects_unaudited_register_negate(mnemonic, dst, src, negated_index):
    src[negated_index].set_minus(True)
    with pytest.raises(ValueError, match="negate/absolute flags are unsupported"):
        emit_asm(
            "gfx1250",
            32,
            [
                {
                    "kind": "instruction",
                    "mnemonic": mnemonic,
                    "dst": dst,
                    "src": src,
                }
            ],
        )


def test_emit_asm_rejects_absolute_vfma_src0():
    source = vgpr(1)
    source.set_abs(True)
    with pytest.raises(ValueError, match="negate/absolute flags are unsupported"):
        emit_asm(
            "gfx1250",
            32,
            [
                {
                    "kind": "instruction",
                    "mnemonic": "v_fma_f32",
                    "dst": [vgpr(0)],
                    "src": [source, vgpr(2), vgpr(3)],
                }
            ],
        )


def test_emit_asm_supports_t04_scalar_special_hwreg_and_float_operands():
    assembly = emit_asm(
        "gfx1250",
        32,
        [
            {"kind": "instruction", "mnemonic": "s_mov_b32", "dst": [Register("m", 0, 1)],
             "src": [Register(0)]},
            {"kind": "instruction", "mnemonic": "s_mov_b32", "dst": [sgpr(1)],
             "src": [Register("ttmp3")]},
            {"kind": "instruction", "mnemonic": "s_setreg_IMM32_b32",
             "dst": [hwreg(1, 2, 3)], "src": [Register(7)]},
            {"kind": "instruction", "mnemonic": "s_prefetch_inst",
             "src": [sgpr(2, 2), Register(0), Register("m", 0, 1), Register(31)]},
            {"kind": "instruction", "mnemonic": "v_add_f32", "dst": [vgpr(0)],
             "src": [Register(1.0), vgpr(1)]},
        ],
    )
    assert "s_mov_b32 m0, 0" in assembly
    assert "s_mov_b32 s1, ttmp3" in assembly
    assert "s_setreg_IMM32_b32 hwreg(1,2,3), 7" in assembly
    assert "s_prefetch_inst s[2:3], 0, m0, 31" in assembly
    assert "v_add_f32 v0, 1.0, v1" in assembly


def test_emit_asm_supports_t04_wait_and_modifier_families():
    assembly = emit_asm(
        "gfx1250",
        32,
        [
            {"kind": "instruction", "mnemonic": "s_wait_alu", "src": [Register(0)],
             "modifiers": {"waitalu": {"va_vdst": 0}}},
            {"kind": "instruction", "mnemonic": "s_waitcnt", "src": [Register(0)],
             "modifiers": {"swaitcnt": {"kmcnt": 0}}},
            {"kind": "instruction", "mnemonic": "s_wait_tensorcnt", "src": [Register(2)],
             "modifiers": {"swaittensorcnt": {"value": 2}}},
            {"kind": "instruction", "mnemonic": "ds_load_b128", "dst": [vgpr(4, 4)],
             "src": [vgpr(8)], "modifiers": {"ds": {"offset": 32}}},
            {"kind": "instruction", "mnemonic": "v_pk_add_f32", "dst": [vgpr(12, 2)],
             "src": [vgpr(14, 2), vgpr(16, 2)],
             "modifiers": {"vop3p": {"op_sel_hi": [0, 1, 1]}}},
            {"kind": "instruction", "mnemonic": "v_pk_mul_f32", "dst": [vgpr(18, 2)],
             "src": [vgpr(20, 2), vgpr(22, 2)]},
        ],
    )
    assert "s_wait_alu depctr_va_vdst(0)" in assembly
    assert "s_waitcnt lgkmcnt(0)" in assembly
    assert "s_wait_tensorcnt 2" in assembly
    assert "ds_load_b128 v[4:7], v8 offset:32" in assembly
    assert "v_pk_add_f32" in assembly
    assert "v_pk_mul_f32 v[18:19], v[20:21], v[22:23]" in assembly


def test_emit_asm_allows_only_audited_ds_load_b128_vgpr_boundary_crossing():
    assembly = _emit(
        [
            {"kind": "instruction", "mnemonic": "ds_load_b128",
             "dst": [vgpr(254, 4)], "src": [vgpr(8)],
             "modifiers": {"ds": {"offset": 32}}},
        ]
    )
    assert "ds_load_b128 v[254:257], v8 offset:32" in assembly

    with pytest.raises(ValueError, match="VGPR range exceeds"):
        _emit(
            [
                {"kind": "instruction", "mnemonic": "ds_load_b128",
                 "dst": [vgpr(253, 4)], "src": [vgpr(8)]},
            ]
        )

    with pytest.raises(ValueError, match="VGPR range exceeds"):
        _emit(
            [
                {"kind": "instruction", "mnemonic": "ds_store_b128",
                 "src": [vgpr(8), vgpr(254, 4)]},
            ]
        )


def test_emit_asm_omits_synthetic_flat_scalar_address_token():
    assembly = _emit(
        [
            {"kind": "instruction", "mnemonic": "flat_load_b32",
             "dst": [vgpr(6)], "src": [vgpr(2, 2), Register("off")]},
            {"kind": "instruction", "mnemonic": "flat_store_b32",
             "src": [vgpr(4, 2), vgpr(6), Register("off")]},
        ]
    )
    assert "flat_load_b32 v6, v[2:3]" in assembly
    assert "flat_store_b32 v[4:5], v6" in assembly
    assert ", off" not in assembly


def test_emit_asm_supports_exact_mfma_smem_and_global_modifier_families():
    assembly = _emit(
        [
            {"kind": "instruction", "mnemonic": "v_wmma_f32_16x16x32_bf16",
             "dst": [vgpr(0, 8)], "src": [vgpr(8, 8), vgpr(16, 8), vgpr(0, 8)],
             "modifiers": {"mfma": {"reuse_a": True}}},
            {"kind": "instruction", "mnemonic": "s_load_b32", "dst": [sgpr(0)],
             "src": [sgpr(2, 2), Register(0)],
             "modifiers": {"smem": {"offset": 4}}},
            {"kind": "instruction", "mnemonic": "global_prefetch_b8",
             "src": [vgpr(24), sgpr(4, 2)],
             "modifiers": {"global": {"offset": 8}}},
        ]
    )
    assert "v_wmma_f32_16x16x32_bf16" in assembly
    assert "matrix_a_reuse" in assembly
    assert "s_load_b32 s0, s[2:3], 0 offset:4" in assembly
    assert "global_prefetch_b8 v24, s[4:5] offset:8" in assembly


@pytest.mark.parametrize(
    ("modifier", "message"),
    [
        ({"mfma": {}}, "modifier 'mfma' is only valid"),
        ({"smem": {}}, "modifier 'smem' is only valid"),
        ({"global": {}}, "modifier 'global' is only valid"),
        ({"swaitcnt": {}}, "modifier 'swaitcnt' is only valid"),
        ({"swaittensorcnt": {}}, "modifier 'swaittensorcnt' is only valid"),
        ({"waitalu": {}}, "modifier 'waitalu' is only valid"),
    ],
)
def test_emit_asm_rejects_modifier_on_unrelated_instruction(modifier, message):
    with pytest.raises(ValueError, match=message):
        _emit([{"kind": "instruction", "mnemonic": "s_endpgm", "modifiers": modifier}])


def test_emit_asm_accepts_selected_immediate_field_boundaries():
    assembly = _emit(
        [
            {"kind": "instruction", "mnemonic": "s_bitset0_b32",
             "dst": [sgpr(0)], "src": [Register(31)]},
            {"kind": "instruction", "mnemonic": "s_prefetch_inst",
             "src": [sgpr(2, 2), Register(-(1 << 23)), Register("m", 0, 1), Register(0)]},
            {"kind": "instruction", "mnemonic": "s_prefetch_inst",
             "src": [sgpr(4, 2), Register((1 << 24) - 1), Register("m", 0, 1), Register(31)]},
        ]
    )
    assert "s_bitset0_b32 s0, 31" in assembly
    assert "s_prefetch_inst s[2:3], -8388608, m0, 0" in assembly
    assert "s_prefetch_inst s[4:5], 16777215, m0, 31" in assembly


@pytest.mark.parametrize(
    "item",
    [
        {"kind": "instruction", "mnemonic": "s_bitset0_b32",
         "dst": [sgpr(0)], "src": [Register(-1)]},
        {"kind": "instruction", "mnemonic": "s_bitset1_b32",
         "dst": [sgpr(0)], "src": [Register(32)]},
    ],
)
def test_emit_asm_rejects_out_of_range_simm5(item):
    with pytest.raises(ValueError, match="5-bit immediate is out of range"):
        _emit([item])


@pytest.mark.parametrize("value", [-(1 << 23) - 1, 1 << 24])
def test_emit_asm_rejects_out_of_range_simm24(value):
    with pytest.raises(ValueError, match="24-bit signed/raw immediate is out of range"):
        _emit(
            [{"kind": "instruction", "mnemonic": "s_prefetch_inst",
              "src": [sgpr(2, 2), Register(value), Register("m", 0, 1), Register(0)]}]
        )


def test_emit_asm_supports_selected_wait_and_vnop_expansions():
    assembly = _raw_emit_asm(
        "gfx1250",
        32,
        [
            {"kind": "instruction", "form": "gfx1250::s_wait_dscnt::from_s_waitcnt",
             "mnemonic": "s_wait_dscnt", "src": [Register(1)]},
            {"kind": "instruction", "form": "gfx1250::s_wait_kmcnt::from_s_waitcnt",
             "mnemonic": "s_wait_kmcnt", "src": [Register(2)]},
            {"kind": "instruction", "form": "gfx1250::s_wait_loadcnt::from_s_waitcnt",
             "mnemonic": "s_wait_loadcnt", "src": [Register(3)]},
            {"kind": "instruction", "form": "gfx1250::s_wait_storecnt::from_s_waitcnt",
             "mnemonic": "s_wait_storecnt", "src": [Register(4)]},
            {"kind": "instruction", "form": "gfx1250::v_nop::from_repeat",
             "mnemonic": "v_nop"},
        ],
    )
    for expected in (
        "s_wait_dscnt 1",
        "s_wait_kmcnt 2",
        "s_wait_loadcnt 3",
        "s_wait_storecnt 4",
        "v_nop",
    ):
        assert expected in assembly


def test_emit_asm_supports_smem_literal_or_sgpr_offset():
    assembly = _emit(
        [
            {"kind": "instruction", "mnemonic": "s_load_b64", "dst": [sgpr(0, 2)],
             "src": [sgpr(4, 2), Register(16)]},
            {"kind": "instruction", "mnemonic": "s_load_b64", "dst": [sgpr(2, 2)],
             "src": [sgpr(4, 2), sgpr(6)]},
        ]
    )
    assert "s_load_b64 s[0:1], s[4:5], 16" in assembly
    assert "s_load_b64 s[2:3], s[4:5], s6" in assembly


def test_emit_asm_supports_two_source_tensor_shapes():
    assembly = emit_asm(
        "gfx1250",
        32,
        [
            {"kind": "instruction", "mnemonic": "tensor_load_to_lds",
             "src": [sgpr(0, 4), sgpr(8, 8)]},
            {"kind": "instruction", "mnemonic": "tensor_store_from_lds",
             "src": [sgpr(16, 4), sgpr(24, 8)]},
        ],
    )
    assert "tensor_load_to_lds s[0:3], s[8:15]" in assembly
    assert "tensor_store_from_lds s[16:19], s[24:31]" in assembly


@pytest.mark.parametrize(
    "items, message",
    [
        ([{"kind": "instruction", "mnemonic": "s_mov_b32", "dst": [sgpr(0)],
           "src": [Register("ttmpx")]}], "string literals are only allowed"),
        ([{"kind": "instruction", "mnemonic": "s_setreg_IMM32_b32",
           "dst": [Register(1)], "src": [Register(7)]}], "structured hwreg"),
        ([{"kind": "instruction", "mnemonic": "s_setreg_IMM32_b32",
           "dst": [hwreg(3)], "src": [Register(7)]}], "not defined"),
        ([{"kind": "instruction", "mnemonic": "s_mov_b32", "dst": [sgpr(0)],
           "src": [Register("m0")]}], "string literals are only allowed"),
        ([{"kind": "instruction", "mnemonic": "s_and_b32", "dst": [sgpr(0)],
           "src": [Register("ttmp16"), Register(1)]}], "audited TTMP"),
        ([{"kind": "instruction", "mnemonic": "v_add_f32", "dst": [vgpr(0)],
           "src": [Register("ttmp3"), vgpr(1)]}], "audited TTMP"),
        ([{"kind": "instruction", "mnemonic": "s_and_b32", "dst": [sgpr(0)],
           "src": [Register("m", 0, 1), Register(1)]}], "m0 is not valid"),
        ([{"kind": "instruction", "mnemonic": "s_wait_alu", "src": [Register(0)],
           "modifiers": {"waitalu": {"hold_cnt": 2}}}], "hardware range"),
    ],
)
def test_emit_asm_rejects_invalid_t04_special_and_modifier_shapes(items, message):
    with pytest.raises(ValueError, match=message):
        _emit( items)
