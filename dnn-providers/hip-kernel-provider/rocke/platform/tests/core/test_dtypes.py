# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core dtype names are independent of target support and retain public aliases.

The scalar quantization helpers are checked here too: ``quant_max_abs`` is the
largest value of the encoding the target converts to (240 for gfx942's e4m3
FNUZ, 448 for OCP e4m3), ``quantize_scalar_f32`` saturates finite values and
+-inf to it and keeps NaN, and the quantizing instances use the target's bound
in both engines. On a gfx942 or gfx950 GPU the helper's and SmoothQuant's
outputs are checked against the target's encoding.
"""

import ast
import dataclasses
import os
import re
import subprocess
from typing import Literal, get_args, get_type_hints

import numpy as np
import pytest

from rocke.core.dtypes import normalize_dtype
from rocke.core.dtypes import DTypeCategory, dtype_info, dtype_names
from rocke.core.arch.target import MmaScaleDType
from rocke.core import ir
from rocke.core.ir import (
    FP8E4M3,
    I8,
    IRBuilder,
    PtrType,
    Type,
    dtype_to_ir_type,
    type_bits,
)
from rocke.helpers.mma_io import storage_ir_type
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import llvm_f32_literal
from rocke.helpers.compile import compile_kernel
from rocke.helpers.fp8_encoding import fp8_element_format
from rocke.helpers.spec import sig_param
from rocke.tests.comgr import require_comgr_target
from rocke.tests.compiler_tools import require_tools_or_skip
from rocke.tests.engines import (
    ENGINES,
    available_engines,
    instance_llvm_from_all_engines,
    lower_with_engine,
)
from rocke.tests.fp8_gemm_checks import Fp8HostCodec, fp8_host_codec
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.llc import find_llvm_tool
from rocke.helpers.quant import (
    quant_ir_type,
    ir_to_qdtype,
    dequantize_scalar_to_f32,
    quant_max_abs,
    quantize_prescaled_f32,
    quantize_scalar_f32,
    validate_quant_out_dtype,
)


@pytest.mark.parametrize(
    "dtype,bits",
    [
        ("fp4", 4),
        ("fp6", 6),
        ("bf6", 6),
        ("fp8", 8),
        ("bf8", 8),
        ("f16", 16),
        ("bf16", 16),
        ("f32", 32),
        ("f64", 64),
        ("e8m0", 8),
        ("e4m3", 8),
        ("e5m3", 8),
    ],
)
def test_logical_dtype_and_serialization(dtype, bits):
    info = dtype_info(dtype)
    logical = dtype_to_ir_type(dtype)
    assert info.encoded_bits == bits
    assert logical != I8
    b = IRBuilder("dtype_transport")
    b.param("code", logical)
    assert serialize(parse(serialize(b.kernel))) == serialize(b.kernel)


# The width of every IR scalar type, written out by hand so that the format table
# in core/dtypes.py and type_bits are checked against an independent source.
# Adding an IR scalar type means adding its width here; see section 5.4,
# "Adding a dtype", in dsl_docs/development/extending.md.
_IR_SCALAR_BITS = {
    "i1": 1,
    "i4": 4,
    "u4": 4,
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "tf32": 32,
    "f64": 64,
    "fp8e4m3": 8,
    "bf8e5m2": 8,
    "fp6e2m3": 6,
    "fp6e3m2": 6,
    "fp4e2m1": 4,
    "e8m0": 8,
    "e5m3": 8,
}


def test_every_ir_scalar_type_has_a_reference_width():
    defined = {t.name for t in vars(ir).values() if type(t) is Type}
    assert not defined - set(_IR_SCALAR_BITS), (
        f"add the width of {sorted(defined - set(_IR_SCALAR_BITS))} to "
        "_IR_SCALAR_BITS"
    )
    assert not set(_IR_SCALAR_BITS) - defined


@pytest.mark.parametrize(("name", "bits"), sorted(_IR_SCALAR_BITS.items()))
def test_an_ir_scalar_type_has_its_reference_width(name, bits):
    assert dtype_info(name).encoded_bits == bits
    assert type_bits(Type(name)) == bits


def test_native_scalar_widths_match_the_reference():
    # The C++ engine lists every scalar kind it defines (a static_assert keeps
    # the list complete) with the width rocke_i_type_bits gives it.
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    lines = subprocess.run(
        [executable, "--scalar-bits"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    native = {name: int(bits) for name, bits in (line.split() for line in lines)}
    assert native == _IR_SCALAR_BITS


@pytest.mark.parametrize("dtype", ["fp4", "fp6", "bf6"])
def test_quant_type_recognition_does_not_enable_integer_conversion(dtype):
    logical = quant_ir_type(dtype)
    assert logical == dtype_to_ir_type(dtype)
    assert ir_to_qdtype(logical) == dtype_info(dtype).name
    b = IRBuilder("no_lowbit_conversion")
    value = b.param("value", logical)
    scale = b.const_f32(1)
    with pytest.raises(ValueError, match="unsupported input type"):
        dequantize_scalar_to_f32(b, value, scale=scale)


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_info("custom_format")


def test_the_integer_dtypes_are_i8_and_i4_only():
    # "iu8" / "iu4" appear only in RDNA integer WMMA instruction names: signedness is
    # an instruction operand, not a property of the data, so they are not dtypes.
    from rocke.core.ir import I4

    assert dtype_to_ir_type("i8") is I8
    assert dtype_to_ir_type("i4") is I4
    assert dtype_info("i4").encoded_bits == 4
    assert storage_ir_type("i4") is I8
    for name in ("iu8", "iu4"):
        with pytest.raises(ValueError, match="unknown dtype"):
            dtype_info(name)


def test_dtype_names_lists_one_category_in_table_order():
    """Error texts that list the dtypes an operation accepts take the list from
    the table, so a new dtype appears in them without a second edit."""
    assert dtype_names(DTypeCategory.INTEGER) == (
        "i4",
        "i8",
        "u4",
        "u8",
        "i16",
        "i32",
        "i64",
    )
    assert dtype_names(DTypeCategory.PREDICATE) == ("i1",)


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        (" HALF ", "fp16"),
        ("bfloat16", "bf16"),
        (" Float\t", "fp32"),
        (" Double ", "fp64"),
        ("F64", "fp64"),
        ("FP8", "fp8e4m3"),
        (" E4M3 ", "fp8e4m3"),
        ("BF8", "bf8e5m2"),
        ("FP6", "fp6e2m3"),
        ("BF6", "fp6e3m2"),
        ("FP4", "fp4e2m1"),
        ("int32", "i32"),
        ("XF32", "tf32"),
        (" TF32 ", "tf32"),
        (" Custom_Format ", "custom_format"),
        ("", ""),
    ],
)
def test_normalize_dtype(spelling, canonical):
    assert normalize_dtype(spelling) == canonical
    assert normalize_dtype(canonical) == canonical


def test_architecture_entry_points_reexport_core_normalization():
    from rocke.core import normalize_dtype as core_normalize
    from rocke.core.arch import normalize_dtype as arch_normalize
    from rocke.core.arch.target import normalize_dtype as target_normalize

    assert normalize_dtype.__module__ == "rocke.core.dtypes"
    assert core_normalize is arch_normalize is target_normalize is normalize_dtype


@pytest.mark.parametrize("spelling", ["e4m3", "fp8e4m3", "fp8", MmaScaleDType.E4M3])
def test_e4m3_shares_encoding_type_and_storage(spelling):
    assert dtype_info(spelling) is dtype_info("fp8e4m3")
    assert dtype_to_ir_type(spelling) is FP8E4M3
    assert storage_ir_type(spelling) is FP8E4M3
    assert quant_ir_type(spelling) is FP8E4M3
    b = IRBuilder("e4m3_conversion")
    value = b.param("value", dtype_to_ir_type(spelling))
    b.cvt_fp8_to_f32(value)
    assert serialize(b.kernel) == serialize(parse(serialize(b.kernel)))


def test_scale_role_remains_independent_of_encoding_aliases():
    assert MmaScaleDType("fp8e4m3") is MmaScaleDType.E4M3
    assert str(MmaScaleDType.E4M3) == "e4m3"
    assert dtype_info("e5m3") is not dtype_info("bf8e5m2")
    assert dtype_to_ir_type("e5m3") != dtype_to_ir_type("bf8e5m2")
    with pytest.raises(ValueError):
        MmaScaleDType("bf8e5m2")


def test_quant_type_diagnostic_lists_every_accepted_spelling():
    expected = {
        "i8",
        "int8",
        "fp8",
        "fp8e4m3",
        "fp8_e4m3",
        "e4m3",
        "bf8",
        "bf8e5m2",
        "fp8_e5m2",
        "fp4",
        "fp4e2m1",
        "fp6",
        "fp6e2m3",
        "bf6",
        "fp6e3m2",
    }
    with pytest.raises(ValueError) as error:
        quant_ir_type("invalid")
    accepted = ast.literal_eval(str(error.value).split("expected one of ")[1])
    assert accepted == sorted(expected)
    assert all(quant_ir_type(name) is not None for name in accepted)


@pytest.mark.parametrize(
    "dtype", ["fp4", "fp4e2m1", "fp6", "fp6e2m3", "bf6", "fp6e3m2"]
)
def test_type_alias_does_not_enable_scalar_quantization(dtype):
    assert quant_ir_type(dtype) is not None
    with pytest.raises(ValueError) as error:
        quant_max_abs(dtype, arch="gfx950")
    assert str(error.value) == (
        f"unsupported quant dtype {dtype!r}; expected one of ['bf8', 'bf8e5m2', "
        "'e4m3', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8e4m3', 'i8', 'int8']"
    )


def test_quant_dtype_annotations_match_scalar_and_logical_contracts():
    from rocke import helpers
    from rocke.helpers import quant

    scalar = {"i8", "fp8e4m3", "bf8e5m2"}
    assert set(get_args(quant.QDType)) == scalar
    assert all(
        quant_max_abs(dtype, arch="gfx950") > 0 for dtype in get_args(quant.QDType)
    )
    logical = scalar | {"fp4e2m1", "fp6e2m3", "fp6e3m2"}
    assert set(get_args(quant.LogicalQDType)) == logical
    assert get_type_hints(ir_to_qdtype)["return"] == quant.LogicalQDType
    assert all(ir_to_qdtype(quant_ir_type(dtype)) == dtype for dtype in logical)
    assert helpers.QDType is quant.QDType
    assert helpers.LogicalQDType is quant.LogicalQDType


def test_u8_and_u4_are_logical_unsigned_types():
    # Types only: they name unsigned data for matrix instructions with a signedness flag.
    import rocke
    from rocke.core.ir import U4, U8, IRBuilder, PtrType
    from rocke.tests.engines import (
        lower_hip_with_both_engines,
        lower_with_both_engines,
    )
    from rocke.tests.ir_validity import assert_target_accepts

    for spelling, ir_type, bits in (
        ("u8", U8, 8),
        ("uint8", U8, 8),
        ("u4", U4, 4),
        ("uint4", U4, 4),
    ):
        assert dtype_to_ir_type(spelling) is ir_type
        assert dtype_info(spelling).encoded_bits == bits
        assert dtype_info(spelling).category.value == "integer"
    assert U8 != I8
    assert rocke.U8 is U8 and rocke.U4 is U4
    assert storage_ir_type("u4") is I8
    # U8 values are LLVM i8 (LLVM integers carry no sign) and HIP uint8_t.
    b = IRBuilder("u8_pointer")
    src = b.param("SRC", PtrType(U8, "global"))
    dst = b.param("DST", PtrType(U8, "global"))
    tid = b.thread_id_x()
    loaded = b.global_load(src, tid, U8)
    b.global_store(dst, tid, loaded)
    ir_text = lower_with_both_engines(b.kernel, "gfx90a")
    assert "load i8, ptr addrspace(1)" in ir_text
    assert_target_accepts(ir_text, "gfx90a")
    hip = lower_hip_with_both_engines(b.kernel, "gfx90a")
    assert "uint8_t* __restrict__ SRC" in hip


# ---------------------------------------------------------------------------
# Scalar quantization: the bound of the target's encoding and the NaN policy.
# ---------------------------------------------------------------------------

_QUANT_ARCHS = ("gfx942", "gfx950")
_NO_GFX90A_ENCODING = (
    "fp8_element_format failed - arch='gfx90a', expected one of "
    "('gfx942', 'gfx950', 'gfx1250')"
)


@pytest.mark.parametrize(
    "qdtype, arch, bound",
    [
        ("fp8e4m3", "gfx942", 240.0),
        ("fp8", "gfx942", 240.0),
        ("fp8e4m3", "gfx950", 448.0),
        ("fp8e4m3", "gfx1250", 448.0),
        ("bf8e5m2", "gfx942", 57344.0),
        ("bf8e5m2", "gfx950", 57344.0),
        ("i8", "gfx942", 127.0),
        ("i8", "gfx90a", 127.0),
    ],
)
def test_quant_max_abs_is_the_bound_of_the_target_encoding(qdtype, arch, bound):
    """The C++ engine's rocke_quant_max_abs gives the same bounds
    (tests/core/test_storage.cpp)."""
    assert quant_max_abs(qdtype, arch=arch) == bound


def test_quant_max_abs_refuses_fp8_on_a_target_without_an_encoding():
    with pytest.raises(ValueError) as error:
        quant_max_abs("fp8e4m3", arch="gfx90a")
    assert str(error.value) == _NO_GFX90A_ENCODING


def _quantize_kernel(
    qdtype: str,
    arch: str,
    *,
    mode: Literal["scalar", "prescaled", "convert"] = "scalar",
):
    """Y[i] = quantize_scalar_f32(X[i], inv_scale=1, qdtype, arch), one lane
    per element; with ``mode="prescaled"``, quantize_prescaled_f32(X[i],
    qdtype, arch); with ``mode="convert"``, the bare f32 -> fp8 / bf8
    conversion of X[i]."""
    b = IRBuilder(f"{mode}_{qdtype}_{arch}")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(ir.F32, "global"))
    y = b.param("Y", PtrType(quant_ir_type(qdtype), "global"))
    lane = b.thread_id_x()
    value = b.global_load_f32(x, lane)
    if mode == "prescaled":
        result = quantize_prescaled_f32(b, value, qdtype=qdtype, arch=arch)
    elif mode == "scalar":
        result = quantize_scalar_f32(
            b, value, inv_scale=b.const_f32(1.0), qdtype=qdtype, arch=arch
        )
    elif qdtype == "fp8e4m3":
        result = b.cvt_f32_to_fp8(value)
    else:
        result = b.cvt_f32_to_bf8(value)
    b.global_store(y, lane, result, align=1)
    b.ret()
    return b.kernel


# Each quantize helper as (its _quantize_kernel mode, the value its clamp
# reads, the number of the clamp's first register): quantize_scalar_f32 clamps
# its multiply, quantize_prescaled_f32 the loaded value itself.
_QUANTIZE_HELPERS = [("scalar", "%fmul_6", 7), ("prescaled", "%gl_2", 5)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "mode, scaled, first", _QUANTIZE_HELPERS, ids=["scalar", "prescaled"]
)
@pytest.mark.parametrize(
    "qdtype, arch, cvt",
    [
        ("fp8e4m3", "gfx942", "@llvm.amdgcn.cvt.pk.fp8.f32"),
        ("fp8e4m3", "gfx950", "@llvm.amdgcn.cvt.pk.fp8.f32"),
        ("bf8e5m2", "gfx942", "@llvm.amdgcn.cvt.pk.bf8.f32"),
    ],
)
def test_quantize_helpers_clamp_to_the_target_bound_and_keep_nan(
    engine, qdtype, arch, cvt, mode, scaled, first
):
    """The clamp of quantize_scalar_f32 and quantize_prescaled_f32 uses the
    bound of ``arch``'s encoding, and a NaN bypasses it: the conversion reads
    ``isnan(scaled) ? scaled : clamped``."""
    bound = quant_max_abs(qdtype, arch=arch)
    kernel = _quantize_kernel(qdtype, arch, mode=mode)
    text = lower_with_engine(kernel, arch, engine=engine)
    fmax, fmin, fcmp, sel = (
        f"%{name}_{first + offset}"
        for offset, name in enumerate(("fmax", "fmin", "fcmp", "sel"))
    )
    assert (
        f"  {fmax} = call float @llvm.maxnum.f32("
        f"float {llvm_f32_literal(-bound)}, float {scaled})\n"
        f"  {fmin} = call float @llvm.minnum.f32(float {llvm_f32_literal(bound)}, "
        f"float {fmax})\n"
        f"  {fcmp} = fcmp uno float {scaled}, {scaled}\n"
        f"  {sel} = select i1 {fcmp}, float {scaled}, float {fmin}\n"
    ) in text
    assert f"call i32 {cvt}(float {sel}, " in text


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "mode, scaled, first", _QUANTIZE_HELPERS, ids=["scalar", "prescaled"]
)
def test_quantize_helpers_clamp_i8_without_a_nan_select(engine, mode, scaled, first):
    """i8 has no NaN: the clamp alone feeds the conversion (a NaN gives -127),
    in quantize_scalar_f32 and quantize_prescaled_f32."""
    kernel = _quantize_kernel("i8", "gfx942", mode=mode)
    text = lower_with_engine(kernel, "gfx942", engine=engine)
    assert "fcmp uno" not in text
    assert f"call float @llvm.rint.f32(float %fmin_{first + 1})" in text


@pytest.mark.parametrize(
    "inputs, message",
    [
        (("i32", "f32"), "quantize_scalar_f32 expects f32 input, got i32"),
        (("f32", "i32"), "quantize_scalar_f32 expects f32 inv_scale, got i32"),
        (("f32", "f32"), _NO_GFX90A_ENCODING),
    ],
)
def test_quantize_scalar_f32_refusals(inputs, message):
    """The C++ engine's rocke_b_quantize_scalar_f32 refuses with the same texts
    (tests/core/test_storage.cpp); the quantize_prescaled_f32 check is Python
    only."""
    b = IRBuilder("quantize_refusal")
    constants = {"f32": b.const_f32(1.0), "i32": b.const_i32(0)}
    value, inv_scale = (constants[name] for name in inputs)
    with pytest.raises(ValueError) as error:
        quantize_scalar_f32(
            b, value, inv_scale=inv_scale, qdtype="fp8e4m3", arch="gfx90a"
        )
    assert str(error.value) == message
    if inputs[0] != "f32":
        # quantize_prescaled_f32, the clamp and conversion without the multiply,
        # refuses a non-f32 value the same way (Python only: no C++ caller).
        with pytest.raises(ValueError) as error:
            quantize_prescaled_f32(b, value, qdtype="fp8e4m3", arch="gfx950")
        assert str(error.value) == (
            f"quantize_prescaled_f32 expects f32 input, got {inputs[0]}"
        )


def _quant_instances():
    """(family, build, is_valid_spec, spec type, extra fields): the three
    instances that quantize with quant_max_abs."""
    from rocke.instances.common import add_rmsnorm2d_rdquant as rdquant
    from rocke.instances.common import moe_smoothquant, smoothquant

    return (
        (
            "smoothquant",
            smoothquant.build_smoothquant,
            smoothquant.is_valid_spec,
            smoothquant.SmoothQuantSpec,
            {},
        ),
        (
            "moe_smoothquant",
            moe_smoothquant.build_moe_smoothquant,
            moe_smoothquant.is_valid_spec,
            moe_smoothquant.MoeSmoothQuantSpec,
            {"topk": 2, "experts": 8},
        ),
        (
            "add_rmsnorm2d_rdquant",
            rdquant.build_add_rmsnorm2d_rdquant,
            rdquant.is_valid_spec,
            rdquant.AddRmsnorm2DRdquantSpec,
            {},
        ),
    )


def _quant_spec(case, *, vec: int):
    _, _, _, spec_type, extra = case
    return spec_type(
        n_per_block=512,
        dtype="f16",
        out_dtype="fp8e4m3",
        block_size=128,
        vec=vec,
        **extra,
    )


@pytest.mark.parametrize(
    "arch, other_arch", [("gfx942", "gfx950"), ("gfx950", "gfx942")]
)
@pytest.mark.parametrize("vec", [2, 4])
@pytest.mark.parametrize("case", _quant_instances(), ids=lambda case: case[0])
def test_quant_instances_scale_to_the_target_fp8_bound(case, vec, arch, other_arch):
    """The row scale ``amax * (1 / bound)`` uses the bound of ``arch``'s e4m3
    encoding, on the scalar (vec 2) and the packed (vec 4) path, and the C++
    builder emits the same IR as the Python one."""
    family, build, _, _, _ = case
    text = instance_llvm_from_all_engines(
        _quant_spec(case, vec=vec),
        arch,
        build_kernel=build,
        binding=f"{family}_lower_llvm",
    )
    reciprocal = llvm_f32_literal(1.0 / quant_max_abs("fp8e4m3", arch=arch))
    other = llvm_f32_literal(1.0 / quant_max_abs("fp8e4m3", arch=other_arch))
    assert re.search(rf"= fmul float %\S+, {reciprocal}\n", text)
    assert other not in text


@pytest.mark.parametrize(
    "out_dtype, arch, reason",
    [
        ("fp4", "gfx950", "unsupported out_dtype 'fp4'"),
        # An alias of a quant dtype is refused: the spec takes canonical names.
        ("fp8", "gfx950", "unsupported out_dtype 'fp8'"),
        (
            "fp8e4m3",
            "gfx1151",
            (
                "out_dtype 'fp8e4m3' needs the CDNA-only v_cvt_pk_{fp8,bf8}_f32 "
                "conversion; gfx1151 (family 'rdna') has no fp8/bf8 pack op -- "
                "use out_dtype='i8'"
            ),
        ),
        # gfx90a is CDNA but has no fp8 conversion.
        ("fp8e4m3", "gfx90a", _NO_GFX90A_ENCODING),
    ],
)
@pytest.mark.parametrize("case", _quant_instances(), ids=lambda case: case[0])
def test_quant_instances_refuse_an_out_dtype_the_target_cannot_write(
    case, out_dtype, arch, reason
):
    """``is_valid_spec`` refuses with the reason
    :func:`~rocke.helpers.quant.validate_quant_out_dtype` gives, and both
    engines' builders refuse with that reason."""
    family, build, is_valid_spec, _, _ = case
    spec = dataclasses.replace(_quant_spec(case, vec=4), out_dtype=out_dtype)
    assert validate_quant_out_dtype(out_dtype, arch) == (False, reason)
    assert is_valid_spec(spec, arch) == (False, reason)
    message = f"invalid {family} spec: {reason}"
    with pytest.raises(ValueError) as error:
        build(spec, arch=arch)
    assert str(error.value) == message
    if "cpp" in available_engines():
        from rocke.core.backend import import_engine

        fields = {k: v for k, v in dataclasses.asdict(spec).items() if v is not None}
        with pytest.raises(RuntimeError) as native_error:
            getattr(import_engine(), f"{family}_lower_llvm")(fields, arch=arch)
        assert str(native_error.value).endswith(message)


# The floating-point mode every rocKE kernel runs in, from its kernel descriptor
# (dsl_docs/numerics/README.md, "The MODE register"): round to nearest even,
# subnormals kept, IEEE NaN handling, and FP16_OVFL clear, which is why the
# f32 -> fp8 / bf8 conversions do not saturate and quantize_scalar_f32 clamps.
_KERNEL_FP_MODE = {
    ".amdhsa_float_round_mode_32": 0,
    ".amdhsa_float_round_mode_16_64": 0,
    ".amdhsa_float_denorm_mode_32": 3,
    ".amdhsa_float_denorm_mode_16_64": 3,
    ".amdhsa_dx10_clamp": 1,
    ".amdhsa_ieee_mode": 1,
    ".amdhsa_fp16_overflow": 0,
}


@pytest.mark.parametrize("arch", _QUANT_ARCHS)
def test_quantize_kernel_descriptor_sets_the_documented_fp_mode(arch, tmp_path):
    """Compile the quantize kernel through COMGR and read its kernel
    descriptor back with llvm-objdump: the FP mode fields hold the values the
    numerics policy relies on. Skips when COMGR cannot target ``arch`` or
    llvm-objdump is missing (fails instead under
    ``ROCKE_REQUIRE_COMPILER_TOOLS=1``)."""
    require_comgr_target(arch)
    objdump = find_llvm_tool("llvm-objdump")
    if objdump is None:
        require_tools_or_skip("llvm-objdump not found (ROCM_PATH / PATH)")
    kernel = _quantize_kernel("fp8e4m3", arch)
    code_object = tmp_path / "quantize.hsaco"
    code_object.write_bytes(compile_kernel(kernel, arch=arch).hsaco)
    descriptor = subprocess.run(
        [objdump, "-D", f"--disassemble-symbols={kernel.name}.kd", str(code_object)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    fields = {
        name: int(value)
        for name, value in re.findall(
            r"^\s*(\.amdhsa_\w+)\s+(\d+)$", descriptor, re.MULTILINE
        )
    }
    assert {name: fields.get(name) for name in _KERNEL_FP_MODE} == _KERNEL_FP_MODE


def _host_codec(qdtype: str, arch: str) -> Fp8HostCodec:
    """The host codec of ``qdtype`` in ``arch``'s encoding."""
    return fp8_host_codec(fp8_element_format(qdtype, arch))


def _run_one_lane_per_value(kernel, arch: str, values: np.ndarray) -> np.ndarray:
    """Launch ``kernel`` (X: f32 in, Y: 8-bit codes out) on ``values``; return
    the codes."""
    x = np.zeros(64, dtype=np.float32)
    x[: values.size] = values
    codes = np.zeros(64, dtype=np.uint8)
    launch_with_buffers(
        compile_kernel(kernel, arch=arch),
        [sig_param("X", "f32"), sig_param("Y", "u8")],
        {"X": x, "Y": codes},
        grid=(1, 1, 1),
        block=(64, 1, 1),
    )
    return codes[: values.size]


def _bound_and_step(qdtype: str, arch: str) -> tuple[float, float]:
    """``qdtype``'s bound on ``arch`` and its code step there. Halfway from the
    bound to the next step (which the encoding does not hold) is where the
    conversion alone stops rounding to the bound and gives NaN / +-inf (128
    for i8)."""
    bound = quant_max_abs(qdtype, arch=arch)
    if qdtype == "i8":
        return bound, 1.0
    table = _host_codec(qdtype, arch).decode_table()
    magnitudes = np.unique(np.abs(table[np.isfinite(table)]))
    return bound, float(magnitudes[-1] - magnitudes[-2])


@pytest.mark.parametrize("qdtype", ["fp8e4m3", "bf8e5m2"])
def test_fp8_conversion_does_not_saturate_on_gpu(qdtype):
    """The bare f32 -> fp8 / bf8 conversion gives NaN for a value past the
    halfway point above the largest finite value and for +-inf and NaN, except
    OCP bf8, which gives +-inf for the first four: the reason
    quantize_scalar_f32 clamps."""
    arch = current_device_arch_in(_QUANT_ARCHS, subject="the fp8 / bf8 conversion")
    bound, step = _bound_and_step(qdtype, arch)
    above = bound + 0.6 * step
    values = np.array([above, -above, np.inf, -np.inf, np.nan], np.float32)
    codes = _run_one_lane_per_value(
        _quantize_kernel(qdtype, arch, mode="convert"), arch, values
    )
    if qdtype == "bf8e5m2" and arch == "gfx950":
        # OCP e5m2 infinities: 0x7c / 0xfc; NaN: exponent all ones, mantissa != 0.
        assert codes[:4].tolist() == [0x7C, 0xFC, 0x7C, 0xFC]
        assert (codes[4] & 0x7F) > 0x7C
    else:
        assert np.isnan(_host_codec(qdtype, arch).decode_table()[codes]).all(), codes


@pytest.mark.parametrize("qdtype", ["fp8e4m3", "bf8e5m2", "i8"])
def test_quantize_scalar_f32_saturates_and_keeps_nan_on_gpu(qdtype):
    """quantize_scalar_f32 on the GPU's encoding: values within the bound are
    kept, a value under the halfway point above the bound rounds to the bound,
    one over it, finite values far past it and +-inf saturate to +-bound, zero
    (also -0 and a negative value that rounds to zero) gives zero, never FNUZ's
    NaN code 0x80, and NaN converts to the encoding's NaN; i8 has none, so NaN
    gives -127."""
    arch = current_device_arch_in(_QUANT_ARCHS, subject="quantize_scalar_f32")
    bound, step = _bound_and_step(qdtype, arch)
    below, above = bound + 0.4 * step, bound + 0.6 * step
    values = np.array(
        [
            0.0,
            -0.0,
            -(2.0**-20),
            1.0,
            -2.0,
            bound,
            -bound,
            below,
            -below,
            above,
            -above,
            1e30,
            -1e30,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    expected = np.clip(values, -bound, bound)
    expected[2] = 0.0
    if qdtype == "i8":
        expected[np.isnan(values)] = -bound
    codes = _run_one_lane_per_value(_quantize_kernel(qdtype, arch), arch, values)
    if qdtype == "i8":
        got = codes.view(np.int8).astype(np.float32)
    else:
        got = _host_codec(qdtype, arch).decode_table()[codes]
    np.testing.assert_array_equal(got, expected, err_msg=f"{qdtype} on {arch}")


@pytest.mark.parametrize("vec", [2, 4])
def test_smoothquant_fp8_output_is_exact_and_reaches_the_bound_on_gpu(vec):
    """SmoothQuant scales each row to the bound of the GPU's e4m3 encoding:
    each output is ``x * (1 / yscale)`` rounded to the nearest code (vec 2 goes
    through quantize_scalar_f32, vec 4 through the packed conversion), so every
    output is finite and each row's largest magnitude converts to the bound."""
    from rocke.instances.common.smoothquant import (
        SmoothQuantSpec,
        build_smoothquant,
        smoothquant_signature,
    )

    arch = current_device_arch_in(_QUANT_ARCHS, subject="smoothquant fp8")
    codec = _host_codec("fp8e4m3", arch)
    spec = SmoothQuantSpec(
        n_per_block=512, dtype="f16", out_dtype="fp8e4m3", block_size=128, vec=vec
    )
    rows, cols = 8, spec.n_per_block
    x = np.random.default_rng(2649).uniform(-4.0, 4.0, (rows, cols)).astype(np.float16)
    codes = np.zeros((rows, cols), dtype=np.uint8)
    yscale = np.zeros(rows, dtype=np.float32)
    launch_with_buffers(
        compile_kernel(build_smoothquant(spec, arch=arch), arch=arch),
        smoothquant_signature(spec),
        {"X": x, "SmScale": np.ones(cols, np.float32), "QY": codes, "YScale": yscale},
        grid=(rows, 1, 1),
        block=(spec.block_size, 1, 1),
        scalars={"M": rows, "N": cols, "eps": 1e-6},
    )
    # The kernel's reciprocal is a correctly rounded f32 division and its
    # multiply an f32 multiply, so the host computes the same f32 values.
    inverse = np.float32(1.0) / yscale
    np.testing.assert_array_equal(
        codes, codec.round_to_codes(x.astype(np.float32) * inverse[:, None])
    )
    got = codec.decode_table()[codes]
    np.testing.assert_array_equal(
        np.abs(got).max(axis=1), np.full(rows, quant_max_abs("fp8e4m3", arch=arch))
    )
