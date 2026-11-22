# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single-activation consumer contracts, CPU-only.

Feature: retaining a validated single type, one-body emission, helper identity,
and exact enum whitelist make these pass; normalization to none, broad dispatch,
or dropping the whitelist break them again. Preservation: stock runtime/default
semantics and serialized row bytes, each proven by a scratch production mutation.
The retained gradient acceptance and unsupported-to-none tests are intentionally
replaced by generation refusal: bespoke generation covers dense forward GEMM only.
"""
import copy
from pathlib import Path
import re
import sys

import pytest
import yaml

from tensilelite.Activation import ActivationModule, ActivationType
from tensilelite.Common.DataType import DataType
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit
HERE = Path(__file__).resolve().parent
LOGIC = HERE / "test_data/single_activation/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml"
CODEGEN = HERE / "characterization/_codegen"
FORWARD = ("gelu", "relu", "sigmoid", "silu", "clamp")
ENUMS = {"gelu": "Gelu", "relu": "Relu", "sigmoid": "Sigmoid", "silu": "Silu", "clamp": "Clamp"}
ARGS = {"gelu": [], "relu": [], "sigmoid": [], "silu": [], "clamp": ["activationAlpha", "activationBeta"]}
GELU_MARKER = "0x3d372713"  # float32 0.044715, the GELU polynomial coefficient


def make_pt(**overrides):
    return ProblemType({"DataType": 0, **overrides}, False)


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_is_kept(name):
    pt = make_pt(Activation=True, ActivationType=name, UseBias=0)
    assert pt["ActivationType"] == name
    assert "_%s_S" % name.upper() in str(pt)


def test_single_type_name_is_case_insensitive():
    assert make_pt(Activation=True, ActivationType="Gelu")["ActivationType"] == "gelu"


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_needs_activation_true(name):
    with pytest.raises(SystemExit):
        make_pt(Activation=False, ActivationType=name)


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_needs_activation_enable_flag(name):
    with pytest.raises(SystemExit):
        make_pt(ActivationType=name)


@pytest.mark.parametrize("name", ["abs", "clippedrelu", "leakyrelu", "tanh", "geluscaling", "swish",
                                 "exp", "not_an_activation", "gellu", "hipblaslt_relu", "hipblaslt_gelu"])
def test_unsupported_or_misspelled_explicit_type_is_rejected(name):
    with pytest.raises(SystemExit):
        make_pt(Activation=True, ActivationType=name)


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_rejects_grouped_gemm(name):
    with pytest.raises(SystemExit):
        make_pt(Activation=True, ActivationType=name, GroupedGemm=True)


@pytest.mark.parametrize("name", FORWARD)
@pytest.mark.parametrize("field,value", [("Gradient", True), ("UseE", True), ("OutputAmaxD", True),
                                        ("UseScaleCD", True)])
def test_single_type_rejects_out_of_domain_epilogue(name, field, value):
    with pytest.raises(SystemExit):
        make_pt(Activation=True, ActivationType=name, UseBias=1 if field == "Gradient" else 0,
                **{field: value})


@pytest.mark.parametrize("name", ["dgelu", "drelu"])
def test_gradient_single_type_is_refused(name):
    with pytest.raises(SystemExit):
        make_pt(Activation=True, ActivationType=name, Gradient=True, UseBias=1)


def test_sigmoid_single_type_with_bias_is_refused():
    with pytest.raises(SystemExit):
        make_pt(Activation=True, ActivationType="sigmoid", UseBias=1)


@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("name", ["all", "hipblaslt_all"])
def test_run_time_types_unchanged(name, activation):
    assert make_pt(Activation=activation, ActivationType=name)["ActivationType"] == name


@pytest.mark.parametrize("config", [{}, {"ActivationType": "none"}, {"Activation": True},
                                  {"Activation": False}, {"Activation": False, "ActivationType": "none"}])
def test_default_and_none_unchanged(config):
    assert make_pt(**config)["ActivationType"] == "none"


def test_grouped_gemm_run_time_type_unchanged():
    assert make_pt(Activation=True, ActivationType="hipblaslt_all", GroupedGemm=True)["ActivationType"] == "hipblaslt_all"


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(CODEGEN))
    try:
        import codegen_harness
        yield codegen_harness
    finally:
        sys.path.remove(str(CODEGEN))


def variant(tmp_path, name, *, afc=False, user_args=True):
    doc = yaml.safe_load(LOGIC.read_text())
    doc["ProblemType"].update(Activation=name != "none", ActivationType=name,
                              SupportUserArgs=user_args, UseBias=0 if name == "sigmoid" else 1)
    for sol in doc["Solutions"]:
        sol["ActivationFuncCall"] = afc
        # GlobalSplitU=0 is a derived SK value, not a public YAML request.
        sol["GlobalSplitU"] = 1
        # A single activation is a bespoke request (generation refuses it without the key).
        sol["BespokeNarrowing"] = name not in ("none", "all", "hipblaslt_all")
    path = tmp_path / (name + ("_afc" if afc else "") + ("_ua" if user_args else "") + ".yaml")
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_register_usage_matches_hipblaslt_all(name, harness):
    with harness._isolated_globals():
        harness.get_assembler()
        harness._init_rocisa_for({"ISA": (9, 5, 0), "WavefrontSize": 64})
        single = ActivationModule().getAllGprUsage(DataType("s"), ActivationType(name))
        general = ActivationModule().getAllGprUsage(DataType("s"), ActivationType("hipblaslt_all"))
        assert name in single, single
        assert single == {name: general[name]}
        if name in ("gelu", "silu"):
            assert single[name]["vgpr"] >= 1


@pytest.mark.parametrize("name", FORWARD)
def test_contraction_predicates_for_single_type(name):
    from tensilelite import Contractions
    state = copy.deepcopy(yaml.safe_load(LOGIC.read_text())["ProblemType"])
    state.update(ActivationType=name, UseBias=0 if name == "sigmoid" else 1)
    pt = Contractions.ProblemType.FromOriginalState(state)
    assert {p.tag: p.value for p in pt.predicates(includeType=True)}["Activation"] == name
    assert [p.value for p in pt.predicates(includeOperation=True)
            if p.tag == "ActivationEnumWhiteList"] == [[ENUMS[name]]]
    assert pt.activationArgLength == len(ARGS[name])


def emit(harness, path):
    results = harness.emit_kernels_from_logic(path, canonical=False)
    assert len(results) == 1
    base, src, err = results[0]
    assert err == 0
    return base, src


def assemble(base, src, tmp_path):
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain
    path = tmp_path / (base + ".s")
    path.write_text(src)
    obj = path.with_suffix(".o")
    assembler = makeAssemblyToolchain(validateToolchain("amdclang++"),
                                     validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER), "4").assembler
    assembler("gfx950", 64, str(path), str(obj))
    assert obj.stat().st_size > 0


@pytest.mark.parametrize("name", FORWARD)
@pytest.mark.parametrize("user_args", [True, False], ids=["userargs", "kernargs"])
def test_single_type_kernel(name, user_args, harness, tmp_path):
    base, src = emit(harness, variant(tmp_path, name, user_args=user_args))
    assert "_%s_S_" % name.upper() in base
    assert "sgprActivationType" not in src
    assert not re.search(r"^label_Activation_", src, re.M)
    assert re.findall(r"\.name:\s+(activation\w*)", src) == ARGS[name]
    if name != "gelu":
        assert GELU_MARKER not in src
    assemble(base, src, tmp_path)


def test_gelu_kernel_has_one_body_per_store_path(harness, tmp_path):
    _, ha = emit(harness, variant(tmp_path, "hipblaslt_all"))
    _, gelu = emit(harness, variant(tmp_path, "gelu"))
    _, none = emit(harness, variant(tmp_path, "none"))
    assert re.search(r"^label_Activation_Gelu\w*:", ha, re.M)
    assert gelu.count(GELU_MARKER) == ha.count(GELU_MARKER) > 0
    assert none.count(GELU_MARKER) == 0
    assert len(none) < len(gelu) < len(ha)
    assert "sgprActivationType" in ha
    assert re.findall(r"\.name:\s+(activation\w*)", ha) == ["activationAlpha", "activationBeta", "activationType"]


@pytest.mark.parametrize("name,opcode", [("gelu", r"v_exp_f32"), ("relu", r"v_(?:pk_)?max_f32"),
                                      ("sigmoid", r"v_exp_f32"), ("silu", r"v_exp_f32"),
                                      ("clamp", r"v_(?:pk_)?min_f32")])
def test_inline_kernel_contains_requested_activation_body(name, opcode, harness, tmp_path):
    single_path = variant(tmp_path, name)
    none_path = variant(tmp_path, "none")
    if name == "sigmoid":
        control = yaml.safe_load(none_path.read_text())
        control["ProblemType"]["UseBias"] = 0
        none_path.write_text(yaml.safe_dump(control, sort_keys=False))
    _, single = emit(harness, single_path)
    _, none = emit(harness, none_path)
    assert len(re.findall(r"^" + opcode + r"\b", single, re.M)) > len(re.findall(r"^" + opcode + r"\b", none, re.M))


@pytest.mark.parametrize("name", FORWARD)
def test_activation_loop_has_one_body(name, harness, tmp_path):
    from rocisa.label import LabelManager
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    class Writer:
        labels = LabelManager()

    sol, = harness.solutions_from_logic(variant(tmp_path, name))
    for beta in (False, True):
        _, labels, enums = KernelWriterAssembly.initActivationLoop(Writer(), sol, beta)
        assert labels == [""]
        assert enums == [name]


@pytest.mark.parametrize("name", FORWARD)
def test_single_type_kernel_keeps_function_call(name, harness, tmp_path):
    path = variant(tmp_path, name, afc=True)
    sol, = harness.solutions_from_logic(path)
    assert sol["ActivationFuncCall"]
    base, src = emit(harness, path)
    bodies = re.findall(r"^label_Activation_(\w+)_VW\d+:", src, re.M)
    assert bodies and set(bodies) == {ENUMS[name]}
    assert "sgprActivationType" not in src
    assert "activationType ==" not in src
    assemble(base, src, tmp_path)


def helpers(harness, path):
    from tensilelite.TensileCreateLibrary.Run import generateKernelHelperObjects, generateKernelObjectsFromSolutions
    with harness._isolated_globals():
        sols = harness._solutions_from_logic_unguarded(path)
        kernels = generateKernelObjectsFromSolutions(sols)
        out = {}
        for ko in generateKernelHelperObjects(kernels, str(harness.get_assembler().path), harness.get_isa_info_map()):
            err, src = ko.getSourceFileString()
            assert err == 0
            out[ko.getKernelName()] = (src, ko.getHeaderFileString())
        return out


@pytest.mark.parametrize("name", ["gelu", "clamp"])
def test_single_type_helper_kernels(name, harness, tmp_path):
    generated = helpers(harness, variant(tmp_path, name))
    wanted = "TensileActivation_S_" + ENUMS[name]
    assert wanted in generated, list(generated)
    header = generated[wanted][1]
    assert "activation_" + ENUMS[name] + "(" in header
    assert "activationType" not in header
    assert not any(n.startswith("TensileActivationEnum") for n in generated)
    conversions = {n: s for n, (s, _) in generated.items() if "_PostGSU" in n}
    assert conversions
    for kernel_name, src in conversions.items():
        assert "_%sS_" % name.upper() in kernel_name
        assert "activation_" + ENUMS[name] + "(" in src
        assert "arg.activationType" not in src


@pytest.mark.parametrize("lazy", [True, False], ids=["placeholder", "matching"])
@pytest.mark.parametrize("format", ["yaml", "msgpack"])
def test_stock_row_serialized_bytes_unchanged(lazy, format, tmp_path):
    from tensilelite import Contractions, LibraryIO, SolutionLibrary
    from tensilelite.Common.Utilities import state
    names = ["TensileLibrary_BB_BB_Bias_SAV_UA", "TensileLibrary_BB_BB_HA_Bias_SAV_UA"]
    rows = []
    expected_rows = []
    for name, act in zip(names, ["none", "hipblaslt_all"]):
        child = (SolutionLibrary.PlaceholderLibrary(name) if lazy else
                 SolutionLibrary.MatchingLibrary([], [], "Equality"))
        pred = Contractions.ProblemPredicate("Activation", value=ActivationType(act))
        rows.append({"predicate": pred, "library": child})
        expected_rows.append({"predicate": {"type": "Activation", "value": "None" if act == "none" else "Hipblaslt_all"},
                              "library": {"type": "Placeholder", "value": name} if lazy else
                              {"type": "Matching", "properties": [], "table": [], "distance": "Equality", "useKdTree": False}})
    if not lazy:
        # Stock nonlazy ordering compares the activation strings: hipblaslt_all < none.
        expected_rows = [expected_rows[1], expected_rows[0]]
    actual = SolutionLibrary.PredicateLibrary(tag="ProblemType", rows=[rows[1]])
    actual.merge(SolutionLibrary.PredicateLibrary(tag="ProblemType", rows=[rows[0]]))
    expected = {"type": "ProblemType", "rows": expected_rows}
    LibraryIO.write(str(tmp_path / "actual"), state(actual), format)
    LibraryIO.write(str(tmp_path / "expected"), expected, format)
    suffix = ".yaml" if format == "yaml" else ".dat.zlib"
    assert (tmp_path / ("actual" + suffix)).read_bytes() == (tmp_path / ("expected" + suffix)).read_bytes()
