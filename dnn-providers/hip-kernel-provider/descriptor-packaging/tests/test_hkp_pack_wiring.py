"""The CMake wiring of a build without rocKE, driven by real sub-configures.

A synthetic consumer includes `HkpPackaging.cmake` and calls its functions, as
`tests/rocke/test_hkp_python_environment.py` does for the rocKE wheel lifecycle.
Here no rocKE wheel, comgr or private import directory exists, and the supplied
interpreter has no pip: a pack wired with `ENABLE_ROCKE OFF` must need none of
them.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cmake_harness import PKG, SuppliedPython, consumer_preamble
from pack_helpers import GENERIC_TARGETS_JSON

pytestmark = pytest.mark.quick

# Stands in for hkp_pack.py: records how the pack step launched it.
CONSUMER = """
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
out_root = Path(argv[argv.index('--out-root') + 1])
out_root.mkdir(parents=True, exist_ok=True)
(out_root / 'invocation.json').write_text(json.dumps({
    'python': sys.executable,
    'argv': argv,
    'env': {key: os.environ.get(key) for key in
            ('ROCKE_BACKEND', 'ROCKE_CPP_STRICT', 'ROCKE_COMGR_LIB',
             'HKP_PACK_JOBS', 'PYTHONPATH', 'HKP_OP_SCHEMA_REGISTRY',
             'HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE')},
}), encoding='utf-8')
"""


class _Consumer(SuppliedPython):
    def __init__(self, root, *, cmake, make_program):
        self.cmake = cmake
        self.make_program = make_program
        self.source = root / "source with spaces"
        self.build_dir = root / "build with spaces"
        self.source.mkdir(parents=True)
        super().__init__(root, pip=False)
        (self.source / "consumer.py").write_text(CONSUMER, encoding="utf-8")
        (self.source / "authored").mkdir()
        (self.source / "kpack with spaces" / "rocm_kpack").mkdir(parents=True)

    def wire(self, rocke_keywords, *, prelude=""):
        """One root, wired with `rocke_keywords` spliced into the call, after
        `prelude`."""
        self._write(
            prelude
            + """set(HKP_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/consumer.py")
hkp_wire_pack_target(
    NAME off
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/authored"
    OUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/off"
    ARCHES gfx942 HIPCC unused
    ROCM_KPACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kpack with spaces"
    """
            + rocke_keywords
            + ")\n"
        )

    def register_tests(self, enable_rocke):
        self._write(
            f"""enable_testing()
set(HIPKERNELPROVIDER_ENABLE_TESTS ON)
set(HIPKERNELPROVIDER_ENABLE_ROCKE {enable_rocke})
hkp_register_tests("" unused "")
"""
        )

    def _write(self, body):
        (self.source / "CMakeLists.txt").write_text(
            consumer_preamble("HkpPackWiring") + body, encoding="utf-8"
        )

    def configure(self, *definitions, python=None, success=True):
        proc = self.run(
            self.cmake,
            "-S",
            self.source,
            "-B",
            self.build_dir,
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={self.make_program}",
            f"-DPython3_EXECUTABLE={python or self.python}",
            *definitions,
            success=success,
        )
        return " ".join((proc.stdout + proc.stderr).split())

    def build(self):
        self.run(self.cmake, "--build", self.build_dir, "--target", "hkp_packaging_off")

    def invocation(self):
        return json.loads(
            (self.build_dir / "off" / "invocation.json").read_text(encoding="utf-8")
        )

    def registered_commands(self):
        ctest = Path(self.cmake).with_name("ctest" + Path(self.cmake).suffix)
        proc = self.run(ctest, "--show-only=json-v1", "--test-dir", self.build_dir)
        return {t["name"]: t["command"] for t in json.loads(proc.stdout)["tests"]}


@pytest.fixture
def consumer(tmp_path, cmake, cmake_make_program):
    return _Consumer(tmp_path, cmake=cmake, make_program=cmake_make_program)


def test_a_root_without_rocke_packs_under_a_base_interpreter_without_pip(consumer):
    """The pack runs under the supplied interpreter with no rocKE environment, no
    private import directory on PYTHONPATH, and rocke pruned as a disabled kind in
    place of the wheel stamp; each excluded family folder reaches the tool. pip is
    absent, so any rocKE wheel step would fail configure or build."""
    consumer.run(consumer.python, "-m", "pip", "--version", success=False)
    consumer.wire("ENABLE_ROCKE OFF EXCLUDE_FOLDERS rocKE asm PACK_JOBS 2")
    consumer.configure()
    consumer.build()

    record = consumer.invocation()
    assert Path(record["python"]) == consumer.python
    pairs = list(zip(record["argv"], record["argv"][1:]))
    assert ("--disable-kind", "rocke") in pairs
    assert ("--exclude-folder", "rocKE") in pairs
    assert ("--exclude-folder", "asm") in pairs
    assert "--rocke-wheel-stamp" not in record["argv"]
    # The harness strips PYTHONPATH from the build's environment, so any value
    # the pack sees is a prepend the command made.
    assert record["env"] == {
        "ROCKE_BACKEND": None,
        "ROCKE_CPP_STRICT": None,
        "ROCKE_COMGR_LIB": None,
        "HKP_PACK_JOBS": "2",
        "PYTHONPATH": None,
        "HKP_OP_SCHEMA_REGISTRY": None,
        "HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE": None,
    }
    assert "path_list_prepend" not in (consumer.build_dir / "build.ninja").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "rocke_keywords",
    [
        'ROCKE_INTERP "x"',
        'ROCKE_INTERP ""',
        "ROCKE_INTERP PACK_JOBS 2",
    ],
    ids=["valued", "empty-value", "no-value"],
)
def test_a_rocke_keyword_on_a_root_without_rocke_fails_configure(
    consumer, rocke_keywords
):
    """Each ROCKE_* keyword names a toolchain the build does not have, however it
    is spelled: with a value, with an empty one, or with none."""
    consumer.wire(f"ENABLE_ROCKE OFF {rocke_keywords}")
    diagnostic = consumer.configure(success=False)
    assert "disables rocKE but was wired with ROCKE_INTERP" in diagnostic


def test_a_root_wired_without_enable_rocke_fails_configure(consumer):
    """Neither mode is reachable by omission."""
    consumer.wire("PACK_JOBS 2")
    diagnostic = consumer.configure(success=False)
    assert "was wired without ENABLE_ROCKE" in diagnostic


def _collected_ids(command, *, drop_marker=False):
    """The test ids a registered pytest command selects, listed rather than run."""
    argv = [a for a in command if a != "-v"]
    if drop_marker:
        # The `-m <expression>` after the `-m pytest` that starts the module.
        at = argv.index("-m", argv.index("-m") + 1)
        del argv[at : at + 2]
    proc = subprocess.run(
        [*argv, "--collect-only", "-q"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {line for line in proc.stdout.splitlines() if "::" in line}


@pytest.mark.parametrize("enable_rocke", ["ON", "OFF"])
def test_the_registered_suites_collect_tests_rocke_only_with_rocke(
    consumer, enable_rocke
):
    """The two registered pytest entries split the suite between them -- every test
    in exactly one -- and select the tests under tests/rocke/ if and only if the
    build has rocKE. Configured under this interpreter, which can import pytest as
    hkp_register_tests requires."""
    consumer.register_tests(enable_rocke)
    consumer.configure(python=sys.executable)
    commands = consumer.registered_commands()
    assert set(commands) == {
        "hip-kernel-provider-hkp-pack-quick",
        "hip-kernel-provider-hkp-pack",
    }

    quick = _collected_ids(commands["hip-kernel-provider-hkp-pack-quick"])
    standard = _collected_ids(commands["hip-kernel-provider-hkp-pack"])
    everything = _collected_ids(
        commands["hip-kernel-provider-hkp-pack"], drop_marker=True
    )
    assert quick and standard
    assert not quick & standard
    assert quick | standard == everything

    rocke_ids = {i for i in everything if i.startswith("tests/rocke/")}
    assert bool(rocke_ids) == (enable_rocke == "ON")
    assert {i for i in quick | standard if i.startswith("tests/rocke/")} == rocke_ids


@pytest.mark.parametrize(
    ("enable_rocke", "folders", "expected", "offered"),
    [
        (
            "ON",
            "",
            {"gfx950": ["test_fixture:attention"], "gfx942": []},
            ["test_fixture:attention"],
        ),
        ("OFF", "", {"gfx950": [], "gfx942": []}, []),
        ("ON", "rocKE", {"gfx950": [], "gfx942": []}, []),
    ],
    ids=["rocke-on", "kind-disabled", "folder-excluded"],
)
def test_the_root_probe_answers_under_a_base_interpreter_without_pip(
    consumer, rocke_fixture, enable_rocke, folders, expected, offered
):
    """The configure-time probe that decides dormancy and engine availability runs
    the packer's own selection under the pip-less interpreter, through a path with
    spaces, with the build's filters. The fixture's rocKE kernel is scoped to
    gfx950, so only an unfiltered gfx950 ships its engine, while the offered list
    names it whatever arches the build packs; a probe that could not run would
    print no JSON at all."""
    shutil.copytree(rocke_fixture, consumer.source / "authored" / "rocKE" / "attention")
    kinds = "rocke" if enable_rocke == "OFF" else ""
    consumer._write(
        f"""_hkp_root_probe(_shipped _offered _ok "${{CMAKE_CURRENT_SOURCE_DIR}}/authored"
    "gfx950;gfx942" "{folders}" "{kinds}")
{_report("ok", "${_ok}")}
{_report("shipped", "${_shipped}")}
{_report("offered", "${_offered}")}
"""
    )
    output = consumer.configure()
    assert _reported(output, "ok") == "TRUE"
    assert json.loads(_reported(output, "shipped")) == expected
    assert _reported(output, "offered") == ";".join(offered)


def _report(key, value):
    return f'message(STATUS "@{key}={value}@")'


def _reported(output, key):
    return re.search(rf"@{key}=(.*?)@", output).group(1)


def _cmake_string(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _wildcard_pointwise(empty_arch_fixture, destination):
    """A root of one wildcard-arch KDP holding an embedded_source UKD, so it ships
    for whatever arches a build packs and its engine is `test_fixture:solo`."""
    shutil.copytree(empty_arch_fixture, destination)
    kdp_path = destination / "solo.kdp.json"
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    kdp["arch"] = []
    kdp["kernelDescriptors"][0]["kernel_source"] = {
        "kind": "embedded_source",
        "source_file": "PointwiseAdd.cpp",
        "entry_point": "PointwiseAdd",
    }
    kdp_path.write_text(json.dumps(kdp), encoding="utf-8")


def _wire_root(
    name="off", *, arches="gfx942", enable_rocke="OFF", keywords="", exclude=""
):
    """One `_hkp_wire_root` call over the consumer's authored root, with
    EXCLUDE_FOLDERS last as the function requires."""
    return f"""set(HKP_TOOL "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer.py")
_hkp_wire_root(
    NAME {name}
    SOURCE_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/authored"
    OUT_ROOT "${{CMAKE_CURRENT_BINARY_DIR}}/{name}"
    ARCHES "{arches}"
    ENABLE_ROCKE {enable_rocke}
    HIPCC unused
    ROCM_KPACK_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/kpack with spaces"
    {keywords}
    EXCLUDE_FOLDERS {exclude})
"""


def test_the_op_schema_registry_follows_its_default_until_overridden(consumer):
    """An unset registry follows the default across reconfigures, as when the
    build is pointed at a different hipDNN; an explicit value survives a later
    change of the default."""
    registries = {name: consumer.root / f"{name}.json" for name in "abcd"}
    for path in registries.values():
        path.write_text("{}", encoding="utf-8")
    consumer._write(
        f"""add_library(hipdnn_flatbuffers_sdk INTERFACE IMPORTED)
set(HIPDNN_FLATBUFFERS_SDK_OP_SCHEMA_REGISTRY "${{HKP_TEST_DEFAULT}}")
_hkp_resolve_op_schema_registry()
{_report("registry", "${HIPKERNELPROVIDER_OP_SCHEMA_REGISTRY}")}
"""
    )

    def registry(definition, name):
        output = consumer.configure(f"-D{definition}={registries[name]}")
        return Path(_reported(output, "registry"))

    assert registry("HKP_TEST_DEFAULT", "a") == registries["a"]
    assert registry("HKP_TEST_DEFAULT", "b") == registries["b"]
    assert registry("HIPKERNELPROVIDER_OP_SCHEMA_REGISTRY", "c") == registries["c"]
    assert registry("HKP_TEST_DEFAULT", "d") == registries["c"]


@pytest.mark.parametrize(
    ("option", "expected"),
    [("OFF", "fam"), ("ON", "")],
    ids=["option-off", "option-on"],
)
def test_a_family_folder_is_disabled_exactly_when_its_option_is_off(
    consumer, option, expected
):
    """The family table names a folder and the option that enables it."""
    consumer._write(
        f"""option(HKP_TEST_FAM_OPT "" {option})
set(HKP_DESCRIPTOR_FAMILIES "fam=HKP_TEST_FAM_OPT")
_hkp_disabled_families(_folders)
{_report("folders", "${_folders}")}
"""
    )
    assert _reported(consumer.configure(), "folders") == expected


@pytest.mark.parametrize(
    "entry",
    ["fam=HKP_TEST_FAM_OPT", "fam", "fam=a=b", "=HKP_TEST_FAM_OPT"],
    ids=["undefined-option", "no-option", "two-equals", "no-folder"],
)
def test_a_malformed_family_table_fails_configure(consumer, entry):
    """A family entry that is not `<folder>=<option>`, or whose option nothing
    defines, is an error naming the entry, rather than silently excluding its
    folder."""
    consumer._write(
        f"""set(HKP_DESCRIPTOR_FAMILIES "{entry}")
_hkp_disabled_families(_folders)
"""
    )
    diagnostic = consumer.configure(success=False)
    assert f"HKP_DESCRIPTOR_FAMILIES entry '{entry}'" in diagnostic


def test_a_root_with_nothing_to_pack_is_dormant_and_forwards_every_keyword_when_wired(
    consumer, empty_arch_fixture
):
    """A test root whose only content is in an excluded family folder is a skip end
    to end: no pack target, a dormant label, its stale output removed, and one
    STATUS line naming the folders that exist. With no folder excluded the same root
    is wired, and every keyword reaches the pack: the arch list stays one value and
    PACK_JOBS, which precedes EXCLUDE_FOLDERS, is not swallowed by it."""
    _wildcard_pointwise(
        empty_arch_fixture, consumer.source / "authored" / "fam" / "pointwise"
    )
    state = f"""
get_property(_dormant GLOBAL PROPERTY HKP_PACK_DORMANT_LABELS)
get_property(_shipped GLOBAL PROPERTY HKP_PACK_SHIPPED_ENGINES_off SET)
if(TARGET hkp_packaging_off)
    set(_target TRUE)
else()
    set(_target FALSE)
endif()
{_report("dormant", "${_dormant}")}
{_report("shipped_set", "${_shipped}")}
{_report("target", "${_target}")}
"""
    keywords = "PACK_JOBS 1"
    arches = "gfx942;gfx950"

    stale = consumer.build_dir / "off"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("left by an earlier configuration")
    consumer._write(
        _wire_root(arches=arches, keywords=keywords, exclude="fam other") + state
    )
    output = consumer.configure()
    assert _reported(output, "target") == "FALSE"
    assert _reported(output, "dormant") == "off"
    assert _reported(output, "shipped_set") == "0"
    assert not stale.exists()
    line = re.search(r"hkp: root 'off'.*?(?= -- |$)", output).group(0)
    assert "disabled folder(s) fam excluded" in line
    assert not re.search(r"\bother\b", line)

    consumer._write(_wire_root(arches=arches, keywords=keywords, exclude="") + state)
    output = consumer.configure()
    assert _reported(output, "target") == "TRUE"
    assert _reported(output, "shipped_set") == "1"
    consumer.build()
    record = consumer.invocation()
    pairs = list(zip(record["argv"], record["argv"][1:]))
    assert ("--arches", "gfx942,gfx950") in pairs
    assert record["env"]["HKP_PACK_JOBS"] == "1"


@pytest.mark.parametrize("enable_rocke", ["ON", "OFF"])
def test_the_arch_aware_predicate_reads_the_probe_and_falls_back_to_the_rocke_option(
    consumer, enable_rocke
):
    """gfx950 dense attention is available when the product pack ships it for
    gfx950. A probe that could not answer reads as HIPKERNELPROVIDER_ENABLE_ROCKE,
    since the bundle is rocKE content."""

    def answer(shipped, probe_ok):
        return f"""set(HIPKERNELPROVIDER_ENABLE_ROCKE {enable_rocke})
set_property(GLOBAL PROPERTY HKP_PACK_LABELS product)
set_property(GLOBAL PROPERTY HKP_PACK_ARCHES_product gfx950)
set_property(GLOBAL PROPERTY HKP_PACK_PROBE_OK_product {probe_ok})
set_property(GLOBAL PROPERTY HKP_PACK_SHIPPED_ENGINES_product
    "{_cmake_string(json.dumps(shipped))}")
hkp_gfx950_attention_dense_available(_available)
if(_available)
    set(_available TRUE)
else()
    set(_available FALSE)
endif()
{_report("available", "${_available}")}
"""

    def evaluate(shipped, probe_ok):
        consumer._write(answer(shipped, probe_ok))
        return _reported(consumer.configure(), "available")

    assert evaluate({"gfx950": ["x:Other"]}, "TRUE") == "FALSE"
    assert evaluate({"gfx950": ["hipkernel:Gfx950AttentionDense"]}, "TRUE") == "TRUE"
    assert evaluate({}, "FALSE") == ("TRUE" if enable_rocke == "ON" else "FALSE")


def test_a_probe_failure_prints_the_packers_reason_and_wires_the_root(consumer):
    """A root the packer cannot load is still wired, so the pack reports the
    problem, and configure says why the probe could not answer."""
    broken = consumer.source / "authored" / "x"
    broken.mkdir(parents=True)
    (broken / "broken.kdp.json").write_text("{", encoding="utf-8")
    consumer._write(
        _wire_root()
        + f"""if(TARGET hkp_packaging_off)
    {_report("target", "TRUE")}
endif()
"""
    )
    output = consumer.configure()
    assert "could not ask the packer" in output
    assert "malformed descriptor JSON" in output
    assert _reported(output, "target") == "TRUE"


def test_hidden_paths_and_excluded_folders_are_neither_pack_inputs_nor_configure_dependencies(
    consumer, empty_arch_fixture
):
    """Editing a file the packer never reads must not repack or reconfigure: the
    input manifest and the probe's configure dependencies skip hidden paths and the
    excluded family folders."""
    authored = consumer.source / "authored"
    _wildcard_pointwise(empty_arch_fixture, authored / "pointwise")
    for hidden in (authored / ".hidden", authored / "fam"):
        hidden.mkdir()
    (authored / ".hidden" / "x.kdp.json").write_text("{", encoding="utf-8")
    (authored / "fam" / "y.kdp.json").write_text("{", encoding="utf-8")
    consumer._write(
        _wire_root(exclude="fam")
        + f"""get_property(_deps DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
{_report("deps", "${_deps}")}
"""
    )
    output = consumer.configure()

    deps = _reported(output, "deps").split(";")
    assert any("/pointwise/" in dep for dep in deps)
    manifest = (consumer.build_dir / "hkp-off-inputs.txt").read_text(encoding="utf-8")
    inputs = manifest.splitlines()
    assert any("/pointwise/" in path for path in inputs)
    for path in [*deps, *inputs]:
        assert "/.hidden/" not in path
        assert "/fam/" not in path


@pytest.mark.parametrize("enable_rocke", ["ON", "OFF"])
def test_the_bundle_gate_reads_the_root_not_the_build_arches(
    consumer, rocke_fixture, enable_rocke
):
    """The product root carries a gfx950-only rocKE bundle and the build packs
    gfx90a alone, so the root is dormant, whether rocKE is enabled or not, and the
    arch-aware predicate says no.
    The arch-agnostic gate that decides which host sources compile still says yes
    when rocKE is enabled, and no when its kind is pruned. A probe that could not
    answer reads as the rocKE option."""
    shutil.copytree(rocke_fixture, consumer.source / "authored" / "rocKE" / "attention")

    def gates(force_probe_failure):
        forced = (
            "set_property(GLOBAL PROPERTY HKP_PACK_PROBE_OK_product FALSE)"
            if force_probe_failure
            else ""
        )
        return (
            f"set(HIPKERNELPROVIDER_ENABLE_ROCKE {enable_rocke})\n"
            + _wire_root("product", arches="gfx90a", enable_rocke=enable_rocke)
            + f"""{forced}
hkp_product_offers_engine(_offered "test_fixture:attention")
hkp_gfx950_attention_dense_available(_available)
get_property(_dormant GLOBAL PROPERTY HKP_PACK_DORMANT_LABELS)
{_report("dormant", "${_dormant}")}
foreach(_name IN ITEMS offered available)
    if(_${{_name}})
        {_report("${_name}", "TRUE")}
    else()
        {_report("${_name}", "FALSE")}
    endif()
endforeach()
"""
        )

    consumer._write(gates(False))
    output = consumer.configure()
    assert _reported(output, "offered") == ("TRUE" if enable_rocke == "ON" else "FALSE")
    assert _reported(output, "available") == "FALSE"
    # Dormant with rocKE enabled too: the gfx950 bundle is pruned for gfx90a, and a
    # build that names its product root this way gets a STATUS line, not a failure.
    assert _reported(output, "dormant").split(";") == ["product"]

    consumer._write(gates(True))
    output = consumer.configure()
    assert _reported(output, "offered") == ("TRUE" if enable_rocke == "ON" else "FALSE")


# A pattern over `pointwise`, as the packer's own tests write one.
_POINTWISE_PATTERN = {
    "nodes": [
        {
            "kind": "op",
            "id": "pointwise",
            "op": "pointwise",
            "operands": {"in_0": "$input_a", "in_1": "$input_b"},
            "results": {"out_0": "$output"},
        }
    ]
}
_POINTWISE_SCHEMA = {
    "operands": [
        {"name": "in_0", "optional": False},
        {"name": "in_1", "optional": False},
    ],
    "results": [{"name": "out_0", "optional": False}],
}


def _graph_match_root(empty_arch_fixture, destination, graph_match):
    """The empty-arch fixture, its UED matching by `graph_match`. It ships
    `test_fixture:solo` for gfx942."""
    shutil.copytree(empty_arch_fixture, destination, dirs_exist_ok=True)
    path = destination / "solo.ued.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["graph_match"] = graph_match
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_registry(path, ops):
    path.write_text(json.dumps({"ops": ops}), encoding="utf-8")


def _probe_authored_root():
    return f"""_hkp_root_probe(_shipped _offered _ok
    "${{CMAKE_CURRENT_SOURCE_DIR}}/authored" "gfx942" "" "")
{_report("ok", "${_ok}")}
{_report("shipped", "${_shipped}")}
"""


_NO_REGISTRY = """add_library(hipdnn_flatbuffers_sdk INTERFACE IMPORTED)
_hkp_resolve_op_schema_registry()
get_property(_reason GLOBAL PROPERTY HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE)
"""
_NO_REGISTRY_REASON = (
    "was built with HIPDNN_ENABLE_KERNEL_INGESTOR=OFF (or predates the op-schema "
    "registry). Rebuild it or set HIPKERNELPROVIDER_OP_SCHEMA_REGISTRY."
)


def test_a_hipdnn_without_a_registry_fails_configure_only_for_a_nodes_root(
    consumer, empty_arch_fixture
):
    """An installed hipDNN built without the kernel ingestor ships no registry. A
    root whose UED is `native` never reads one, so configure goes on; a root with
    a `nodes` pattern cannot be packed, and configure says why."""
    authored = consumer.source / "authored"
    _graph_match_root(
        empty_arch_fixture, authored, {"native": "hipkernel.pointwise.graph_match"}
    )
    consumer._write(
        _NO_REGISTRY + _report("reason", "${_reason}") + "\n" + _probe_authored_root()
    )
    output = consumer.configure()
    assert _NO_REGISTRY_REASON in _reported(output, "reason")
    assert _reported(output, "ok") == "TRUE"
    assert json.loads(_reported(output, "shipped"))["gfx942"] == ["test_fixture:solo"]

    _graph_match_root(empty_arch_fixture, authored, _POINTWISE_PATTERN)
    diagnostic = consumer.configure(success=False)
    assert "holds a graph_match `nodes` pattern" in diagnostic
    assert _NO_REGISTRY_REASON in diagnostic


@pytest.mark.parametrize("registry_available", [True, False])
def test_the_pack_step_is_given_the_registry_or_the_reason_there_is_none(
    consumer, registry_available
):
    """The pack runs with the build's registry, never one the caller's environment
    happens to carry, and without a registry it is told why."""
    consumer.env["HKP_OP_SCHEMA_REGISTRY"] = str(consumer.root / "ambient.json")
    registry = consumer.root / "op_schema_registry.json"
    _write_registry(registry, {})
    prelude = "add_library(hipdnn_flatbuffers_sdk INTERFACE IMPORTED)\n"
    if registry_available:
        prelude += (
            f'set(HIPDNN_FLATBUFFERS_SDK_OP_SCHEMA_REGISTRY "{registry.as_posix()}")\n'
        )
    consumer.wire(
        "ENABLE_ROCKE OFF", prelude=prelude + "_hkp_resolve_op_schema_registry()\n"
    )
    consumer.configure()
    consumer.build()

    env = consumer.invocation()["env"]
    if registry_available:
        assert Path(env["HKP_OP_SCHEMA_REGISTRY"]) == registry
        assert env["HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE"] is None
    else:
        assert env["HKP_OP_SCHEMA_REGISTRY"] is None
        assert _NO_REGISTRY_REASON in env["HKP_OP_SCHEMA_REGISTRY_UNAVAILABLE"]


def test_a_registry_without_the_patterns_opcode_fails_the_probe_derive_and_pack(
    consumer, empty_arch_fixture, rocm_kpack_dir
):
    """Probe, derive and pack all resolve a `nodes` pattern against the configured
    registry. The same root answers the probe and derives against a registry that
    holds `pointwise`, so the registry is what fails them."""
    authored = consumer.source / "authored"
    _graph_match_root(empty_arch_fixture, authored, _POINTWISE_PATTERN)
    complete = consumer.root / "complete.json"
    _write_registry(complete, {"pointwise": _POINTWISE_SCHEMA})
    lacking = consumer.root / "lacking.json"
    _write_registry(lacking, {})
    missing = "opcode 'pointwise' is not in the op-schema registry"
    derive = f"""include("{(PKG / "cmake" / "HkpPackagingProbes.cmake").as_posix()}")
_hkp_probe_derive_root(derived "${{CMAKE_CURRENT_SOURCE_DIR}}/authored" gfx942
    "${{CMAKE_CURRENT_BINARY_DIR}}/derived/root" _expect)
{_report("expect", "${_expect}")}
"""

    def registry(path):
        return f"-DHIPKERNELPROVIDER_OP_SCHEMA_REGISTRY={path}"

    consumer._write(_probe_authored_root())
    output = consumer.configure(registry(complete))
    assert _reported(output, "ok") == "TRUE"
    output = consumer.configure(registry(lacking))
    assert _reported(output, "ok") == "FALSE"
    assert missing in output

    consumer._write(derive)
    output = consumer.configure(registry(complete))
    assert _reported(output, "expect").endswith("expect.json")
    diagnostic = consumer.configure(registry(lacking), success=False)
    assert "deriving the probe root failed" in diagnostic
    assert missing in diagnostic

    # The pack tool as the pack step launches it, with the registry in its
    # environment. It runs under the test's own interpreter, which has rocm_kpack.
    command = [
        sys.executable,
        str(PKG / "tools" / "hkp_pack.py"),
        "--source-root",
        str(authored),
        "--out-root",
        str(consumer.root / "out"),
        "--arches",
        "gfx942",
        "--hipcc",
        "unused",
        "--generic-targets-json",
        str(GENERIC_TARGETS_JSON),
        "--inter-root",
        str(consumer.root / "inter"),
        "--source-label",
        "wiring",
    ]
    if rocm_kpack_dir:
        command += ["--kpack-python-dir", rocm_kpack_dir]
    proc = subprocess.run(
        command,
        env={**os.environ, "HKP_OP_SCHEMA_REGISTRY": str(lacking)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert missing in proc.stderr
