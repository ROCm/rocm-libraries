"""Kernel argument signatures read out of a compiled code object.

Two levels. The `quick` tests drive the parser against synthesised objects, so
every branch -- bundle unwrapping, hidden-argument exclusion, absent names, the
error paths -- is reachable without a toolchain and without depending on what a
particular compiler happened to emit. The hipcc-gated test is the oracle that
matters: it compiles a kernel, changes its signature, recompiles, and asserts
the extracted signature followed. That is what proves the field is derived from
the object rather than from anything an author wrote.
"""

import pytest

from conftest import _arg, _bundle, _elf, _kernel, _note, _object
from hkp_pack.errors import HkpPackError
from hkp_pack.hip_compile import compile_hip_variant
from hkp_pack.kernel_signature import amdgcn_object, kernel_signature

try:
    import msgpack
except ImportError:
    msgpack = None

ARCH = "gfx942"
WHERE = "test object"

pytestmark = pytest.mark.skipif(
    msgpack is None, reason="msgpack is not importable in this environment"
)


THREE_BUFFERS = [
    _arg("global_buffer", 8, 0),
    _arg("global_buffer", 8, 8),
    _arg("global_buffer", 8, 16),
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_reads_arguments_in_declaration_order():
    code = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])

    assert kernel_signature(code, "PointwiseAdd", WHERE) == [
        {"kind": "global_buffer", "size": 8, "offset": 0},
        {"kind": "global_buffer", "size": 8, "offset": 8},
        {"kind": "global_buffer", "size": 8, "offset": 16},
    ]


@pytest.mark.quick
def test_hidden_arguments_are_excluded():
    """The compiler appends these; the host never marshals them.

    Keeping them would make the recorded signature structurally incomparable
    with the pack's argument list -- three marshalled pointers would have to
    match a six-entry signature.
    """
    code = _object(
        [
            _kernel(
                "PointwiseAdd",
                THREE_BUFFERS
                + [
                    _arg("hidden_block_count_x", 4, 24),
                    _arg("hidden_global_offset_x", 8, 32),
                ],
            )
        ]
    )

    assert [a["kind"] for a in kernel_signature(code, "PointwiseAdd", WHERE)] == [
        "global_buffer",
        "global_buffer",
        "global_buffer",
    ]


@pytest.mark.quick
def test_names_are_recorded_when_the_producer_emits_them():
    code = _object(
        [
            _kernel(
                "fmha_bwd",
                [
                    _arg("global_buffer", 8, 0, name="dQ"),
                    _arg("by_value", 8, 8, name="pad"),
                ],
            )
        ]
    )

    assert kernel_signature(code, "fmha_bwd", WHERE) == [
        {"kind": "global_buffer", "size": 8, "offset": 0, "name": "dQ"},
        {"kind": "by_value", "size": 8, "offset": 8, "name": "pad"},
    ]


@pytest.mark.quick
def test_names_are_omitted_rather_than_emptied_when_absent():
    """A hip kernel's arguments carry no names, and the field must say so.

    Recording "" would make an unnamed argument indistinguishable from one the
    producer named the empty string, and would let a comparison believe it had
    checked a permutation it cannot see.
    """
    code = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])

    assert all("name" not in a for a in kernel_signature(code, "PointwiseAdd", WHERE))


@pytest.mark.quick
def test_a_kernel_taking_no_arguments_yields_an_empty_signature():
    """Distinct from a missing signature, which raises."""
    code = _object([_kernel("Barrier", [])])

    assert kernel_signature(code, "Barrier", WHERE) == []


@pytest.mark.quick
def test_the_named_kernel_is_selected_from_several():
    code = _object(
        [
            _kernel("PointwiseAdd", THREE_BUFFERS),
            _kernel("PointwiseAddSecondSymbol", THREE_BUFFERS[:2]),
        ]
    )

    assert len(kernel_signature(code, "PointwiseAddSecondSymbol", WHERE)) == 2


# ---------------------------------------------------------------------------
# Bundle unwrapping
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_reads_through_a_clang_offload_bundle():
    """What hipcc --genco actually writes, and what the packer stores."""
    inner = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])
    code = _bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx1152", inner),
        ]
    )

    assert len(kernel_signature(code, "PointwiseAdd", WHERE)) == 3


@pytest.mark.quick
def test_a_bare_elf_is_passed_through():
    """What the ASM producers store. Both shapes ship in this tree."""
    inner = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])

    assert amdgcn_object(inner, WHERE) == inner


@pytest.mark.quick
def test_the_host_entry_is_never_mistaken_for_the_device_entry():
    """The zero-length host entry can share the device entry's offset.

    Selecting by position or by taking the first entry returns host bytes with
    a plausible-looking offset, so selection is on the triple.
    """
    inner = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])
    code = _bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx942", inner),
        ]
    )

    assert amdgcn_object(code, WHERE) == inner


@pytest.mark.quick
def test_a_bundle_without_an_amdgcn_entry_is_rejected():
    with pytest.raises(HkpPackError, match="no amdgcn entry"):
        amdgcn_object(_bundle([("host-x86_64-unknown-linux-gnu-", b"junk")]), WHERE)


@pytest.mark.quick
def test_a_multi_arch_bundle_is_rejected():
    """One arch per object is this packer's invariant; guessing would silently
    record the wrong arch's signature."""
    inner = _object([_kernel("PointwiseAdd", THREE_BUFFERS)])
    code = _bundle(
        [
            ("hipv4-amdgcn-amd-amdhsa--gfx942", inner),
            ("hipv4-amdgcn-amd-amdhsa--gfx950", inner),
        ]
    )

    with pytest.raises(HkpPackError, match="2 amdgcn entries"):
        amdgcn_object(code, WHERE)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_an_unknown_symbol_names_what_the_object_holds():
    code = _object(
        [
            _kernel("PointwiseAdd", THREE_BUFFERS),
            _kernel("PointwiseAddSecondSymbol", THREE_BUFFERS),
        ]
    )

    with pytest.raises(HkpPackError) as excinfo:
        kernel_signature(code, "PointwiseSubtract", WHERE)

    message = str(excinfo.value)
    assert "PointwiseSubtract" in message
    assert "PointwiseAddSecondSymbol" in message


@pytest.mark.quick
def test_an_object_without_the_metadata_note_is_rejected():
    """Absence fails the pack run rather than emitting a descriptor with no
    signature: the runtime check is total only if the field is always there."""
    code = _elf(_note("GNU", 1, b"\x00\x00\x00\x00"))

    with pytest.raises(HkpPackError, match="no AMDGPU metadata note"):
        kernel_signature(code, "PointwiseAdd", WHERE)


@pytest.mark.quick
def test_something_that_is_neither_elf_nor_bundle_is_rejected():
    with pytest.raises(HkpPackError, match="neither an ELF nor a clang offload bundle"):
        kernel_signature(b"not a code object at all", "PointwiseAdd", WHERE)


# ---------------------------------------------------------------------------
# Oracle: the signature follows the compiled object
# ---------------------------------------------------------------------------


_KERNEL_TEMPLATE = """
extern "C" __global__ void SignatureProbe(float* a, float* b, float* out{extra})
{{
    out[0] = a[0] + b[0];
}}
"""


def _compile_probe(hipcc, tmp_path, extra_parameters, tag):
    source_root = tmp_path / tag
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "probe.hip").write_text(
        _KERNEL_TEMPLATE.format(extra=extra_parameters), encoding="utf-8"
    )
    out_dir = tmp_path / f"{tag}-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    co_path = compile_hip_variant(
        hipcc, source_root, "", "probe.hip", {}, ARCH, out_dir
    )
    return co_path.read_bytes()


def test_signature_follows_the_kernel_it_was_compiled_from(hipcc, tmp_path):
    """The oracle for this stream.

    Compile, change the signature, recompile, and assert the extracted
    signature changed with it. A field derived from anything an author wrote
    would be identical across both compiles.
    """
    baseline = kernel_signature(
        _compile_probe(hipcc, tmp_path, "", "baseline"), "SignatureProbe", WHERE
    )
    widened = kernel_signature(
        _compile_probe(hipcc, tmp_path, ", int count", "widened"),
        "SignatureProbe",
        WHERE,
    )

    assert [a["kind"] for a in baseline] == ["global_buffer"] * 3
    assert [a["kind"] for a in widened] == ["global_buffer"] * 3 + ["by_value"]
    assert widened[3]["size"] == 4
    assert baseline != widened


def test_hip_kernels_carry_no_argument_names(hipcc, tmp_path):
    """Pins the producer-dependence the recorded shape is designed around.

    If a future toolchain starts emitting names on the hip path, this fails and
    the omit-when-absent branch stops being the hip path's normal case.
    """
    signature = kernel_signature(
        _compile_probe(hipcc, tmp_path, "", "names"), "SignatureProbe", WHERE
    )

    assert all("name" not in argument for argument in signature)
