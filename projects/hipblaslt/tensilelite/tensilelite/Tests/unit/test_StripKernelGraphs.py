# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for removing rocRoller's codegen provenance.

A custom kernel is a hand-captured assembly file that hipBLASLt ships alongside
the kernels it generates. Each carries a YAML block inside its
``.amdgpu_metadata`` section: ``custom.config`` is the registration record that
drives kernel launch, and ``amdhsa.kernels`` is what the GPU runtime reads.
Kernels captured from rocRoller also arrive with rocRoller's internal view of
the kernel, in three forms that nothing reads: a graph dump under the
``.kernel_graph``, ``.kernel_graph_dot`` and ``.command`` keys, rocRoller's
launch metadata (``.workitem_count``, ``.dynamic_sharedmemory_bytes``,
``.kernel_dimensions``, ``.workgroup_size``, and ``.expression`` /
``.variableType`` on each argument), and whole-line ``//`` comments narrating
codegen.

``StripKernelGraphs`` deletes all three. These tests pin the properties that
make the deletion safe to apply to a kernel that ships: it must leave
instructions, their trailing comments, ``.amdgcn_target``, ``custom.config``
and the rest of ``amdhsa.kernels`` untouched; it must act on metadata keys only
inside the metadata section, at their own column, and on comments only outside
it; it must keep licence and annotation comments; and it must leave the
comments and launch metadata of a kernel that did not come from rocRoller
alone.
"""

import os

import pytest
import yaml

import tensilelite
from tensilelite.StripKernelGraphs import (COMMENT_LABEL, ROCROLLER_ARG_KEYS,
                                           ROCROLLER_KERNEL_KEYS, StripError, main,
                                           processFile, stripLaunchMetadata,
                                           stripLines, stripTraceComments)

pytestmark = pytest.mark.unit


# The three keys sit at four-space indentation, as siblings of the argument
# list, inside the single entry of amdhsa.kernels. Each is written here as the
# key line plus the lines it owns, so a fixture can mix and match them.
KERNEL_GRAPH = """\
    .kernel_graph:
      control:
        elements:
          - id: 1
            type: Node.Kernel
      coordinate:
        elements:
          - id: 1
            type: Dimension.User
"""

KERNEL_GRAPH_DOT = (
    '    .kernel_graph_dot: "digraph {\\n\\"coord1\\"[label=\\"User{NA}(1)\\"];\\n}"\n'
)

COMMAND = """\
    .command:
      sync: false
      operations:
        - type: Tensor
          tag: 0
"""

# Everything that must survive, split so a test can assert on each piece.
TARGET_LINE = '.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"\n'

INSTRUCTIONS = """\
.text
.globl k
.p2align 8
.type k,@function
k:
    s_load_b64 s[0:1], s[0:1], 0x0
    s_nop 0
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel k
  .amdhsa_group_segment_fixed_size 6400
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
.end_amdhsa_kernel
"""

CUSTOM_CONFIG = """\
custom.config:
  Source:
    Origin: rocroller
  Version: 1.0.0
  Features:
    SupportsBias: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: B6
  CustomKernel:
    args: [ { type: int64, semantic: StrideA0 } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
"""

KERNEL_HEAD = """\
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .symbol: k.kd
    .kernarg_segment_size: 128
    .sgpr_count: 56
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
"""

# A key that follows the dump at the same indentation. The strip must stop at
# it rather than swallowing it along with the block above.
TRAILING_SIBLING = "    .max_flat_workgroup_size: 64\n"

# Codegen-trace comments, copied in shape from the shipped kernels: register
# allocation, expression lowering, liveness, and the options banner. Note the
# varying indentation, including the tab rocRoller uses for nested records.
TRACE_COMMENTS = [
    "// (op 1122) Assign stride expression\n",
    " // Allocated : 1 VGPR (Value: UInt32) (op 1243): v0\n",
    " // Freeing : 1 SGPR (Value: Int32) (op 481): s14\n",
    "// Deleting tag 84\n",
    "//\tt.unitStride = 0\n",
    " // ------Dst: {LDS:(Offset: 0, Size: 8192), }\n",
    " //   maxVGPRs:                                 256\n",
    " // Kernel execution starts here. Use this label instead of the kernel"
    " name when setting a breakpoint.\n",
    "//   E_Mul 11, 9, 5\n",
]

# Comments that must survive: the licence header the repository requires, and
# the markers a human writes when leaving a note in a captured kernel.
LICENCE_COMMENTS = [
    "// Copyright Advanced Micro Devices, Inc., or its affiliates.\n",
    "// SPDX-License-Identifier: MIT\n",
    "// Portions under the Apache License.\n",
    "// copyright holder in lower case\n",
]

ANNOTATION_COMMENTS = [
    "// TODO: recapture once the ABI settles\n",
    "// FIXME: kernarg 3 is mis-sized\n",
    "// NOTE: the grid is driven by custom.config, not by this comment\n",
    "// XXX do not reorder these two stores\n",
    "// HACK: works around a scheduler bug\n",
    "// WARNING: assembles only with sramecc+\n",
]


# A kernel entry carrying rocRoller's launch metadata, in the shapes rocRoller
# writes: flow-style one-liners for most keys, a block-style expression for the
# second argument. .reqd_workgroup_size is a standard key that shares a suffix
# with rocRoller's .workgroup_size and must survive.
LAUNCH_KERNEL_HEAD = """\
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: k
    .symbol: k.kd
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 256
    .reqd_workgroup_size: [256, 1, 1]
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: CommandArgument, name: M}, rhs: 4}, 1, 1]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_0_stride_0}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: WGM
        .size: 4
        .offset: 8
        .expression:
          type: CommandArgument
          name: WGM
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
"""

LAUNCH_KERNEL_HEAD_STRIPPED = """\
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: k
    .symbol: k.kd
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 256
    .reqd_workgroup_size: [256, 1, 1]
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: WGM
        .size: 4
        .offset: 8
        .value_kind: by_value
"""

LAUNCH_REMOVED = {
    "workgroup_size": 1,
    "kernel_dimensions": 1,
    "workitem_count": 1,
    "dynamic_sharedmemory_bytes": 1,
    "args[].expression": 4,
    "args[].variableType": 2,
}


def buildSource(dump="", trailing="", preamble=None, configFirst=True,
                comments="", origin="rocroller", kernelHead=KERNEL_HEAD):
    """Assemble a synthetic custom kernel around an optional dump."""
    head = TARGET_LINE + INSTRUCTIONS if preamble is None else preamble
    config = CUSTOM_CONFIG.replace("Origin: rocroller", f"Origin: {origin}")
    body = config + kernelHead if configFirst else kernelHead + config
    return (
        comments
        + head
        + ".amdgpu_metadata\n---\n"
        + body
        + dump
        + trailing
        + "...\n.end_amdgpu_metadata\n"
    )


def writeKernel(tmp_path, source, name="k.s"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def metadataYaml(text):
    """Parse the embedded YAML the way CustomKernels._readEmbeddedYaml does."""
    lines, inYaml = [], False
    for line in text.splitlines():
        if line == "---":
            inYaml = True
            continue
        if line == "..." and inYaml:
            break
        if inYaml:
            lines.append(line)
    return yaml.safe_load("\n".join(lines))


# --------------------------------------------------------------------------
# Each key is removed, alone and together
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dump, key, ownedLines",
    [
        (KERNEL_GRAPH, "kernel_graph", 9),
        (KERNEL_GRAPH_DOT, "kernel_graph_dot", 1),
        (COMMAND, "command", 5),
    ],
)
def test_each_key_is_removed_on_its_own(tmp_path, dump, key, ownedLines):
    path = writeKernel(tmp_path, buildSource(dump=dump))
    removed, bytesRemoved = processFile(str(path))
    assert removed == {key: ownedLines}
    assert bytesRemoved == len(dump.encode("utf-8"))
    assert f".{key}:" not in path.read_text(encoding="utf-8")


def test_all_three_keys_are_removed_together(tmp_path):
    dump = KERNEL_GRAPH + KERNEL_GRAPH_DOT + COMMAND
    path = writeKernel(tmp_path, buildSource(dump=dump))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 9, "kernel_graph_dot": 1, "command": 5}
    assert path.read_text(encoding="utf-8") == buildSource()


# --------------------------------------------------------------------------
# Prefix collision: ".kernel_graph" is a strict prefix of ".kernel_graph_dot"
# --------------------------------------------------------------------------

def test_dot_key_alone_is_not_mistaken_for_the_graph_key(tmp_path):
    """Only .kernel_graph_dot present: it must go, and be reported as itself.

    A match that stops after the shorter name would report the removal under
    the wrong key, and a match anchored to the shorter name without requiring
    the colon would treat the rest of the line as the block's children.
    """
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH_DOT, trailing=TRAILING_SIBLING))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph_dot": 1}
    assert "kernel_graph" not in path.read_text(encoding="utf-8")
    assert TRAILING_SIBLING in path.read_text(encoding="utf-8")


def test_graph_key_alone_does_not_claim_a_dot_key_that_is_absent(tmp_path):
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH, trailing=TRAILING_SIBLING))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 9}
    assert TRAILING_SIBLING in path.read_text(encoding="utf-8")


def test_graph_key_immediately_followed_by_dot_key_removes_exactly_two_blocks(tmp_path):
    """The shorter key must not swallow the longer one, nor stop short of it."""
    path = writeKernel(
        tmp_path,
        buildSource(dump=KERNEL_GRAPH + KERNEL_GRAPH_DOT, trailing=TRAILING_SIBLING),
    )
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 9, "kernel_graph_dot": 1}
    assert path.read_text(encoding="utf-8") == buildSource(trailing=TRAILING_SIBLING)


def test_a_key_that_merely_starts_with_a_graph_name_is_kept(tmp_path):
    """.kernel_graph_version is not one of the three keys and must survive."""
    lookalike = "    .kernel_graph_version: 3\n"
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH, trailing=lookalike))
    processFile(str(path))
    assert lookalike in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Indentation scoping
# --------------------------------------------------------------------------

def test_children_go_and_the_next_sibling_stays(tmp_path):
    path = writeKernel(tmp_path, buildSource(dump=COMMAND, trailing=TRAILING_SIBLING))
    processFile(str(path))
    text = path.read_text(encoding="utf-8")
    assert "sync: false" not in text
    assert "operations:" not in text
    assert TRAILING_SIBLING in text


def test_a_shallower_key_after_the_dump_stays(tmp_path):
    """A key at lower indentation ends the block just as a sibling does."""
    shallower = "  - .symbol: second_kernel.kd\n"
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH, trailing=shallower))
    processFile(str(path))
    assert shallower in path.read_text(encoding="utf-8")


def test_a_nested_lookalike_inside_the_dump_is_consumed_once(tmp_path):
    """.command nested deeper inside .kernel_graph belongs to that block."""
    nested = (
        "    .kernel_graph:\n"
        "      control:\n"
        "        .command: inner\n"
        "        elements: []\n"
    )
    path = writeKernel(tmp_path, buildSource(dump=nested, trailing=TRAILING_SIBLING))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 4}
    assert path.read_text(encoding="utf-8") == buildSource(trailing=TRAILING_SIBLING)


# --------------------------------------------------------------------------
# What must survive byte for byte
# --------------------------------------------------------------------------

@pytest.mark.parametrize("configFirst", [True, False], ids=["config-before", "config-after"])
def test_custom_config_survives_byte_for_byte(tmp_path, configFirst):
    """custom.config is the registration record; it drives kernel launch."""
    dump = KERNEL_GRAPH + KERNEL_GRAPH_DOT + COMMAND
    path = writeKernel(tmp_path, buildSource(dump=dump, configFirst=configFirst))
    processFile(str(path))
    assert CUSTOM_CONFIG in path.read_text(encoding="utf-8")


def test_amdgcn_target_survives_byte_for_byte(tmp_path):
    """The target id fixes the processor the file assembles for."""
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH))
    processFile(str(path))
    text = path.read_text(encoding="utf-8")
    assert TARGET_LINE in text
    assert "gfx950:sramecc+" in text


def test_instructions_and_descriptor_survive_byte_for_byte(tmp_path):
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH + COMMAND))
    processFile(str(path))
    assert INSTRUCTIONS in path.read_text(encoding="utf-8")


def test_runtime_kernel_entry_survives_byte_for_byte(tmp_path):
    """amdhsa.kernels carries the argument layout the GPU runtime needs."""
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH_DOT))
    processFile(str(path))
    assert KERNEL_HEAD in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Only the metadata region is edited
# --------------------------------------------------------------------------

def test_a_lookalike_in_the_instruction_stream_is_left_alone(tmp_path):
    """A line reading like a graph key outside the metadata section stays."""
    preamble = TARGET_LINE + INSTRUCTIONS + "    .command: not_metadata\n"
    path = writeKernel(tmp_path, buildSource(dump=COMMAND, preamble=preamble))
    removed, _ = processFile(str(path))
    assert removed == {"command": 5}
    assert "    .command: not_metadata\n" in path.read_text(encoding="utf-8")


def test_a_lookalike_after_the_metadata_section_is_left_alone(tmp_path):
    trailer = "    .kernel_graph: after_the_end\n"
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH))
    path.write_text(path.read_text(encoding="utf-8") + trailer, encoding="utf-8")
    processFile(str(path))
    assert trailer in path.read_text(encoding="utf-8")


def test_a_file_with_no_metadata_section_is_untouched(tmp_path):
    source = TARGET_LINE + INSTRUCTIONS + "    .kernel_graph: stray\n"
    path = writeKernel(tmp_path, source)
    removed, bytesRemoved = processFile(str(path))
    assert removed == {}
    assert bytesRemoved == 0
    assert path.read_text(encoding="utf-8") == source


def test_an_unterminated_metadata_section_is_untouched(tmp_path):
    """Without .end_amdgpu_metadata the region has no end, so nothing is cut."""
    source = (
        TARGET_LINE + INSTRUCTIONS + ".amdgpu_metadata\n---\n" + CUSTOM_CONFIG + KERNEL_GRAPH
    )
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {}
    assert path.read_text(encoding="utf-8") == source


# --------------------------------------------------------------------------
# Idempotence, clean files, ordering, blank lines
# --------------------------------------------------------------------------

def test_running_twice_equals_running_once(tmp_path):
    dump = KERNEL_GRAPH + KERNEL_GRAPH_DOT + COMMAND
    path = writeKernel(tmp_path, buildSource(dump=dump))
    processFile(str(path))
    once = path.read_text(encoding="utf-8")
    removed, bytesRemoved = processFile(str(path))
    assert removed == {}
    assert bytesRemoved == 0
    assert path.read_text(encoding="utf-8") == once


def test_a_clean_file_is_not_rewritten(tmp_path):
    """No dump means no write at all, not merely an identical write."""
    source = buildSource()
    path = writeKernel(tmp_path, source)
    os.utime(str(path), ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    before = os.stat(str(path)).st_mtime_ns
    removed, bytesRemoved = processFile(str(path))
    assert removed == {}
    assert bytesRemoved == 0
    assert path.read_text(encoding="utf-8") == source
    assert os.stat(str(path)).st_mtime_ns == before


def test_keys_in_a_different_order_are_all_removed(tmp_path):
    dump = COMMAND + KERNEL_GRAPH_DOT + KERNEL_GRAPH
    path = writeKernel(tmp_path, buildSource(dump=dump))
    removed, _ = processFile(str(path))
    assert removed == {"command": 5, "kernel_graph_dot": 1, "kernel_graph": 9}
    assert path.read_text(encoding="utf-8") == buildSource()


def test_blank_lines_inside_a_dump_are_removed_with_it(tmp_path):
    dump = (
        "    .kernel_graph:\n"
        "      control:\n"
        "        elements: []\n"
        "\n"
        "      coordinate:\n"
        "        elements: []\n"
    )
    path = writeKernel(tmp_path, buildSource(dump=dump, trailing=TRAILING_SIBLING))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 6}
    assert path.read_text(encoding="utf-8") == buildSource(trailing=TRAILING_SIBLING)


def test_a_blank_line_before_the_next_sibling_is_kept(tmp_path):
    """Blank lines trailing a dump belong to what follows, not to the dump."""
    dump = "    .kernel_graph:\n      control: []\n\n"
    path = writeKernel(tmp_path, buildSource(dump=dump, trailing=TRAILING_SIBLING))
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 2}
    assert path.read_text(encoding="utf-8") == buildSource(trailing="\n" + TRAILING_SIBLING)


# --------------------------------------------------------------------------
# The gfx942 shape: dot and command present, kernel_graph absent
# --------------------------------------------------------------------------

def test_dot_and_command_without_the_graph_key(tmp_path):
    """One shipped kernel has this shape; an all-or-nothing strip would break."""
    path = writeKernel(
        tmp_path, buildSource(dump=KERNEL_GRAPH_DOT + COMMAND, trailing=TRAILING_SIBLING)
    )
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph_dot": 1, "command": 5}
    assert path.read_text(encoding="utf-8") == buildSource(trailing=TRAILING_SIBLING)


# --------------------------------------------------------------------------
# The result is still the YAML the loader expects
# --------------------------------------------------------------------------

def test_stripped_metadata_still_parses_and_keeps_both_live_keys(tmp_path):
    dump = KERNEL_GRAPH + KERNEL_GRAPH_DOT + COMMAND
    path = writeKernel(tmp_path, buildSource(dump=dump))

    before = metadataYaml(path.read_text(encoding="utf-8"))
    assert ".kernel_graph" in before["amdhsa.kernels"][0]

    processFile(str(path))
    after = metadataYaml(path.read_text(encoding="utf-8"))

    assert after["custom.config"] == before["custom.config"]
    assert after["amdhsa.kernels"][0][".args"] == before["amdhsa.kernels"][0][".args"]
    assert after["amdhsa.kernels"][0][".kernarg_segment_size"] == 128
    for key in (".kernel_graph", ".kernel_graph_dot", ".command"):
        assert key not in after["amdhsa.kernels"][0]


# --------------------------------------------------------------------------
# stripLines works on lines alone, without touching the filesystem
# --------------------------------------------------------------------------

def test_strip_lines_preserves_line_endings(tmp_path):
    source = buildSource(dump=KERNEL_GRAPH).replace("\n", "\r\n")
    newLines, removed = stripLines(source.splitlines(keepends=True))
    assert removed == {"kernel_graph": 9}
    assert all(line.endswith("\r\n") for line in newLines)


# --------------------------------------------------------------------------
# Codegen-trace comments go
# --------------------------------------------------------------------------

@pytest.mark.parametrize("comment", TRACE_COMMENTS)
def test_each_trace_comment_shape_is_removed(tmp_path, comment):
    path = writeKernel(tmp_path, buildSource(comments=comment))
    removed, bytesRemoved = processFile(str(path))
    assert removed == {COMMENT_LABEL: 1}
    assert bytesRemoved == len(comment.encode("utf-8"))
    assert comment not in path.read_text(encoding="utf-8")


def test_a_run_of_trace_comments_goes_in_one_pass(tmp_path):
    path = writeKernel(tmp_path, buildSource(comments="".join(TRACE_COMMENTS)))
    removed, _ = processFile(str(path))
    assert removed == {COMMENT_LABEL: len(TRACE_COMMENTS)}
    assert "//" not in path.read_text(encoding="utf-8")


def test_trace_comments_between_instructions_go(tmp_path):
    preamble = (
        TARGET_LINE
        + ".text\n"
        + "// (op 1) Assign VGPR stride\n"
        + "k:\n"
        + " // Allocated : 1 VGPR (Value: UInt32) (op 5): v0\n"
        + "    s_endpgm\n"
    )
    path = writeKernel(tmp_path, buildSource(preamble=preamble))
    removed, _ = processFile(str(path))
    assert removed == {COMMENT_LABEL: 2}
    text = path.read_text(encoding="utf-8")
    assert TARGET_LINE in text and "k:\n" in text and "    s_endpgm\n" in text


def test_graphs_and_comments_are_reported_separately(tmp_path):
    source = buildSource(dump=KERNEL_GRAPH + COMMAND,
                         comments="".join(TRACE_COMMENTS))
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {
        "kernel_graph": 9, "command": 5, COMMENT_LABEL: len(TRACE_COMMENTS),
    }


# --------------------------------------------------------------------------
# Load-bearing comments stay
# --------------------------------------------------------------------------

@pytest.mark.parametrize("comment", LICENCE_COMMENTS + ANNOTATION_COMMENTS)
def test_a_marked_comment_is_kept(tmp_path, comment):
    source = buildSource(comments=comment + "".join(TRACE_COMMENTS))
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {COMMENT_LABEL: len(TRACE_COMMENTS)}
    assert comment in path.read_text(encoding="utf-8")


def test_the_licence_header_survives_byte_for_byte(tmp_path):
    header = "".join(LICENCE_COMMENTS[:2]) + "\n"
    source = buildSource(comments=header + "".join(TRACE_COMMENTS))
    path = writeKernel(tmp_path, source)
    processFile(str(path))
    assert path.read_text(encoding="utf-8").startswith(header)


def test_an_instruction_keeps_its_trailing_comment(tmp_path):
    preamble = TARGET_LINE + ".text\nk:\n    s_nop 0 // (op 1243) reg expression\n"
    path = writeKernel(tmp_path, buildSource(preamble=preamble,
                                             comments=TRACE_COMMENTS[0]))
    removed, _ = processFile(str(path))
    assert removed == {COMMENT_LABEL: 1}
    assert "    s_nop 0 // (op 1243) reg expression\n" in path.read_text(
        encoding="utf-8")


def test_a_comment_inside_the_metadata_section_is_kept(tmp_path):
    # "//" is not a YAML comment, so a line like this inside the metadata
    # block would be data. The strip must not reach into the section.
    source = buildSource(trailing="    .note: // not a comment here\n",
                         comments=TRACE_COMMENTS[0])
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {COMMENT_LABEL: 1}
    assert "    .note: // not a comment here\n" in path.read_text(
        encoding="utf-8")


def test_blank_lines_and_directives_around_a_comment_are_kept(tmp_path):
    preamble = TARGET_LINE + "\n" + TRACE_COMMENTS[0] + "\n" + INSTRUCTIONS
    path = writeKernel(tmp_path, buildSource(preamble=preamble))
    processFile(str(path))
    assert TARGET_LINE + "\n\n" + INSTRUCTIONS in path.read_text(
        encoding="utf-8")


@pytest.mark.parametrize("configFirst", [True, False])
def test_custom_config_survives_the_comment_strip(tmp_path, configFirst):
    source = buildSource(comments="".join(TRACE_COMMENTS),
                         configFirst=configFirst)
    path = writeKernel(tmp_path, source)
    processFile(str(path))
    assert CUSTOM_CONFIG in path.read_text(encoding="utf-8")


def test_the_registration_record_still_parses_after_both_strips(tmp_path):
    source = buildSource(dump=KERNEL_GRAPH + KERNEL_GRAPH_DOT + COMMAND,
                         comments="".join(TRACE_COMMENTS))
    path = writeKernel(tmp_path, source)
    processFile(str(path))
    parsed = metadataYaml(path.read_text(encoding="utf-8"))
    assert set(parsed) == {"custom.config", "amdhsa.version", "amdhsa.kernels"}
    assert parsed["custom.config"] == metadataYaml(source)["custom.config"]


# --------------------------------------------------------------------------
# Only rocRoller's kernels are touched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("origin", ["tensile", "ck", "triton", "wave"])
def test_a_kernel_from_another_source_keeps_its_comments(tmp_path, origin):
    source = buildSource(comments="".join(TRACE_COMMENTS), origin=origin)
    path = writeKernel(tmp_path, source)
    removed, bytesRemoved = processFile(str(path))
    assert removed == {}
    assert bytesRemoved == 0
    assert path.read_text(encoding="utf-8") == source


def test_a_kernel_with_no_metadata_keeps_its_comments(tmp_path):
    source = TARGET_LINE + "".join(TRACE_COMMENTS) + INSTRUCTIONS
    path = writeKernel(tmp_path, source)
    assert processFile(str(path)) == ({}, 0)
    assert path.read_text(encoding="utf-8") == source


def test_a_graph_dump_is_still_removed_from_a_non_rocroller_kernel(tmp_path):
    # The origin gate guards comments only: a dump is unread whoever wrote it.
    source = buildSource(dump=KERNEL_GRAPH, comments=TRACE_COMMENTS[0],
                         origin="tensile")
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 9}
    assert TRACE_COMMENTS[0] in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# rocRoller's launch metadata goes from rocRoller's kernels only
# --------------------------------------------------------------------------

@pytest.mark.parametrize("configFirst", [True, False], ids=["config-before", "config-after"])
def test_launch_metadata_is_removed_from_a_rocroller_kernel(tmp_path, configFirst):
    source = buildSource(kernelHead=LAUNCH_KERNEL_HEAD, configFirst=configFirst)
    path = writeKernel(tmp_path, source)
    removed, bytesRemoved = processFile(str(path))
    assert removed == LAUNCH_REMOVED
    expected = buildSource(kernelHead=LAUNCH_KERNEL_HEAD_STRIPPED, configFirst=configFirst)
    assert path.read_text(encoding="utf-8") == expected
    assert bytesRemoved == len(source.encode("utf-8")) - len(expected.encode("utf-8"))


@pytest.mark.parametrize("origin", ["tensile", "ck", "triton", "wave"])
def test_launch_metadata_of_another_source_is_kept(tmp_path, origin):
    source = buildSource(kernelHead=LAUNCH_KERNEL_HEAD, origin=origin)
    path = writeKernel(tmp_path, source)
    assert processFile(str(path)) == ({}, 0)
    assert path.read_text(encoding="utf-8") == source


def test_launch_metadata_and_graph_dump_go_together(tmp_path):
    source = buildSource(kernelHead=LAUNCH_KERNEL_HEAD, dump=KERNEL_GRAPH + COMMAND,
                         comments="".join(TRACE_COMMENTS))
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == dict(LAUNCH_REMOVED, kernel_graph=9, command=5,
                           **{COMMENT_LABEL: len(TRACE_COMMENTS)})
    assert path.read_text(encoding="utf-8") == buildSource(
        kernelHead=LAUNCH_KERNEL_HEAD_STRIPPED)


def test_stripped_launch_metadata_still_parses_with_the_runtime_keys(tmp_path):
    path = writeKernel(tmp_path, buildSource(kernelHead=LAUNCH_KERNEL_HEAD))
    before = metadataYaml(path.read_text(encoding="utf-8"))["amdhsa.kernels"][0]
    processFile(str(path))
    after = metadataYaml(path.read_text(encoding="utf-8"))["amdhsa.kernels"][0]

    kernelKeys = {"." + k for k in ROCROLLER_KERNEL_KEYS}
    argKeys = {"." + k for k in ROCROLLER_ARG_KEYS}
    assert kernelKeys <= set(before)
    assert {k: v for k, v in after.items() if k != ".args"} == {
        k: v for k, v in before.items() if k not in kernelKeys and k != ".args"}
    assert after[".args"] == [{k: v for k, v in arg.items() if k not in argKeys}
                              for arg in before[".args"]]


def test_an_arg_key_outside_the_args_list_is_kept(tmp_path):
    """.expression at the kernel-key column is not an argument's expression."""
    head = KERNEL_HEAD.replace("    .args:\n", "    .expression: kept\n    .args:\n")
    path = writeKernel(tmp_path, buildSource(kernelHead=head))
    assert processFile(str(path)) == ({}, 0)


def test_a_kernel_key_inside_an_argument_is_kept(tmp_path):
    """.workgroup_size at the argument-key column belongs to the argument."""
    head = KERNEL_HEAD + "        .workgroup_size: kept\n"
    path = writeKernel(tmp_path, buildSource(kernelHead=head))
    assert processFile(str(path)) == ({}, 0)


def test_launch_metadata_strip_is_idempotent(tmp_path):
    source = buildSource(kernelHead=LAUNCH_KERNEL_HEAD)
    once, removed = stripLines(source.splitlines(keepends=True))
    assert removed == LAUNCH_REMOVED
    twice, removedAgain = stripLines(list(once))
    assert twice == once
    assert removedAgain == {}


def test_launch_metadata_strip_preserves_line_endings():
    source = buildSource(kernelHead=LAUNCH_KERNEL_HEAD).replace("\n", "\r\n")
    newLines, removed = stripLaunchMetadata(source.splitlines(keepends=True))
    assert removed == LAUNCH_REMOVED
    assert "".join(newLines) == buildSource(
        kernelHead=LAUNCH_KERNEL_HEAD_STRIPPED).replace("\n", "\r\n")


def _withoutCustomKernel(source):
    """The same kernel with its custom.config CustomKernel block removed."""
    block = ("  CustomKernel:\n"
             "    args: [ { type: int64, semantic: StrideA0 } ]\n"
             "    macrotile: [32, 32, 64]\n"
             "    threads: [64, 1, 1]\n"
             "    grid: [TilesX, TilesY, Batch]\n")
    assert source.count(block) == 1
    return source.replace(block, "")


@pytest.mark.parametrize("configFirst", [True, False], ids=["config-before", "config-after"])
def test_launch_metadata_of_a_kernel_without_custom_kernel_is_refused(tmp_path, capsys,
                                                                     configFirst):
    """Such a kernel's arguments are rebuilt from .expression and .variableType."""
    source = _withoutCustomKernel(
        buildSource(kernelHead=LAUNCH_KERNEL_HEAD, configFirst=configFirst))
    path = writeKernel(tmp_path, source)
    with pytest.raises(StripError, match="CustomKernel"):
        processFile(str(path))
    assert main(["--check", str(tmp_path)]) == 2
    assert main([str(tmp_path)]) == 2
    assert path.read_text(encoding="utf-8") == source
    assert "CustomKernel" in capsys.readouterr().err


def test_a_kernel_without_custom_kernel_and_no_launch_metadata_is_not_refused(tmp_path):
    source = _withoutCustomKernel(buildSource(dump=KERNEL_GRAPH))
    path = writeKernel(tmp_path, source)
    removed, _ = processFile(str(path))
    assert removed == {"kernel_graph": 9}


def test_a_nested_custom_kernel_key_does_not_count(tmp_path):
    source = _withoutCustomKernel(buildSource(kernelHead=LAUNCH_KERNEL_HEAD)).replace(
        "  Source:\n", "  Source:\n    CustomKernel: not-the-block\n")
    path = writeKernel(tmp_path, source)
    with pytest.raises(StripError, match="CustomKernel"):
        processFile(str(path))


def test_a_key_that_opens_its_entry_is_refused(tmp_path, capsys):
    head = KERNEL_HEAD.replace("      - .name: Tensor_0_stride_0_0\n",
                               "      - .expression: {type: CommandArgument}\n"
                               "        .name: Tensor_0_stride_0_0\n")
    source = buildSource(kernelHead=head)
    path = writeKernel(tmp_path, source)
    with pytest.raises(StripError):
        processFile(str(path))
    assert main([str(tmp_path)]) == 2
    assert path.read_text(encoding="utf-8") == source
    assert "opens its entry" in capsys.readouterr().err


@pytest.mark.parametrize("label", sorted(LAUNCH_REMOVED))
def test_check_mode_reports_each_launch_key_by_name(tmp_path, capsys, label):
    writeKernel(tmp_path, buildSource(kernelHead=LAUNCH_KERNEL_HEAD))
    assert main(["--check", str(tmp_path)]) == 1
    assert f"  .{label}: {LAUNCH_REMOVED[label]} lines" in capsys.readouterr().out


# --------------------------------------------------------------------------
# stripTraceComments works on lines alone
# --------------------------------------------------------------------------

def test_strip_trace_comments_is_idempotent(tmp_path):
    source = buildSource(dump=KERNEL_GRAPH, comments="".join(TRACE_COMMENTS))
    once, removed = stripLines(source.splitlines(keepends=True))
    assert removed
    twice, removedAgain = stripLines(list(once))
    assert twice == once
    assert removedAgain == {}


def test_strip_trace_comments_preserves_line_endings(tmp_path):
    source = buildSource(comments="".join(TRACE_COMMENTS)).replace("\n", "\r\n")
    newLines, removed = stripTraceComments(source.splitlines(keepends=True))
    assert removed == len(TRACE_COMMENTS)
    assert all(line.endswith("\r\n") for line in newLines)


# --------------------------------------------------------------------------
# Command-line behaviour
# --------------------------------------------------------------------------

def test_check_mode_reports_without_writing_and_exits_one(tmp_path, capsys):
    source = buildSource(dump=KERNEL_GRAPH + COMMAND)
    path = writeKernel(tmp_path, source)
    os.utime(str(path), ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    before = os.stat(str(path)).st_mtime_ns

    assert main(["--check", str(tmp_path)]) == 1

    assert path.read_text(encoding="utf-8") == source
    assert os.stat(str(path)).st_mtime_ns == before
    out = capsys.readouterr().out
    assert "would strip" in out
    assert "1 of 1 file(s)" in out


def test_check_mode_on_a_clean_tree_exits_zero(tmp_path, capsys):
    writeKernel(tmp_path, buildSource())
    assert main(["--check", str(tmp_path)]) == 0
    assert "0 of 1 file(s)" in capsys.readouterr().out


def test_strip_mode_writes_and_exits_zero(tmp_path):
    path = writeKernel(tmp_path, buildSource(dump=KERNEL_GRAPH))
    assert main([str(tmp_path)]) == 0
    assert ".kernel_graph:" not in path.read_text(encoding="utf-8")


def test_check_mode_reports_trace_comments_by_name(tmp_path, capsys):
    writeKernel(tmp_path, buildSource(comments="".join(TRACE_COMMENTS)))
    assert main(["--check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{COMMENT_LABEL} ({len(TRACE_COMMENTS)} lines)" in out
    assert f"  {COMMENT_LABEL}: {len(TRACE_COMMENTS)} lines" in out


def test_a_directory_argument_finds_nested_kernels(tmp_path):
    nested = tmp_path / "rocroller"
    nested.mkdir()
    writeKernel(nested, buildSource(dump=KERNEL_GRAPH))
    writeKernel(tmp_path, buildSource(dump=COMMAND), name="top.s")
    (tmp_path / "notes.txt").write_text(".kernel_graph: not assembly\n", encoding="utf-8")

    assert main(["--check", str(tmp_path)]) == 1
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == (
        ".kernel_graph: not assembly\n"
    )


# --------------------------------------------------------------------------
# Against the kernels that actually ship
# --------------------------------------------------------------------------

def customKernelRoot():
    root = os.path.join(os.path.dirname(tensilelite.__file__), "CustomKernels")
    assert os.path.isdir(root)
    return root


def test_no_shipped_custom_kernel_carries_provenance(capsys):
    """Regression gate: a kernel added later must not bring provenance back.

    That covers the graph dump, the launch metadata and the trace comments;
    the next test pins the launch keys so the gate cannot quietly lose one.
    """
    assert main(["--check", customKernelRoot()]) == 0
    assert "0 of" in capsys.readouterr().out


def test_the_gate_covers_every_launch_metadata_key():
    assert set(ROCROLLER_KERNEL_KEYS) | {"args[]." + k for k in ROCROLLER_ARG_KEYS} == set(
        LAUNCH_REMOVED)


def test_shipped_rocroller_kernels_keep_only_their_licence_header():
    root = os.path.join(customKernelRoot(), "rocroller")
    for name in sorted(f for f in os.listdir(root) if f.endswith(".s")):
        with open(os.path.join(root, name), encoding="utf-8") as handle:
            whole = [l for l in handle if l.lstrip().startswith("//")]
        assert all("Copyright" in l or "SPDX" in l for l in whole), name


def test_the_other_shipped_custom_kernels_keep_their_comments():
    """The origin gate must leave hand-written kernels documented."""
    root = customKernelRoot()
    kept = 0
    for base, _, names in os.walk(root):
        if os.path.basename(base) == "rocroller":
            continue
        for name in names:
            if not name.endswith(".s"):
                continue
            with open(os.path.join(base, name), encoding="utf-8") as handle:
                kept += sum(1 for l in handle if l.lstrip().startswith("//"))
    assert kept > 100


def test_the_shipped_rocroller_kernels_still_expose_their_registration_record():
    """Stripping must leave every rocRoller kernel loadable by name."""
    from tensilelite.CustomKernels import readCustomKernelConfig

    root = os.path.join(os.path.dirname(tensilelite.__file__), "CustomKernels", "rocroller")
    names = sorted(f[:-2] for f in os.listdir(root) if f.endswith(".s"))
    assert len(names) >= 90

    for name in names:
        config = readCustomKernelConfig(name, directory=root)
        assert "ProblemType" in config, name
        assert set(config["CustomKernel"]) >= {"args", "macrotile", "threads", "grid"}, name
