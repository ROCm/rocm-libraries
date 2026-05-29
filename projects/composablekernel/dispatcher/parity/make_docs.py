#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate the two parity PDFs with reportlab.

  parity_design.pdf  -- every file in this directory and how it serves parity
  parity_usage.pdf   -- basic, copy-pasteable usage

Run: python make_docs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
_ss = getSampleStyleSheet()
AMD_RED = colors.HexColor("#ED1C24")
INK = colors.HexColor("#1A1A1A")
MUTE = colors.HexColor("#5A5A5A")
CODE_BG = colors.HexColor("#F4F4F6")
CODE_BORDER = colors.HexColor("#D8D8DE")
RULE = colors.HexColor("#C9C9CF")

H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=20, textColor=INK, spaceBefore=6, spaceAfter=10,
                    leading=24)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=14, textColor=AMD_RED, spaceBefore=16, spaceAfter=6,
                    leading=18)
H3 = ParagraphStyle("H3", parent=_ss["Heading3"], fontName="Helvetica-Bold",
                    fontSize=11.5, textColor=INK, spaceBefore=10, spaceAfter=4,
                    leading=15)
BODY = ParagraphStyle("Body", parent=_ss["BodyText"], fontName="Helvetica",
                      fontSize=10, textColor=INK, leading=15, alignment=TA_LEFT,
                      spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, textColor=MUTE,
                       leading=11)
CODE = ParagraphStyle("Code", parent=_ss["Code"], fontName="Courier",
                      fontSize=8.3, textColor=INK, leading=11, backColor=CODE_BG,
                      borderColor=CODE_BORDER, borderWidth=0.6, borderPadding=6,
                      spaceBefore=4, spaceAfter=8, leftIndent=2, rightIndent=2)
SUBTITLE = ParagraphStyle("Subtitle", parent=BODY, fontSize=11, textColor=MUTE,
                          leading=15, spaceAfter=2)
BULLET = ParagraphStyle("Bullet", parent=BODY, spaceAfter=3, leading=14)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def code(text: str):
    body = "<br/>".join(esc(ln) if ln else "&nbsp;" for ln in text.split("\n"))
    return Paragraph(f'<font face="Courier">{body}</font>', CODE)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=10, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=AMD_RED,
    )


def table(rows, col_widths, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFB")]),
    ]
    if header:
        style += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), AMD_RED),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, AMD_RED),
        ]
    t.setStyle(TableStyle(style))
    return t


def cell(text, style=None):
    return Paragraph(text, style or ParagraphStyle("c", parent=BODY, fontSize=8.6,
                                                   leading=11, spaceAfter=0))


def _decoration(canvas, doc, title):
    canvas.saveState()
    canvas.setFillColor(AMD_RED)
    canvas.rect(0, LETTER[1] - 0.32 * inch, LETTER[0], 0.32 * inch, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(0.75 * inch, LETTER[1] - 0.22 * inch, "AMD  ·  CK Tile Dispatcher")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, LETTER[1] - 0.22 * inch, title)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 0.6 * inch, LETTER[0] - 0.75 * inch, 0.6 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTE)
    canvas.drawString(0.75 * inch, 0.42 * inch,
                      "dispatcher/parity  ·  Tile Engine ↔ Dispatcher parity")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build(path: Path, title: str, story):
    doc = SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.62 * inch, bottomMargin=0.78 * inch,
        title=title, author="AMD MLSE",
    )
    deco = lambda c, d: _decoration(c, d, title)
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# PDF 1: design / code explanation
# --------------------------------------------------------------------------- #
def design_story():
    s = []
    s.append(Paragraph("Tile Engine &harr; Dispatcher Parity", H1))
    s.append(Paragraph("Design &amp; code walkthrough: what each file is and how it "
                       "proves parity", SUBTITLE))
    s.append(HRFlowable(width="100%", thickness=1, color=AMD_RED, spaceAfter=10))

    s.append(Paragraph("The goal", H2))
    s.append(Paragraph(
        "The <b>dispatcher</b> selects and launches CK&nbsp;Tile GEMM kernels at "
        "runtime. <b>Tile Engine</b> is the existing offline codegen + benchmark "
        "system. Parity means: feed the same config to both and get the same "
        "kernel, the same registry key (computed offline during codegen "
        "<i>and</i> at runtime), the same numerical result, and &mdash; within a "
        "tolerance &mdash; the same performance.", BODY))
    s.append(Paragraph(
        "The work is split so that everything provable <b>without a GPU</b> is "
        "proven on a CPU box (translation + the registry-key guarantee), while "
        "the GPU-only half (build, run, verify, benchmark) is staged and runs "
        "unchanged on a GPU node.", BODY))

    s.append(Paragraph("Pipeline at a glance", H2))
    s.append(code(
        "Tile Engine config JSON\n"
        "      |\n"
        "      v\n"
        " te_to_dispatcher.py  ----------------->  dispatcher config objects   (a)\n"
        "      |                                        |\n"
        "      |               identifier.py (Python encode_identifier)\n"
        "      |               cpp_identifier_oracle (C++ KernelKey::encode_identifier)\n"
        "      |                                        |\n"
        "      |               check_identifier_parity.py   (b)   g++ only, NO GPU\n"
        "      |\n"
        "      |-- drive_codegen.py -> unified_gemm_codegen.py -> gemm_<id>.hpp   (c)\n"
        "      |                                                       |\n"
        "      |               harness.cpp + build_harness.sh   (d)   hipcc; run=GPU\n"
        "      |\n"
        "      '-- check_parity.py  ------------------------------------------->  (e)(f)\n"
        "            stage 1 identifier   (always, CPU)\n"
        "            stage 2 numerical    (GPU-gated)\n"
        "            stage 3 performance  (GPU-gated)"))

    s.append(Paragraph("The six deliverables", H2))
    rows = [
        [cell("<b>#</b>"), cell("<b>Deliverable</b>"), cell("<b>Files</b>"),
         cell("<b>Needs GPU?</b>")],
        [cell("a"), cell("Translator: TE JSON &rarr; dispatcher config objects"),
         cell("te_to_dispatcher.py"), cell("no")],
        [cell("b"), cell("Kernel identifier matches codegen &harr; runtime"),
         cell("identifier.py, cpp_identifier_oracle.cpp, check_identifier_parity.py"),
         cell("no (g++)")],
        [cell("c"), cell("Drive codegen for a single config"),
         cell("drive_codegen.py"), cell("no")],
        [cell("d"), cell("Minimal C++ harness to run one kernel"),
         cell("harness.cpp, build_harness.sh"), cell("build no / run yes")],
        [cell("e"), cell("Parity checker: dispatcher vs Tile Engine"),
         cell("check_parity.py"), cell("gated")],
        [cell("f"), cell("Numerical parity first, then performance"),
         cell("check_parity.py (3-stage)"), cell("yes")],
    ]
    s.append(table(rows, [0.3 * inch, 2.3 * inch, 2.9 * inch, 1.0 * inch]))

    # ---- file-by-file ----
    s.append(PageBreak())
    s.append(Paragraph("File by file", H2))

    s.append(Paragraph("te_to_dispatcher.py &mdash; the translator (a)", H3))
    s.append(Paragraph(
        "Reads a Tile Engine config JSON and emits one dispatcher config dict per "
        "valid <i>(tile &times; trait)</i> combination. Each dict has three parts:", BODY))
    s.append(bullets([
        "<b><font face='Courier'>_te</font></b> &mdash; the raw TE trait strings "
        "(<font face='Courier'>compv3 / intrawave / default</font>) kept verbatim, "
        "because codegen wants those names, not the canonical dispatcher forms.",
        "<b><font face='Courier'>signature</font></b> &mdash; the dispatcher "
        "<font face='Courier'>KernelKey</font> signature (dtypes, layouts, split_k, "
        "elementwise op&hellip;), already in canonical <font face='Courier'>to_string()</font> form.",
        "<b><font face='Courier'>algorithm</font></b> &mdash; tile/warp sizes, "
        "pipeline, scheduler, epilogue, pad flags, persistent, block_size.",
    ]))
    s.append(Paragraph(
        "It applies the TE&rarr;dispatcher mapping exactly <b>once</b> (e.g. scheduler "
        "<font face='Courier'>default&rarr;auto</font>, fp8/bf8 output&rarr;fp16, int8 "
        "acc&rarr;int32) and drops trait combos CK&nbsp;Tile does not support. Because "
        "the mapping happens here and only here, every downstream identifier is pure "
        "concatenation &mdash; which is what makes Python/C++ agreement provable.", BODY))

    s.append(Paragraph("identifier.py &mdash; Python identifier oracle (b)", H3))
    s.append(Paragraph(
        "Reproduces C++ <font face='Courier'>KernelKey::encode_identifier()</font> "
        "byte-for-byte from a config dict. It is deliberately dumb: no mapping, just "
        "concatenation in the exact field order of the C++ source, with optional "
        "suffixes (<font face='Courier'>_splitk{n}, _{op}, _d{n}, _sparse, "
        "_preshuffle</font>) emitted in the same order. <font face='Courier'>block_size</font> "
        "is intentionally omitted &mdash; it is part of equality but not the identifier.", BODY))
    s.append(code(
        "dtype_a _ layoutABC _ pipeline _ epilogue _ scheduler _\n"
        "padM _ padN _ padK _ persistent _\n"
        "TMxTNxTK _ WMxWNxWK _ WTMxWTNxWTK   [+ optional suffixes]"))

    s.append(Paragraph("cpp_identifier_oracle.cpp &mdash; C++ identifier oracle (b)", H3))
    s.append(Paragraph(
        "Includes the real <font face='Courier'>kernel_key.hpp</font> and calls the "
        "<i>actual runtime</i> <font face='Courier'>encode_identifier()</font>. It reads "
        "flat <font face='Courier'>key=value</font> lines from stdin, rebuilds a "
        "<font face='Courier'>KernelKey</font> via the same "
        "<font face='Courier'>string_to_*</font> helpers the runtime uses, and prints the "
        "identifier. Configs are batched with a <font face='Courier'>---</font> separator "
        "so one process handles thousands of configs (essential at 283,968 configs). "
        "<font face='Courier'>kernel_key.hpp</font> is pure host C++, so this builds with "
        "g++ alone &mdash; no GPU, no hipcc, no cmake.", BODY))

    s.append(Paragraph("check_identifier_parity.py &mdash; the diff (b)", H3))
    s.append(Paragraph(
        "Translates the TE JSON, compiles the C++ oracle if stale, runs every config "
        "through both oracles, and asserts they match byte-for-byte. If they agree, the "
        "registry key computed offline during codegen equals the one computed at runtime, "
        "so dispatch lookups <b>cannot silently miss</b>. Result on the full default "
        "config: <b>283968 / 283968 match</b>.", BODY))

    s.append(PageBreak())
    s.append(Paragraph("drive_codegen.py &mdash; single-config codegen (c)", H3))
    s.append(Paragraph(
        "Picks one translated config (by <font face='Courier'>--index</font>) and invokes "
        "the dispatcher's <font face='Courier'>unified_gemm_codegen.py</font> to emit "
        "exactly that one kernel header. The subtlety: codegen's "
        "<font face='Courier'>--config</font> path expects each tile/trait parameter to be "
        "a <i>flat single-element list</i> (it iterates <font face='Courier'>tc[\"tile_m\"]</font> "
        "directly), so we rebuild a minimal one-value-per-parameter config using the raw "
        "<font face='Courier'>_te</font> strings. The expected registry identifier is printed "
        "so the harness can find the generated <font face='Courier'>Kernel_&lt;id&gt;</font> "
        "struct.", BODY))

    s.append(Paragraph("harness.cpp + build_harness.sh &mdash; the single-kernel runner (d)", H3))
    s.append(Paragraph(
        "<font face='Courier'>harness.cpp</font> runs exactly one generated kernel through "
        "the <font face='Courier'>CK_TILE_SINGLE_KERNEL_INCLUDE</font> contract: defining "
        "that macro before including a generated header exposes a global "
        "<font face='Courier'>SelectedKernel</font> (with static "
        "<font face='Courier'>launch()</font>), the A/B/C data types, and "
        "<font face='Courier'>KERNEL_NAME</font>. The header path is injected at compile "
        "time as <font face='Courier'>PARITY_KERNEL_HEADER</font>, so one .cpp drives "
        "whichever kernel codegen produced.", BODY))
    s.append(bullets([
        "Builds an <font face='Courier'>rcr</font> fp16 GEMM "
        "(<font face='Courier'>A</font> row-major stride K, "
        "<font face='Courier'>B</font> col-major stride K, "
        "<font face='Courier'>C</font> row-major stride N) with deterministic inputs.",
        "Verifies against a CPU fp32 reference; tolerance scales as "
        "<font face='Courier'>1e-2 &middot; sqrt(K)</font>; prints "
        "<font face='Courier'>PASSED / FAILED</font>.",
        "An unsupported argument throws &rarr; caught and reported as "
        "<font face='Courier'>SKIPPED</font> (a skip, not a failure).",
        "Reports kernel time and GFLOP/s when timing is on.",
    ]))
    s.append(Paragraph(
        "<font face='Courier'>build_harness.sh</font> compiles it with "
        "<font face='Courier'>hipcc --offload-arch=&lt;gfx&gt;</font> against the CK include "
        "tree. It builds on a CPU box; only <i>running</i> needs a GPU.", BODY))

    s.append(Paragraph("check_parity.py &mdash; the orchestrator (e, f)", H3))
    s.append(Paragraph("Three escalating stages, in the required order:", BODY))
    s.append(bullets([
        "<b>Stage 1 &mdash; identifier</b> (always runs, CPU): delegates to "
        "check_identifier_parity. The offline&harr;runtime key guarantee.",
        "<b>Stage 2 &mdash; numerical</b> (GPU-gated): codegen &rarr; build harness &rarr; "
        "run <font face='Courier'>-verify=1</font> over several sizes and require "
        "<font face='Courier'>PASSED</font>. With <font face='Courier'>--te-build-dir</font>, "
        "the matching <font face='Courier'>benchmark_gemm_universal_&lt;name&gt;</font> is "
        "also run with verify; both must agree against the same reference.",
        "<b>Stage 3 &mdash; performance</b> (GPU-gated): "
        "<font face='Courier'>|disp - te| / te &le; --perf-tol</font> (default 10%). The "
        "harness reports GFLOP/s; it is converted to TFLOP/s to match Tile Engine's units.",
    ]))
    s.append(Paragraph(
        "A numerical failure short-circuits before performance is judged &mdash; enforcing "
        "&ldquo;numerical first, then performance.&rdquo; Without a GPU, stages 2&ndash;3 "
        "report <font face='Courier'>SKIPPED</font> (not FAILED) and the run still exits 0 "
        "if the identifier stage passed. <font face='Courier'>--dry-run</font> prints the "
        "full command plan without executing anything.", BODY))

    s.append(Paragraph("Two kinds of name &mdash; the key subtlety", H2))
    s.append(Paragraph(
        "There are <b>two</b> distinct names and the orchestrator uses each deliberately:", BODY))
    rows = [
        [cell("<b>Name</b>"), cell("<b>Form</b>"), cell("<b>Built by</b>"), cell("<b>Used for</b>")],
        [cell("Registry identifier"),
         cell("canonical (scheduler <font face='Courier'>default&rarr;auto</font>)"),
         cell("encode_identifier()"),
         cell("dispatch lookup; Stage&nbsp;1 proves it matches")],
        [cell("Kernel / file name"),
         cell("raw TE strings (<font face='Courier'>default</font> stays "
              "<font face='Courier'>default</font>)"),
         cell("te_kernel_name()"),
         cell("names <font face='Courier'>gemm_&lt;name&gt;.hpp</font> and "
              "<font face='Courier'>benchmark_gemm_universal_&lt;name&gt;</font>")],
    ]
    s.append(table(rows, [1.35 * inch, 1.9 * inch, 1.5 * inch, 1.75 * inch]))
    s.append(Paragraph(
        "They coincide for the <font face='Courier'>fp16_rcr&hellip;intrawave</font> example "
        "but diverge whenever a TE string maps to a different canonical form. Conflating "
        "them would make the orchestrator look for a header/executable that does not exist.",
        SMALL))

    s.append(Paragraph("Why this is trustworthy", H2))
    s.append(bullets([
        "The C++ oracle calls the <i>real</i> runtime "
        "<font face='Courier'>encode_identifier()</font> &mdash; not a re-implementation "
        "&mdash; so a match is a genuine codegen&harr;runtime guarantee.",
        "The mapping lives in exactly one place (the translator), so identifiers are pure "
        "concatenation and the Python/C++ agreement is mechanical.",
        "Numerical parity is adjudicated against an independent CPU reference on both sides; "
        "performance parity is a relative tolerance with explicit units.",
        "GPU-gating is honest: missing GPU is SKIPPED, not silently PASSED; "
        "<font face='Courier'>rocminfo</font> is authoritative (a bare "
        "<font face='Courier'>/dev/kfd</font> is treated as no GPU).",
    ]))
    return s


# --------------------------------------------------------------------------- #
# PDF 2: usage
# --------------------------------------------------------------------------- #
def usage_story():
    s = []
    s.append(Paragraph("Tile Engine &harr; Dispatcher Parity", H1))
    s.append(Paragraph("Basic usage", SUBTITLE))
    s.append(HRFlowable(width="100%", thickness=1, color=AMD_RED, spaceAfter=10))

    s.append(Paragraph("Where you are", H2))
    s.append(Paragraph(
        "All commands run from "
        "<font face='Courier'>dispatcher/parity/</font>. This box has python3, g++, and "
        "hipcc but <b>no GPU</b> and <b>no cmake</b>, so GPU stages auto-skip here and run "
        "on a GPU node unchanged.", BODY))

    s.append(Paragraph("What works without a GPU", H2))
    s.append(bullets([
        "Translating a TE config to dispatcher configs.",
        "<b>Identifier parity</b> &mdash; the main offline&harr;runtime guarantee.",
        "Driving codegen for one config (emits the header).",
        "<b>Building</b> the harness with hipcc (running it needs a GPU).",
        "<font face='Courier'>check_parity.py</font> Stage 1; Stages 2&ndash;3 report SKIPPED.",
    ]))

    s.append(Paragraph("Prerequisites", H2))
    s.append(code(
        "python3            # 3.8+\n"
        "g++ (or c++/clang++)   # for the C++ identifier oracle\n"
        "hipcc              # only to BUILD the harness (deliverable d)\n"
        "a ROCm GPU node    # only to RUN numerical / performance stages"))

    s.append(Paragraph("1. Identifier parity (CPU, fast)", H2))
    s.append(Paragraph("The one to run anywhere. Compiles the C++ oracle on first use.", BODY))
    s.append(code(
        "python check_identifier_parity.py configs/single_fp16_rcr.json\n"
        "python check_identifier_parity.py configs/single_fp16_rcr.json --verbose"))
    s.append(Paragraph("Expected tail:", SMALL))
    s.append(code(
        "identifier parity: 1/1 configs match\n"
        "(python encode_identifier vs C++ KernelKey::encode_identifier)"))

    s.append(Paragraph("2. Generate one kernel header", H2))
    s.append(code(
        "python drive_codegen.py configs/single_fp16_rcr.json --index 0\n"
        "\n"
        "# see the codegen command without running it:\n"
        "python drive_codegen.py configs/single_fp16_rcr.json --dry-run"))
    s.append(Paragraph(
        "Writes <font face='Courier'>generated/parity_single/gemm_&lt;name&gt;.hpp</font> "
        "and prints the expected registry identifier.", SMALL))

    s.append(Paragraph("3. Build the harness (hipcc)", H2))
    s.append(code(
        "./build_harness.sh                       # auto-picks the lone gemm_*.hpp\n"
        "./build_harness.sh path/to/gemm_X.hpp gfx942   # explicit header + arch"))
    s.append(Paragraph("Produces the <font face='Courier'>harness</font> binary. Run it on a "
                       "GPU node:", SMALL))
    s.append(code("./harness -m=512 -n=512 -k=512 -verify=1"))

    s.append(PageBreak())
    s.append(Paragraph("4. Full orchestration", H2))
    s.append(Paragraph("CPU box &mdash; Stage 1 runs, Stages 2&ndash;3 skip:", BODY))
    s.append(code("python check_parity.py configs/single_fp16_rcr.json"))
    s.append(Paragraph("See the entire plan (all three stages) without executing:", BODY))
    s.append(code("python check_parity.py configs/single_fp16_rcr.json --dry-run"))

    s.append(Paragraph("On a GPU node", H3))
    s.append(Paragraph("Dispatcher-only numerical + performance:", BODY))
    s.append(code(
        "python check_parity.py configs/single_fp16_rcr.json \\\n"
        "    --sizes 512x512x512,1024x1024x1024,2048x2048x2048 \\\n"
        "    --arch gfx942"))
    s.append(Paragraph("Full dispatcher-vs-Tile-Engine (numerical then performance):", BODY))
    s.append(code(
        "python check_parity.py configs/single_fp16_rcr.json \\\n"
        "    --te-build-dir /path/to/tile_engine/build \\\n"
        "    --perf-tol 0.10"))
    s.append(Paragraph(
        "<font face='Courier'>--te-build-dir</font> is searched recursively for "
        "<font face='Courier'>benchmark_gemm_universal_&lt;name&gt;</font>. Tile Engine "
        "writes <font face='Courier'>latency,tflops,bandwidth</font> to a CSV (only when it "
        "verifies), which the orchestrator parses for both the numerical pass and the "
        "performance baseline.", SMALL))

    s.append(Paragraph("check_parity.py options", H2))
    rows = [
        [cell("<b>Flag</b>"), cell("<b>Default</b>"), cell("<b>Meaning</b>")],
        [cell("<font face='Courier'>config</font>"), cell("(required)"),
         cell("Tile Engine config JSON (positional)")],
        [cell("<font face='Courier'>--index</font>"), cell("0"),
         cell("Which translated config to check")],
        [cell("<font face='Courier'>--sizes</font>"), cell("512&hellip;,1024&hellip;,2048&hellip;"),
         cell("Comma-separated <font face='Courier'>MxNxK</font> problem sizes")],
        [cell("<font face='Courier'>--arch</font>"), cell("gfx942"),
         cell("GPU arch for the harness build")],
        [cell("<font face='Courier'>--te-build-dir</font>"), cell("(none)"),
         cell("Enables dispatcher-vs-TE comparison")],
        [cell("<font face='Courier'>--perf-tol</font>"), cell("0.10"),
         cell("Relative throughput tolerance (10%)")],
        [cell("<font face='Courier'>--output-dir</font>"), cell("generated/"),
         cell("Codegen output directory")],
        [cell("<font face='Courier'>--kernel-set</font>"), cell("parity_single"),
         cell("Kernel-set subdirectory name")],
        [cell("<font face='Courier'>--dry-run</font>"), cell("off"),
         cell("Print the command plan; execute nothing")],
    ]
    s.append(table(rows, [1.55 * inch, 1.5 * inch, 3.45 * inch]))

    s.append(Paragraph("Reading the result", H2))
    s.append(bullets([
        "<b>identifier: PASS</b> &mdash; offline and runtime registry keys match.",
        "<b>numerical: PASS</b> &mdash; dispatcher (and TE, if given) verified for every size.",
        "<b>performance: PASS</b> &mdash; throughput within <font face='Courier'>--perf-tol</font>.",
        "<b>SKIPPED (no GPU / no hipcc)</b> &mdash; gated, not a failure; exit 0 if Stage 1 passed.",
        "Per-size <b>SKIPPED</b> &mdash; the kernel rejected those args; treated as a skip.",
    ]))
    s.append(Paragraph("Exit code is 0 when nothing failed (skips are OK), non-zero on any "
                       "real FAIL.", SMALL))

    s.append(Paragraph("Troubleshooting", H2))
    rows = [
        [cell("<b>Symptom</b>"), cell("<b>Cause / fix</b>")],
        [cell("<font face='Courier'>no host C++ compiler found</font>"),
         cell("Install g++/clang++; needed for the identifier oracle.")],
        [cell("Stage 2/3 always SKIPPED"),
         cell("No GPU detected (<font face='Courier'>rocminfo</font> shows no "
              "<font face='Courier'>gfx</font>). Run on a GPU node.")],
        [cell("<font face='Courier'>expected generated header not found</font>"),
         cell("Codegen failed earlier; re-run drive_codegen.py and read its output.")],
        [cell("TE executable not found"),
         cell("Wrong <font face='Courier'>--te-build-dir</font>, or that kernel was not "
              "built in Tile Engine.")],
        [cell("Harness fails at <font face='Courier'>hipMalloc</font>"),
         cell("No ROCm device &mdash; build is fine, you are on a CPU box.")],
    ]
    s.append(table(rows, [2.2 * inch, 4.3 * inch]))
    return s


def main() -> int:
    build(_HERE / "parity_design.pdf", "Design", design_story())
    build(_HERE / "parity_usage.pdf", "Usage", usage_story())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
