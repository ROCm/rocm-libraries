#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the TensileLite three-phase pipeline on a benchmark YAML for gfx950.

Phases run in order: BenchmarkProblems -> LibraryLogic -> optional ClientWriter.
Every run uses a fresh mktemp working directory so no stale kernel or solution
cache is reused. The produced 3_LibraryLogic/*.yaml files are copied into the
gfx950 device-library logic tree, and a summary of the top benchmark winners
plus any errors and warnings is printed.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

gpuTarget = "gfx950"
defaultTopN = 5

# Repo layout anchors derived from this file: <repo>/tensilelite/scripts/<this>.
scriptsDir = Path(__file__).resolve().parent
tensileliteDir = scriptsDir.parent
repoRoot = tensileliteDir.parent
binTensile = tensileliteDir / "Tensile" / "bin" / "Tensile"
# Base gfx950 schedule variant directory in the device-library logic tree.
defaultLogicDir = (
    repoRoot
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950"
)
venvPython = Path.home() / ".tensile" / "bin" / "python"

Winner = namedtuple("Winner", ["gflops", "sizes", "kernel", "problemType"])


def parseArgs(argv):
    parser = argparse.ArgumentParser(
        description="Run the TensileLite benchmark -> logic pipeline for gfx950.",
    )
    parser.add_argument("yamlFile", type=Path,
                        help="TensileLite benchmark config YAML.")
    parser.add_argument("--top-n", dest="topN", type=int, default=defaultTopN,
                        help="number of top benchmark winners to display.")
    parser.add_argument("--keep-tmp", dest="keepTmp", action="store_true",
                        help="keep the temporary working directory.")
    parser.add_argument("--output-dir", dest="outputDir", type=Path, default=None,
                        help="override the destination for the produced logic YAMLs.")
    parser.add_argument("--with-client", dest="withClient", action="store_true",
                        help="also run the ClientWriter phase (off by default).")
    parser.add_argument("--python", dest="python", type=Path, default=None,
                        help="python interpreter used to run Tensile.")
    return parser.parse_args(argv)


def resolvePython(explicit):
    if explicit is not None:
        return str(explicit)
    if venvPython.exists():
        return str(venvPython)
    print(f"warning: {venvPython} not found; falling back to {sys.executable}",
          file=sys.stderr)
    return sys.executable


def hasTopLevelSection(yamlPath, section):
    """Return True if the YAML has a top-level (unindented) mapping key."""
    pattern = re.compile(rf"^{re.escape(section)}\s*:", re.MULTILINE)
    return bool(pattern.search(yamlPath.read_text()))


def stripLibraryClient(pythonExe, srcYaml, dstYaml):
    """Write a copy of srcYaml with the LibraryClient section removed.

    Done through the Tensile python (which has PyYAML) so this script keeps no
    third-party import of its own. LibraryLogic is deliberately preserved.
    """
    program = (
        "import sys, yaml;"
        "d = yaml.safe_load(open(sys.argv[1]));"
        "d.pop('LibraryClient', None);"
        "yaml.safe_dump(d, open(sys.argv[2], 'w'), sort_keys=False)"
    )
    subprocess.run([pythonExe, "-c", program, str(srcYaml), str(dstYaml)],
                   check=True)


def prepareConfig(args, pythonExe, tmpRoot):
    """Return the config path to run: the original, or a client-stripped copy."""
    if args.withClient or not hasTopLevelSection(args.yamlFile, "LibraryClient"):
        return args.yamlFile
    stripped = tmpRoot / "config_no_client.yaml"
    stripLibraryClient(pythonExe, args.yamlFile, stripped)
    print(f"# Stripped LibraryClient section -> {stripped}")
    return stripped


def buildEnv():
    """Environment for the Tensile subprocess.

    /opt/rocm/lib must be on LD_LIBRARY_PATH or gfx950 chip detection (and thus
    the on-device benchmark) silently fails.
    """
    env = dict(os.environ)
    rocmLib = "/opt/rocm/lib"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{rocmLib}:{existing}" if existing else rocmLib
    return env


def runTensile(pythonExe, configPath, outDir, logPath, phaseArgs, append):
    """Run one Tensile invocation, teeing output to console and logPath."""
    cmd = [
        pythonExe, str(binTensile), str(configPath), str(outDir),
        "--gpu-targets", gpuTarget, *phaseArgs,
    ]
    print(f"# Running: {' '.join(cmd)}")
    with open(logPath, "a" if append else "w") as log:
        proc = subprocess.Popen(cmd, cwd=str(tensileliteDir), env=buildEnv(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    return proc.returncode


def runPipeline(pythonExe, configPath, outDir, logPath):
    """Run the full BenchmarkProblems -> LibraryLogic pipeline in one pass.

    A single Tensile invocation writes ClientParameters with the problem's
    DQuant type/size derived from the solution set. The earlier two-phase
    --build-only/--use-cache split regenerated ClientParameters without those
    fields (the BenchmarkProblems cache path omits dquantType), so every
    solution failed its on-device DQuant predicate and no benchmark data was
    produced.
    """
    return runTensile(pythonExe, configPath, outDir, logPath, [], False)


def toFloat(cell):
    try:
        return float(cell.strip())
    except (ValueError, AttributeError):
        return None


# Bookkeeping columns the benchmark CSV emits before the per-solution columns.
metadataColumns = {"GFlops", "TotalFlops", "GbpsBW"}


def isMetadataColumn(name):
    """True for a bookkeeping column that precedes the per-solution columns."""
    name = name.strip()
    return name in metadataColumns or name.startswith("Size") or name.startswith("LD")


def firstSolutionColumn(header):
    """Index of the first per-solution column; metadata columns come first."""
    for i in range(1, len(header)):
        if not isMetadataColumn(header[i]):
            return i
    return len(header)


def sizeColumns(header):
    """Indices of the problem-size columns (header names start with 'Size')."""
    return [i for i in range(len(header)) if header[i].strip().startswith("Size")]


def parseCsvWinners(csvPath):
    """Return the best (gflops, kernel) Winner for each problem-size row.

    A valid winner needs a strictly positive gflops; rows where every solution
    reports -1/-nan (a rejected or faulted kernel) yield no winner.
    """
    winners = []
    with open(csvPath) as f:
        header = f.readline().strip().split(",")
        solStart = firstSolutionColumn(header)
        sizeIdx = sizeColumns(header)
        names = [n.strip() for n in header[solStart:]]
        for line in f:
            cells = line.rstrip("\n").split(",")
            if len(cells) <= solStart:
                continue
            best, bestIdx = 0.0, -1
            for i, cell in enumerate(cells[solStart:]):
                val = toFloat(cell)
                if val is not None and val > best:
                    best, bestIdx = val, i
            if bestIdx < 0:
                continue
            kernel = names[bestIdx] if bestIdx < len(names) else f"Solution_{bestIdx}"
            sizes = [cells[j].strip() for j in sizeIdx if j < len(cells)]
            winners.append(Winner(best, sizes, kernel, csvPath.stem))
    return winners


def collectWinners(outDir):
    """Parse all benchmark result CSVs, preferring the winner-export files."""
    dataDir = outDir / "2_BenchmarkData"
    if not dataDir.is_dir():
        return []
    allCsv = [Path(p) for p in glob.glob(str(dataDir / "*.csv"))
              if not p.endswith("_Granularity.csv")]
    winnerCsv = [p for p in allCsv if "CSVWinner" in p.name]
    chosen = winnerCsv if winnerCsv else allCsv
    winners = []
    for csvPath in chosen:
        winners.extend(parseCsvWinners(csvPath))
    winners.sort(key=lambda w: w.gflops, reverse=True)
    return winners


def collectDiagnostics(logPath):
    """Return (errors, warnings) lines scraped from the run log."""
    errorRe = re.compile(r"\b(error|exception|traceback|failed)\b|\(ERROR\)", re.I)
    warnRe = re.compile(r"\bwarning\b", re.I)
    errors, warnings = [], []
    for line in Path(logPath).read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if errorRe.search(stripped) and stripped not in errors:
            errors.append(stripped)
        elif warnRe.search(stripped) and stripped not in warnings:
            warnings.append(stripped)
    return errors[:30], warnings[:30]


def inferFailedPhase(outDir, withClient):
    """Best-effort phase attribution from which output dirs got populated."""
    csvs = list((outDir / "2_BenchmarkData").glob("*.csv"))
    if not csvs or all(p.stat().st_size == 0 for p in csvs):
        return "BenchmarkProblems"
    if not list((outDir / "3_LibraryLogic").glob("*.yaml")):
        return "LibraryLogic"
    if withClient and not (outDir / "4_LibraryClient").is_dir():
        return "ClientWriter"
    return "unknown"


def copyLogic(outDir, destDir):
    """Copy produced logic YAMLs into destDir; return the copied paths."""
    logicDir = outDir / "3_LibraryLogic"
    produced = sorted(Path(p) for p in glob.glob(str(logicDir / "*.yaml")))
    if not produced:
        return []
    destDir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in produced:
        dst = destDir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def printWinners(winners, topN):
    print("\n" + "=" * 80)
    print(f"TOP {topN} BENCHMARK WINNERS (by GFlops)")
    print("=" * 80)
    if not winners:
        print("  No benchmark results were produced.")
        return
    for rank, w in enumerate(winners[:topN], start=1):
        sizes = "x".join(w.sizes)
        print(f"  #{rank:<2} {w.gflops:>12.2f} GFlops  sizes=[{sizes}]  ({w.problemType})")
        print(f"       kernel: {w.kernel}")


def printCopied(copied, destDir):
    print("\n" + "-" * 80)
    print("COPIED LOGIC FILES")
    print("-" * 80)
    if not copied:
        print("  None (no 3_LibraryLogic/*.yaml was produced).")
        return
    print(f"  Destination: {destDir}")
    for path in copied:
        print(f"    {path.name}")


def printDiagnostics(errors, warnings):
    print("\n" + "-" * 80)
    print(f"WARNINGS ({len(warnings)})")
    print("-" * 80)
    for w in warnings:
        print(f"  {w}")
    print(f"\nERRORS ({len(errors)})")
    print("-" * 80)
    for e in errors:
        print(f"  {e}")


def main(argv):
    args = parseArgs(argv)
    if not args.yamlFile.is_file():
        print(f"error: config YAML not found: {args.yamlFile}", file=sys.stderr)
        return 2
    if not binTensile.exists():
        print(f"error: Tensile entry point not found: {binTensile}", file=sys.stderr)
        return 2

    pythonExe = resolvePython(args.python)
    destDir = args.outputDir if args.outputDir is not None else defaultLogicDir
    tmpRoot = Path(tempfile.mkdtemp(prefix="benchmark_yaml_"))
    logPath = tmpRoot / "tensile_run.log"
    keepForDebug = args.keepTmp

    try:
        configPath = prepareConfig(args, pythonExe, tmpRoot)
        rc = runPipeline(pythonExe, configPath, tmpRoot, logPath)

        winners = collectWinners(tmpRoot)
        errors, warnings = collectDiagnostics(logPath)

        # A zero exit with no winners still means no usable data was produced
        # (kernels rejected or faulted on device). Treat it as a failure so a
        # bogus or empty logic file is never copied into the device tree.
        failed = rc != 0 or not winners
        copied = [] if failed else copyLogic(tmpRoot, destDir)

        printWinners(winners, args.topN)
        printCopied(copied, destDir)
        printDiagnostics(errors, warnings)

        if not failed:
            return 0

        keepForDebug = True
        phase = inferFailedPhase(tmpRoot, args.withClient)
        reason = f"exit {rc}" if rc != 0 else "no benchmark winners produced"
        print(f"\npipeline FAILED ({reason}) during phase: {phase}",
              file=sys.stderr)
        return rc if rc != 0 else 1
    finally:
        if keepForDebug:
            print(f"\n# Kept temporary working dir: {tmpRoot}")
            print(f"# Full run log: {logPath}")
        else:
            shutil.rmtree(tmpRoot, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
