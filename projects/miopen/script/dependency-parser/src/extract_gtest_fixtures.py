#!/usr/bin/env python3
import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import concurrent.futures
import time
import json
import os
import re
import resource
import shlex
import stat
import subprocess
from pathlib import Path


# Macros whose names are extracted as fixtures/instantiations in the output.
FIXTURE_RE = re.compile(r"\b(?:TEST_F|TEST_P|TYPED_TEST)\s*\(\s*(\w+)")
FIXTURE_MACRO_RE = re.compile(r"\b(?:TEST_F|TEST_P|TYPED_TEST)\b")
INCLUDE_RE = re.compile(r'#\s*include\s*[<"]([^">]+)[">]')
_COMMENT_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _strip_comments(text):
    # Preprocessor strips comments before macros are seen; do the same so a
    # commented-out TEST_F does not register as a phantom fixture. String and
    # char literals are preserved so a "//" inside a string is not mistaken
    # for a comment.
    def repl(match):
        token = match.group(0)
        return " " if token[0] == "/" else token

    return _COMMENT_RE.sub(repl, text)


def _gtest_root(files):
    # Locate the .../test/gtest directory from any TU path so header scanning
    # works regardless of the current working directory.
    for file in files:
        parts = Path(file).parts
        for i in range(len(parts) - 1):
            if parts[i] == "test" and parts[i + 1] == "gtest":
                return Path(*parts[: i + 2])
    return None


def _build_hot_headers(gtest_root):
    # Headers that themselves contain fixture macros: any TU including one gets
    # fixtures via expansion, not from its own text, so it must be preprocessed.
    headers = {}
    for path in Path(gtest_root).rglob("*"):
        if path.suffix in (".hpp", ".h", ".hxx", ".inc", ".ipp") and path.is_file():
            try:
                headers[path.name] = _strip_comments(
                    path.read_text(encoding="utf-8", errors="ignore")
                )
            except OSError:
                pass

    hot = {name for name, text in headers.items() if FIXTURE_MACRO_RE.search(text)}
    # A header that includes a hot header is itself hot (transitive).
    changed = True
    while changed:
        changed = False
        for name, text in headers.items():
            if name in hot:
                continue
            included = {Path(i).name for i in INCLUDE_RE.findall(text)}
            if included & hot:
                hot.add(name)
                changed = True
    return hot


def _fixture_under_conditional(stripped):
    # True only if a fixture macro sits inside an #if/#ifdef/#ifndef block,
    # where conditional compilation could enable or disable it. A conditional
    # that gates only helpers or acts as an include guard does not count, so
    # such files stay on the fast raw path.
    depth = 0
    for line in stripped.splitlines():
        stripline = line.lstrip()
        if not stripline.startswith("#"):
            if depth > 0 and FIXTURE_MACRO_RE.search(line):
                return True
            continue
        directive = stripline[1:].lstrip()
        if directive.startswith(("if", "ifdef", "ifndef")):
            depth += 1
        elif directive.startswith("endif"):
            depth = max(0, depth - 1)
    return False


def _needs_preprocess(text, hot_headers):
    stripped = _strip_comments(text)
    # Conditional compilation can enable/disable a fixture.
    if _fixture_under_conditional(stripped):
        return True
    # A fixture name built by token paste is only correct after expansion.
    for match in re.finditer(r"\b(?:TEST_F|TEST_P|TYPED_TEST)\s*\(", stripped):
        arg = stripped[match.end() : match.end() + 200].split(")", 1)[0]
        if "##" in arg:
            return True
    # Fixtures pulled in from a header rather than written in this TU.
    included = {Path(i).name for i in INCLUDE_RE.findall(stripped)}
    if included & hot_headers:
        return True
    return False


def _find_fixtures_raw(text):
    fixtures = FIXTURE_RE.findall(_strip_comments(text))
    return list(dict.fromkeys(fixtures))


def find_gtests_for_tu(task):
    gtest, file_path, hot_headers = task
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""

    if text and not _needs_preprocess(text, hot_headers):
        return _find_fixtures_raw(text)
    return preprocess_and_find_gtests_thread(gtest)


def preprocess_and_find_gtests_thread(gtest):
    pp_folder = "test/gtest/pp"
    build_command = shlex.split(gtest)
    build_command.insert(1, "/P" if sys.platform == "win32" else "-E")

    # Flags that trigger the HIP/offload codegen toolchain or are otherwise
    # irrelevant to preprocessing. Dropping them avoids per-file driver overhead
    # so the source is preprocessed as plain C++ (files are .cpp).
    preprocess_command = []
    flagged = False
    skip = -1
    drop_next = False
    for index, token in enumerate(build_command):
        if drop_next:
            drop_next = False
            continue

        if token == "-x":
            drop_next = True  # also drop the language value (e.g. "hip")
            continue
        if (
            token.startswith("--offload-arch")
            or token == "--hip-link"
            or token.startswith("-O")
        ):
            continue

        if token.startswith("-I") and not flagged:
            preprocess_command.append(f"-I{pp_folder}")
            preprocess_command.append("-I../test")
            flagged = True
        elif token == "-o":
            skip = index + 1
            outfile = (
                pp_folder + "/" + Path(build_command[index + 1]).with_suffix(".i").name
            )

        if index > skip:
            preprocess_command.append(token)

    try:
        result = subprocess.run(
            preprocess_command, capture_output=True, text=True, check=True
        )

    except subprocess.CalledProcessError as e:
        # 3. Gracefully catch compilation errors if the command fails
        print("❌ Compilation failed!", file=sys.stderr)
        print(f"Exit Code: {e.returncode}", file=sys.stderr)
        print(f"Compiler Error Messages:\n{e.stderr}", file=sys.stderr)
        sys.exit(e.returncode)

    new_fixtures = re.findall(
        r"\b(?:TEST_F|TEST_P|TYPED_TEST)\s*\(\s*(\w+)", result.stdout
    )
    new_fixtures = list(dict.fromkeys(new_fixtures))

    return new_fixtures


def extract_gtext_fixtures(compile_commands: str, output_file: str, pp_folder: str):
    exes = []
    gtests = []
    files = []

    try:
        with open(compile_commands, "r", encoding="utf-8") as f:
            data = json.load(f)

            for entry in data:
                file_path = Path(entry.get("file", ""))
                parts = file_path.parts

                # Check if the file is located in the test/gtest folder
                if any(
                    parts[i] == "test" and parts[i + 1] == "gtest"
                    for i in range(len(parts) - 1)
                ):
                    exes.append(f"bin/test_{file_path.stem}")
                    gtests.append(entry.get("command"))
                    files.append(entry.get("file"))

    except FileNotFoundError:
        print(f"Error: {compile_commands} not found.")
    except json.JSONDecodeError:
        print(f"Error: {compile_commands} is not a valid JSON file.")

    # spoof gtest.h so TEST* macros are not expanded
    gtest_hpp = Path(pp_folder) / "gtest" / "gtest.h"
    gtest_hpp.parent.mkdir(parents=True, exist_ok=True)
    with open(gtest_hpp, "w") as f:
        pass

    # Files whose fixtures can be read straight from source (no conditionals,
    # no token-paste, no fixtures inherited from a header) skip preprocessing.
    gtest_root = _gtest_root(files)
    hot_headers = _build_hot_headers(gtest_root) if gtest_root else set()
    tasks = [(gtest, file, hot_headers) for gtest, file in zip(gtests, files)]

    # Leave cores free for the dependency parser, which runs concurrently.
    # Reserve up to half the machine, 32 cores max.
    cpu_count = os.cpu_count() or 1
    reserved = min(32, cpu_count // 2)
    thread_count = max(1, cpu_count - reserved)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        all_fixtures = executor.map(find_gtests_for_tu, tasks)

    results = {}
    fixture_count = 0
    for exe, fixtures in zip(exes, all_fixtures):
        if fixtures:
            results[exe] = fixtures
            fixture_count += len(fixtures)

    # Write results to JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(
            f"List of {fixture_count} fixtures from {len(results)} files written to {output_file}"
        )
    except Exception as e:
        print(f"Error writing JSON file: {e}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        compile_commands = "compile_commands.json"
    else:
        compile_commands = sys.argv[1]

    compile_commands_path = Path(compile_commands)
    if not compile_commands_path.is_file():
        print(
            f"Usage: {sys.argv[0]} [<path_to_compile_commands.json> [<path_to_miopen_dapper_fixtures.json>]]",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(sys.argv) < 3:
        output_file = "miopen_dapper_fixtures.json"
    else:
        output_file = sys.argv[2]

    pp_folder = "test/gtest/pp"
    t0 = time.monotonic()
    extract_gtext_fixtures(compile_commands, output_file, pp_folder)
    elapsed = time.monotonic() - t0
    print(f"extract_gtext_fixtures completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
