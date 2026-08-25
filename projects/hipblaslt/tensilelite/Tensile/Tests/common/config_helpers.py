################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Pytest-specific test configuration helpers.

For artifact compression/extraction utilities (no pytest dependency),
see artifact_helpers.py.
"""

import os

import pytest
import yaml

from Tensile.Common.DataType import DataType

_TESTS_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Safe either way; bandit's B506 check only recognises the SafeLoader/CSafeLoader spelling,
# so the call sites using this name carry a bare nosec marker. Never spell that marker out
# with its leading hash here, or bandit parses this comment too (SEC-00404).
try:
    DEFAULT_YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    DEFAULT_YAML_LOADER = yaml.SafeLoader


def get_rocm_version_or_none():
    """Gets the ROCm version from the version file."""
    try:
        rocmpath = os.environ.get("ROCM_PATH", "/opt/rocm")
        version_file_path = os.path.join(rocmpath, ".info/version")

        with open(version_file_path, 'r') as f:
            version_string = f.readline().strip()
            return version_string
    except (FileNotFoundError, IOError):
        return None


def walkDict(root, path=""):
    """
    Recursively walks a structure which may consist of dictionaries, lists,
    and other objects. Yields (object, path) for each object in the
    structure.
    """
    yield root, path
    if isinstance(root, dict):
        for key, value in root.items():
            keypath = key
            if path != "":
                keypath = path + "." + str(keypath)
            yield from walkDict(value, keypath)
    elif isinstance(root, list):
        for i, obj in enumerate(root):
            keypath = str(i)
            if path != "":
                keypath = path + "." + keypath
            yield from walkDict(obj, keypath)


def markNamed(name):
    """
    Gets a mark by a name contained in a variable.
    """
    return getattr(pytest.mark, name)


def configMarks(filepath, rootDir, availableArchs):
    """
    Returns a list of marks to add to a particular YAML config path.  Currently gets a mark for:

     - Root directory name.  This separates tests into pre_checkin, nightly, etc.
     - Expected failures. Include 'xfail' in the name of the YAML file.
     - Anything in yaml["TestParameters"]["marks"]
     - TestParameters.RevisionID (gfx1250 only: 1 -> skip-gfx1250v0)
     - Architecture from GlobalParameters.Architecture (e.g. gfx1250)
     - Architecture from filename (e.g. bf16_gfx1250.yaml -> gfx1250;
       gfx1250v0 wins over gfx\\d+)
     - validate / validateAll - whether the test validates (all?) results.
     - Data type(s) used in the YAML
     - Problem type(s) used in the YAML
     - Kernel language(s) used in the YAML
    """
    relpath = os.path.relpath(filepath, rootDir)
    components = relpath.split(os.path.sep)

    # First part of directory - nightly, pre-checkin, etc.
    # Skip underscore-prefixed path components (e.g. characterization's _codegen
    # fixture dir): pytest rejects marks starting with "_", and such dirs hold
    # library logic YAMLs that are filtered out below by the not-dict guard anyway.
    marks = list([markNamed(c) for c in components[:-1] if not c.startswith("_")])

    if 'xfail' in relpath or 'wip' in relpath:
        marks.append(pytest.mark.xfail)
    if 'disabled' in relpath:
        marks.append(pytest.mark.skip)

    try:
        with open(filepath) as f:
            doc = yaml.load(f, DEFAULT_YAML_LOADER)  # nosec B506
    except yaml.parser.ParserError:
        marks.append(pytest.mark.syntax_error)
        return marks

    # A Tensile config is a mapping (GlobalParameters/BenchmarkProblems/...).
    # Top-level sequences are library logic YAMLs (e.g. characterization data
    # files), which are not standalone Tensile.py configs. Signal the caller to
    # skip them rather than crashing on doc["BenchmarkProblems"].
    if not isinstance(doc, dict):
        return None

    if "TestParameters" in doc:
        if "marks" in doc["TestParameters"]:
            marks += [markNamed(m) for m in doc["TestParameters"]["marks"]]

    # gfx1250 RevisionID 1 (default 0) is the same skip as skip-gfx1250v0.
    # StreamK tests that set either are skipped on gfx1250 rev0 only.
    from Tensile.Gfx1250RunGuard import requires_gfx1250_rev1
    if requires_gfx1250_rev1(doc, filepath):
        skip_v0 = markNamed("skip-gfx1250v0")
        if skip_v0 not in marks:
            marks.append(skip_v0)

    arch_val = doc.get("GlobalParameters", {}).get("Architecture")
    if arch_val and markNamed(arch_val) not in marks:
        marks.append(markNamed(arch_val))

    from Tensile.Tests.gpu_detection import filename_arch_token

    arch_in_name = filename_arch_token(components[-1])
    if arch_in_name and markNamed(arch_in_name) not in marks:
        marks.append(markNamed(arch_in_name))

    # Architecture specific xfail marks
    for arch in availableArchs:
        ArchFail = "xfail-%s" % arch
        if markNamed(ArchFail) in marks:
            marks.append(pytest.mark.xfail)
        ArchSkip = "skip-%s" % arch
        if markNamed(ArchSkip) in marks:
            marks.append(pytest.mark.skip)

    # FFM-specific xfail: a config marked ``ffm_fail`` passes on real HW but
    # fails under FFM emulation only. Turn it into an xfail only when running 
    # under FFM — keyed on the emulator's HSA_MODEL_MEMFILE backing plus the 
    # gfx1250 arch — so it never fires on HW or on other emulators/arches, 
    # where the test must still run.
    if (
        os.environ.get("HSA_MODEL_MEMFILE")
        and "gfx1250" in availableArchs
        and markNamed("ffm_fail") in marks
    ):
        marks.append(pytest.mark.xfail)

    validate = True
    validateAll = False
    try:
        if doc["GlobalParameters"]['NumElementsToValidate'] == 0:
            validate = False
        if doc["GlobalParameters"]['NumElementsToValidate'] == -1:
            validateAll = True
    except KeyError:
        pass

    if validate:
        marks.append(pytest.mark.validate)
    if validateAll:
        marks.append(pytest.mark.validateAll)

    dataTypes = set([problem[0]["DataType"] for problem in doc["BenchmarkProblems"]])
    operationTypes = set([problem[0]["OperationType"] for problem in doc["BenchmarkProblems"]])

    languages = set()
    for obj, path in walkDict(doc):
        if "KernelLanguage" in path and isinstance(obj, str):
            languages.add(obj)

    for l in languages:
        marks.append(markNamed(l))

    for dt in dataTypes:
        dataType = DataType(dt)
        marks.append(markNamed(dataType.toName()))

    for operationType in operationTypes:
        marks.append(markNamed(operationType))

    return marks

def findAvailableArchs(gpu_targets=None):
    """Detect available GPU architectures, or use an explicit override.

    Args:
        gpu_targets: Semicolon-separated GPU targets (e.g. "gfx942").
            When provided, skips hardware detection entirely.

    Returns:
        List of architecture strings (e.g. ["gfx942"]).
    """
    if gpu_targets:
        return [t.strip() for t in gpu_targets.split(";") if t.strip()]

    from Tensile.Tests.gpu_detection import get_available_archs
    return get_available_archs()


def findConfigs(rootDir=None, availableArchs=None, skipArchs=None):
    """
    Walks rootDir (defaults to trying to find Tensile/Tests) and returns a
    list of test parameters, one for each YAML file.

    Args:
        rootDir: Directory to walk for YAML configs. Defaults to Tensile/Tests.
        availableArchs: Pre-resolved compile/build GPU architectures
            (PyTestBuildArchNames). When None, calls findAvailableArchs().
        skipArchs: Architectures used for skip-*/xfail-* matching. Compile
            targets are not the skip identity; when None, derived from
            availableArchs plus hardware revision probing.
    """
    if rootDir is None:
        rootDir = _TESTS_ROOT_DIR
        printRoot = os.path.dirname(os.path.dirname(rootDir))
    else:
        printRoot = rootDir

    if availableArchs is None:
        availableArchs = findAvailableArchs()
    globalParamArchsStr = ';'.join(availableArchs)
    os.environ["PyTestBuildArchNames"] = globalParamArchsStr

    if skipArchs is None:
        from Tensile.Tests.gpu_detection import resolve_skip_archs
        skipArchs = resolve_skip_archs(availableArchs)

    rocm_version = get_rocm_version_or_none()

    params = []
    for (dirpath, dirnames, filenames) in os.walk(rootDir):
        for filename in filenames:
            # Conditionally skip icache_flush.yaml on rocm 7.1 due to ROCm bug.
            if filename == "icache_flush.yaml" and rocm_version and rocm_version.startswith("7.1"):
                print(f"INFO: Skipping '{filename}' on ROCm {rocm_version}.")
                continue

            # Skip build client script
            if filename == "build_client.yaml":
                continue
            # filter out yamls in logic_yaml since they are not meant for Tensile.py
            elif filename.endswith('.yaml') and "logic_yaml" not in dirpath:
                filepath = os.path.join(rootDir, dirpath, filename)
                if not "test_data" in filepath:
                    marks = configMarks(filepath, rootDir, skipArchs)
                    if marks is None:
                        # Not a Tensile config (e.g. a library logic YAML); skip.
                        continue

                    # Conditionally xfail icache_flush.yaml on rocm 7.1 due to ROCm bug.
                    if filename == "icache_flush.yaml" and rocm_version and rocm_version.startswith("7.1"):
                        reason = "Test is expected to fail on ROCm 7.1 due to a known bug."
                        marks.append(pytest.mark.xfail(reason=reason, strict=True))

                    relpath = os.path.relpath(filepath, printRoot)
                    params.append(pytest.param(filepath, marks=marks, id=relpath))
    return params
