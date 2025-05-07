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

import os
import pytest
import subprocess
import yaml

from Tensile import Tensile
from Tensile.Common.DataType import DataType

################################################################################
# Locate Executables
# rocm-smi, hip-clang, rocm_agent_enumerator
################################################################################
def isExe( filePath ):
  return os.path.isfile(filePath) and os.access(filePath, os.X_OK)

def locateExe( defaultPath, exeName ): # /opt/rocm/bin, hip-clang
  # look in path first
  for path in os.environ["PATH"].split(os.pathsep):
    exePath = os.path.join(path, exeName)
    if isExe(exePath):
      return exePath
  # look in default path second
  exePath = os.path.join(defaultPath, exeName)
  if isExe(exePath):
    return exePath
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
        for i,obj in enumerate(root):
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
     - validate / validateAll - whether the test validates (all?) results.
     - Data type(s) used in the YAML
     - Problem type(s) used in the YAML
     - Kernel language(s) used in the YAML
    """
    relpath = os.path.relpath(filepath, rootDir)
    components = relpath.split(os.path.sep)

    # First part of directory - nightly, pre-checkin, etc.
    marks = list([markNamed(component) for component in components[:-1]])

    if 'xfail' in relpath or 'wip' in relpath:
        marks.append(pytest.mark.xfail)
    if 'disabled' in relpath:
        marks.append(pytest.mark.skip)

    try:
        with open(filepath) as f:
            doc = yaml.load(f, yaml.SafeLoader)
    except yaml.parser.ParserError:
        marks.append(pytest.mark.syntax_error)
        return marks

    if "TestParameters" in doc:
        if "marks" in doc["TestParameters"]:
            marks += [markNamed(m) for m in doc["TestParameters"]["marks"]]

    # Architecture specific xfail marks
    for arch in availableArchs:
        ArchFail = "xfail-%s" % arch
        if markNamed(ArchFail) in marks:
            marks.append(pytest.mark.xfail)
        ArchSkip = "skip-%s" % arch
        if markNamed(ArchSkip) in marks:
            marks.append(pytest.mark.skip)

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
    #print ("***doc=", doc)
    for obj, path in walkDict(doc):
        #print ("  obj=", obj, "path=", path)
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

def findAvailableArchs():
    availableArchs = []
    rocmpath = "/opt/rocm"
    if "ROCM_PATH" in os.environ:
        rocmpath = os.environ.get("ROCM_PATH")
    if "TENSILE_ROCM_PATH" in os.environ:
        rocmpath = os.environ.get("TENSILE_ROCM_PATH")
    rocmAgentEnum = os.path.join(rocmpath, "bin/rocm_agent_enumerator")
    output = subprocess.check_output([rocmAgentEnum, "-t", "GPU"])
    lines = output.decode().splitlines()
    for line in lines:
        line = line.strip()
        if (not line in availableArchs) and (not "gfx000" in line):
            availableArchs.append(line)
    return availableArchs

def findConfigs(rootDir=None):
    """
    Walks rootDir (defaults to trying to find Tensile/Tests) and returns a
    list of test parameters, one for each YAML file.
    """
    if rootDir ==  None:
        rootDir = os.path.dirname(os.path.dirname(__file__))
        printRoot = os.path.dirname(os.path.dirname(rootDir))
    else:
        printRoot = rootDir

    availableArchs = findAvailableArchs()
    globaParamArchsStr = ';'.join(availableArchs)
    os.environ["PyTestBuildArchNames"] = globaParamArchsStr

    params = []
    for (dirpath, dirnames, filenames) in os.walk(rootDir):
        for filename in filenames:
            # Skip build client script
            if filename == "build_client.yaml":
                continue
            # filter out yamls in logic_yaml since they are not meant for Tensile.py
            elif filename.endswith('.yaml') and "logic_yaml" not in dirpath:
                filepath = os.path.join(rootDir, dirpath, filename)
                if not "test_data" in filepath:
                    marks = configMarks(filepath, rootDir, availableArchs)
                    relpath = os.path.relpath(filepath, printRoot)
                    params.append(pytest.param(filepath, marks=marks, id=relpath))
    return params

@pytest.mark.parametrize("config", findConfigs())
def test_config(tensile_args, config, tmpdir):
    Tensile.Tensile([config, tmpdir.strpath, *tensile_args])
