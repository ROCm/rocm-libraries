################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function

import itertools
import sys
import time
import yaml

sys.path.append('../..')
from Tensile.SolutionStructs import Solution
from Tensile import Utils
from Tensile.Utilities.ConditionalImports import yamlDumper

def merge_libraries(args):
    inFiles = args[:-1]
    outFile = args[-1]

    with open(inFiles[0]) as inf:
        data = yaml.load(inf)

    masterLibrary = MasterSolutionLibrary.FromOriginalState(data)

    for inFile in Utils.tqdm(inFiles[1:], desc="Merge libraries"):
        with open(inFile) as inf:
            data = yaml.load(inf)
        newLibrary = MasterSolutionLibrary.FromOriginalState(data)
        masterLibrary.merge(newLibrary)
        del newLibrary

    masterLibrary.applyMinNaming()
    outData = state(masterLibrary)

    with open(outFile, 'w') as outf:
        if True:
            yaml.dump(outData, outf, yamlDumper)
        else:
            import json
            json.dump(outData, outf, sort_keys=True, indent=2, separators=(",", ": "))

def convert_one(args):

    with open(args[0]) as inFile:
        data = yaml.load(inFile)

    if True:
        masterLibrary = MasterSolutionLibrary.FromOriginalState(data)
        #import pdb
        #pdb.set_trace()
        outData = state(masterLibrary)

    else:
        originalSolutions = data[5]
        #print(originalSolutions)
        newSolutions = []
        for s in originalSolutions:
            newSolutions.append(ContractionSolution.FromOriginalState(s))

        outData = [state(s) for s in newSolutions]

    with open(args[1], 'w') as outFile:
        if True:
            yaml.dump(outData, outFile, yamlDumper)
        else:
            import json
            json.dump(outData, outFile, sort_keys=True, indent=2, separators=(",", ": "))

if __name__ == "__main__":

    for i in Utils.tqdm(itertools.chain([1,2,3], [4,5,6]), desc="Converting YAML"): time.sleep(1)

    merge_libraries(sys.argv[1:])
