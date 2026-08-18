#!/usr/bin/python

################################################################################
#
# Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# Create a test_py script for all *.yaml files in specified directory
# usage: create_tests.py TEST_DIR
# Run from the Tensile/Tests directory, output script goes in the TEST_DIR/test_TEST_DIR.py

# The directory containing the test script can be passed to pytest:
# PYTHONPATH=. py.test-3 --durations=0 -v Tensile/Tests/TEST_DIR/
from __future__ import print_function
import glob, sys, os

targetDir  = sys.argv[1] if len(sys.argv) > 1 else "."
targetFile = "%s/test_%s.py"%(targetDir,os.path.basename(targetDir))
print("info: writing test script to %s" % targetFile)
outfile = open(targetFile, "w" )
outfile.write("import Tensile.Tensile as Tensile\n\n")
for f in glob.glob("%s/*aml"%targetDir):
    baseName = os.path.basename(f)
    testName = os.path.splitext(baseName)[0]
    testName = testName.replace('.','_')
    if not testName.startswith("test_"):
        testName = "test_" + testName

    outfile.write ("def %s(tmpdir):\n" % (testName))
    outfile.write (' Tensile.Tensile([Tensile.TensileTestPath("%s"), tmpdir.strpath])\n\n' % (f))

