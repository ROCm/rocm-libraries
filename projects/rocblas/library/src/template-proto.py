"""Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import argparse
import glob
import re

###############################################################################
# Template to prototype translator
###############################################################################

gPattern = re.compile(r'\{[^\{\}]*\}')


def translateToProto(templateCode):
    global gPattern
    proto = ''.join(templateCode)
    if re.search("ROCBLAS_INTERNAL_EXPORT_NOINLINE", proto) is None:
        return
    # keep warning in proto
    proto = re.sub("ROCBLAS_INTERNAL_EXPORT_NOINLINE", "ROCBLAS_INTERNAL_DEPRECATION", proto)
    n = 1
    while n:
        proto, n = re.subn(gPattern, '', proto)

    if (proto.rstrip()).endswith(';'):
        print(proto.rstrip() + "\n")
    else:
        print(proto.rstrip() + ";\n")


def parseForExportedFunctions(inputFileName):
    with open(inputFileName) as f:
        haveFunction = False
        lines = f.readlines()
        for line in lines:
            if(not haveFunction):
                start = re.match(r'^template|^ROCBLAS_INTERNAL_EXPORT_NOINLINE', line)
                if(start):
                    end = re.match(r'.*\)', line)
                    if(end):
                        translateToProto(line)
                    else:
                        body = []
                        body.append(line)
                        haveFunction = True
            else:
                body.append(line)
                end = re.match(r'.*\)', line)
                if(end):
                    translateToProto(body)
                    haveFunction = False



def RunExporter():
    print("""
// Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.

// Script-generated file -- do not edit

// rocBLAS internal API may change each release. The rocBLAS team strongly advises against its use.

#pragma once

#include "rocblas/internal/rocblas-types.h"

""")

    # Parse Command Line Arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('path', nargs='+',
                           help='Path of a files or directory')
    args = argParser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)
        else:
            files.update(glob.glob(path+'*.h*'))

    headerFiles = sorted(files)
    for f in headerFiles:
        parseForExportedFunctions(f)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    RunExporter()
