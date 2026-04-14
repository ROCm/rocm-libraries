#!/usr/bin/env python3
# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import sys
import os
import argparse
import hashlib
import re


def filename_to_cpp_ident(filename):
    base = os.path.basename(filename)
    return base.replace('-', '_').replace('.', '_')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Write embedded C++ generator file")
    parser.add_argument('--embed',
                        metavar='file',
                        type=str,
                        nargs='+',
                        required=True,
                        help='files to embed into the header')
    parser.add_argument('--logic',
                        metavar='file',
                        type=str,
                        nargs='+',
                        required=True,
                        help='additional files that make up generator logic')
    parser.add_argument('--output',
                        metavar='file',
                        type=str,
                        required=True,
                        help='output file')
    args = parser.parse_args()

    output = args.output

    outfile = open(output, 'w')

    # regex to filter out #include statements, since those can't work
    # for RTC.  The runtime ensures that all the really important
    # includes are already done for us.
    include_regex = re.compile(r'''^\s*#include''')

    # embed files as strings
    outfile.write("#include <array>\n")
    for input in args.embed:
        ident = filename_to_cpp_ident(input)
        outfile.write(f"const char* {ident} {{\n")
        outfile.write('R"_PY_EMBED_(\n')
        with open(input, 'r') as f:
            for line in f:
                if include_regex.match(line):
                    continue
                outfile.write(line)
        outfile.write(')_PY_EMBED_"};\n')

    # hash input files, write sum
    h = hashlib.sha256()
    for input in args.embed + args.logic:
        with open(input, 'rb') as f:
            h.update(f.read())
    outfile.write('const std::array<char,32> generator_sum() { return {')
    for b in h.digest():
        outfile.write("'\\x{:02x}', ".format(b))
    outfile.write('};}\n')
