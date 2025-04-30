# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

import sys, json

import argparse

def replace_string(line, str1, str2):
    index0 = line.find(str1)
    index1 = line[index0+len(str1):].find(' ')

    if index0 == -1:
        return line

    if index1 == -1:
        line = line[:index0] + str2
    else:
        index1 += index0 + len(str1)
        line = line[:index0] + str2 + ' ' + line[index1:]
    return line

def modify(line):
    # remove --bench-x
    line = line.replace('--bench-x ', '')

    # remove --bench-n X
    line = replace_string(line, '--bench-n ', '')

    # replace --rocalution X to --rocalution filename_string
    line = replace_string(line, '--rocalution ', '--rocalution $filename')

    # replace --blockdim X to --blockdim $blockdim
    line = replace_string(line, '--blockdim ', '--blockdim $blockdim')

    # replace --sizen X to --sizen $sizen
    line = replace_string(line, '--sizen ', '--sizen $sizen')

    return line

def main():
    parser = argparse.ArgumentParser("readConfig")
    parser.add_argument('filename')
    args = parser.parse_args()

    print(modify(json.load(open(args.filename))['cmdlines'][0]))

main()
