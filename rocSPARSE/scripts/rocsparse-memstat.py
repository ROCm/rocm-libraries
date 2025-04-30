#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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


import argparse
import json

##
def export_csv(obasename, delim, legend, results, verbose = False,debug = False):
    if verbose:
        print('//rocsparse-memstat  - output file : \'' + obasename + '\'')
    out = open(obasename, "w+")
    for i in range(len(results)):
        for j in range(len(legend)):
            field = results[i][legend[j]]
            if (j>0):
                if (legend[j]=="tag"):
                    out.write(delim+"\""+field+"\"")
                else:
                    out.write(delim+field)
            else:
                if (legend[j]=="tag"):
                    out.write("\""+field+"\"")
                else:
                    out.write(field)
        out.write('\n')
    out.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workingdir',     required=False, default = './')
    parser.add_argument('-o', '--obasename',    required=False, default = 'a.csv')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    if len(unknown_args) > 1:
        print('expecting only one input file.')
    with open(unknown_args[0],"r") as f:
        case=json.load(f)

    results = case['results']
    legend =  case['legend']
    leaks=case['leaks']
    if (len(leaks)==0):
        print('//rocsparse-memstat state: clean.')
    else:
        print('//rocsparse-memstat state: unclean, leaking:')
        for j in range(len(leaks)):
            print(f"mode: {leaks[j]['mode']}",end="")
            print(f", size: {leaks[j]['nbytes']} bytes",end="")
            print(f", location: {leaks[j]['tag']}")
    if verbose:
        print('//rocsparse-memstat  - input file :  \'' + unknown_args[0] + '\'')

    export_csv( obasename,', ', legend, results, verbose,debug)

if __name__ == "__main__":
    main()

