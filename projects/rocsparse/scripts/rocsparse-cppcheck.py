#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
import os
import re
import subprocess

#
#
# This script filters the .json input file and run cpp-check.
# It is expected to use a version of cpp-check offering googleconfig.cfg.
#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ifilename',     required=False, default = 'compile_commands.json',
                        help = '.json file format is required.')
    parser.add_argument('-o', '--ofilename',     required=False, default = 'a.xml',
                        help = 'output file name.')
    parser.add_argument('-e', '--exclude',     required=False, default = '',
                        help = 'Regular expression to Exclude files whom names match with')
    parser.add_argument('-i', '--include',     required=False, default = '',
                        help = 'Regular expression to Include files whom names match with')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = 'store_true', help = 'enable verbose')

    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    exclude=user_args.exclude
    include=user_args.include
    ifilename=user_args.ifilename
    ofilename=user_args.ofilename

    ifilesplit = os.path.splitext(os.path.basename(ifilename))
    ofilesplit = os.path.splitext(os.path.basename(ofilename))

    ibasename=ifilesplit[0]
    iextension=ifilesplit[1]
    obasename=ofilesplit[0]
    oextension=ofilesplit[1]

    if verbose:
        print('// rocsparse-cppcheck.verbose:file \'' + ifilename + '\'')
    if len(unknown_args) > 0:
        print('// rocsparse-cppcheck.verbose:unexpected arguments ',end='')
        print(unknown_args)
        exit(1)
    with open(ifilename,'r') as f:
        compile_commands=json.load(f)

    res_inc=[ i for i in compile_commands if re.search(include,i['file']) != None]
    if exclude != '':
        compile_commands=[ i for i in res_inc if re.search(exclude,i['file']) == None]
    else:
        compile_commands = res_inc

    if verbose:
        print('// rocsparse-cppcheck.verbose:num commands ' + str(len(compile_commands)))

    tempfilename=ibasename + '.tmp.json'
    with open(tempfilename,'w') as f:
        json.dump(compile_commands,f)

    if verbose:
        print('// rocsparse-cppcheck.verbose:' + ofilename)
    cmdcppcheck = ['cppcheck',
                   '--enable=all',
                   '--project='+ tempfilename,
                   '--output-file='+ ofilename,
                   '--library=googletest.cfg',
                   '--suppress=missingIncludeSystem',
                   '--suppress=unusedFunction',
                   '--suppress=cstyleCast',
                   '--suppress=constParameter',
                   '--suppress=templateRecursion',
    #              '--suppress=invalidPointerCast'
                   '--suppress=unmatchedSuppression']

    if oextension == '.xml':
        cmdcppcheck.append('--xml')
    proc = subprocess.Popen(cmdcppcheck)
    proc.wait()
    rc = proc.returncode
    if rc != 0:
        print('// cppcheck failed')
        exit(1)

    if not verbose:
        os.remove(tempfilename)

if __name__ == "__main__":
    main()

