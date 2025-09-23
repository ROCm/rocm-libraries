# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
"""Bench launch utils."""

import os
import logging
import pathlib
import asyncio
import sys
from collections import defaultdict
from typing import Dict
from asyncio.subprocess import PIPE, STDOUT

#####################################
# for hipblaslt-bench, can use --yaml
#####################################
def run_bench(benchExecutable,
              probYamlFolder,
              benchType,
              argsDict:Dict[str, str],
              verbose=False,
              timeout=300):
    """Run bench"""
    cmd = [pathlib.Path(benchExecutable).resolve()]

    for argKey, argValue in argsDict.items():
        if len(argValue) != 0:
            if argKey == "--yaml":
                argValue = pathlib.Path(os.path.join(probYamlFolder, argValue)).resolve()
            cmd += [argKey, argValue]
        else:
            cmd += [argKey]

    cmd = [str(x) for x in cmd]
    logging.info('hipblaslt-perf: ' + ' '.join(cmd))
    if verbose:
        print('hipblaslt-perf: ' + ' '.join(cmd))

    startingToken = "["
    csvKeys = ''
    benchResultsList = []
    capturingValues = False
    isAPIOverhead = False

    async def run_command(*args, benchType=benchType, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal startingToken
        nonlocal csvKeys
        nonlocal benchResultsList
        nonlocal capturingValues
        nonlocal isAPIOverhead

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(),
                                              timeout)
            except asyncio.TimeoutError:
                logging.info(
                    "timeout expired. killed. Please check the process.")
                print("timeout expired. killed. Please check the process.")
                process.kill()  # Timeout or some criterion is not satisfied
                break

            if not line:
                break
            else:
                line = line.decode('utf-8').rstrip('\n')
                line = line.strip()
                if capturingValues:
                    if not line.startswith(benchType): # filter out some irrelative msg
                        # print('irrelative msg:',line)
                        continue
                    print(line)
                    dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
                    benchResultsList += [dd_output]
                    # if is doing api-overhead, the return log will contain serveral values lines
                    capturingValues = True if isAPIOverhead else False
                elif line.startswith(startingToken):
                    line = line.replace('hipblaslt-Gflops', 'gflops')
                    line = line.replace('hipblaslt-GB/s', 'GB/s')
                    splitLine = line.split(':')
                    funcType = splitLine[0]
                    keys = splitLine[1]
                    print(f'\n{keys}')
                    csvKeys = keys.split(',')
                    capturingValues = True
                    isAPIOverhead = (funcType == '[overhead]')
        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=timeout))
    success = returncode == 0

    loop.close()

    return csvKeys, benchResultsList, success


#####################################
# For ./hipblaslt-perf --run_sh
#####################################
def run_sh_cmd(cmdLine,
               verbose=False,
               timeout=300):
    """Run single bench from sh"""

    cmd = cmdLine.split(' ')
    cmd = [str(x) for x in cmd]
    logging.info('running: ' + ' '.join(cmd))
    if verbose:
        print('running: ' + ' '.join(cmd))

    startingToken = "["
    solNameToken = "--Solution name:"
    csvKeys = []
    benchResultsList = {}
    capturingValues = False

    async def run_command(*args, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal startingToken
        nonlocal csvKeys
        nonlocal benchResultsList
        nonlocal capturingValues

        singleValuesList = []
        solutionName = "N/A"

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(),
                                              timeout)
            except asyncio.TimeoutError:
                logging.info(
                    "timeout expired. killed. Please check the process.")
                print("timeout expired. killed. Please check the process.")
                process.kill()  # Timeout or some criterion is not satisfied
                break

            if not line:
                break
            else:
                line = line.decode('utf-8').rstrip('\n')
                line = line.strip()

                # capturing values right after capturing keys
                if capturingValues:
                    singleValuesList = line.split(',')
                    # default is empty if --print_kernel_info is not in the bench cmd
                    solutionName = "N/A"
                    capturingValues = False
                    continue

                if line.startswith(startingToken):
                    line = line.replace('hipblaslt-Gflops', 'gflops')
                    line = line.replace('hipblaslt-GB/s', 'GB/s')
                    splitLine = line.split(':')
                    # SSN = splitLine[0] # should be [0]
                    keys = splitLine[1] + str(",solution-name")
                    csvKeys = keys.split(',')
                    # print(f'\n{keys}')
                    # print(f'\n{csvKeys}')
                    capturingValues = True # Next line must be values
                else:
                    # if is "--Solution name:" (--print_kernel_info), then we capture this
                    if line.startswith(solNameToken):
                        splitLine = line.split(':')
                        solutionName = splitLine[1].strip()
                    # simply ignore irrelative msg
                    else:
                        continue

        singleValuesList.append(solutionName)
        benchResultsList = defaultdict(str, zip(csvKeys, singleValuesList))

        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=timeout))
    success = returncode == 0

    loop.close()

    return csvKeys, benchResultsList, success
