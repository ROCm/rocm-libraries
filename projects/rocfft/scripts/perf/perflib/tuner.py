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
"""Bench launch utils."""

import logging
import pathlib
import re
import subprocess
import time
from perflib.utils import sjoin

import asyncio
import sys
from asyncio.subprocess import PIPE, STDOUT


def run(tuner,
        length,
        direction=-1,
        real=False,
        inplace=True,
        precision='single',
        nbatch=1,
        ntrial=1,
        device=None,
        verbose=False,
        timeout=10):
    """Run rocFFT tuner and return best solution"""
    cmd = [pathlib.Path(tuner).resolve()]
    cmd += ['tune']

    if isinstance(length, int):
        cmd += ['--length', length]
    else:
        cmd += ['--length'] + [sjoin([str(len) for len in length])]

    cmd += ['-N', ntrial]
    cmd += ['-b', nbatch]
    if not inplace:
        cmd += ['-o']
    if precision == 'half':
        cmd += ['--precision', 'half']
    elif precision == 'single':
        cmd += ['--precision', 'single']
    elif precision == 'double':
        cmd += ['--precision', 'double']
    if device is not None:
        cmd += ['--device', device]

    if real:
        if direction == -1:
            cmd += ['-t', 2, '--itype', 2, '--otype', 3]
        if direction == 1:
            cmd += ['-t', 3, '--itype', 3, '--otype', 2]
    else:
        if direction == -1:
            cmd += ['-t', 0]
        if direction == 1:
            cmd += ['-t', 1]

    cmd = [str(x) for x in cmd]
    logging.info('tunning: ' + ' '.join(cmd))
    if verbose:
        print('tunning: ' + ' '.join(cmd))

    tokenToken = "Token: "
    outFileToken = "[OUTPUT_FILE]: "
    resultToken = "[Result]: "
    token = ""
    outFileName = ""
    msg = "[Solution]:\n"

    async def run_command(*args, timeout=None):

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE)

        nonlocal token
        nonlocal outFileName
        nonlocal msg

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
                print(line)
                if line.startswith(tokenToken):
                    token = line[len(tokenToken):]
                elif line.startswith(outFileToken):
                    outFileName = line[len(outFileToken):]
                elif line.startswith(resultToken):
                    msg += line[len(resultToken):] + '\n'
        return await process.wait()  # Wait for the child process to exit

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()

    returncode = loop.run_until_complete(run_command(*cmd, timeout=10))
    success = returncode == 0

    loop.close()

    return token, outFileName, msg, success


def accuracy_test(validator,
                  length,
                  direction=-1,
                  real=False,
                  inplace=True,
                  precision='single',
                  nbatch=1,
                  token=None,
                  timeout=10):
    """Run rocFFT test."""
    cmd = [pathlib.Path(validator).resolve()]

    cmd += ['--gtest_filter=man*']

    # use token if we have it
    if token != None:
        cmd += ['--token', token]
    # else, specify each arg
    else:
        if isinstance(length, int):
            cmd += ['--length', length]
        else:
            cmd += ['--length'] + list(length)

        cmd += ['-b', nbatch]
        if not inplace:
            cmd += ['-o']
        if precision == 'half':
            cmd += ['--precision', 'half']
        elif precision == 'single':
            cmd += ['--precision', 'single']
        elif precision == 'double':
            cmd += ['--precision', 'double']

        if real:
            if direction == -1:
                cmd += ['-t', 2, '--itype', 2, '--otype', 3]
            if direction == 1:
                cmd += ['-t', 3, '--itype', 3, '--otype', 2]
        else:
            if direction == -1:
                cmd += ['-t', 0]
            if direction == 1:
                cmd += ['-t', 1]

    cmd = [str(x) for x in cmd]
    logging.info('accuracy testing: ' + ' '.join(cmd))
    print('accuracy testing: ' + ' '.join(cmd))

    passToken = "[  PASSED  ] 1 test"
    passed = False

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in proc.stdout:
        line = line.decode('utf-8').rstrip('\n')
        if line.startswith(passToken):
            print(line)
            passed = True

    try:
        proc.wait(timeout=None if timeout == 0 else timeout)
    except subprocess.TimeoutExpired:
        logging.info("timeout expired. killed. Please check the process.")
        proc.kill()
    success = proc.returncode == 0

    if not success:
        print('[  FAILED  ]: ' + ' '.join(cmd))

    return success


def merge(merger,
          base_file_path,
          new_files,
          new_probTokens,
          out_file_path,
          verbose=False,
          timeout=30):
    """Run rocFFT tuner with command merge"""

    cmd = [pathlib.Path(merger).resolve()]

    cmd += ['--command', '1']
    cmd += ['--new_sol_file', str(new_files)]
    cmd += ['--new_probkey', str(new_probTokens)]
    cmd += ['--output_sol_file', str(out_file_path)]
    if base_file_path is not None:
        cmd += ['--base_sol_file', str(base_file_path)]

    cmd = [str(x) for x in cmd]
    logging.info('merging: ' + ' '.join(cmd))
    if verbose:
        print('merging: ' + ' '.join(cmd))

    # cpp merger simply return code, so no need to capture msg
    # but since the merger has some recursive operation on tree,
    # so using wait is still good to prevent any infinity loop bug..
    proc = subprocess.Popen(cmd)

    try:
        proc.wait(timeout=None if timeout == 0 else timeout)
    except subprocess.TimeoutExpired:
        logging.info("timeout expired. killed. Please check the process.")
        proc.kill()
    success = proc.returncode == 0

    if not success:
        print('Failed on merging:' + ' '.join(cmd))

    return success
