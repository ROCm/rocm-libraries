# Copyright (C) 2021 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
"""rocfft-bench launch utils."""

import logging
import pathlib
import re
import subprocess
import tempfile
import time


def run(bench,
        length,
        direction=-1,
        real=False,
        inplace=True,
        precision='single',
        nbatch=1,
        ntrial=1,
        mp_exec='/usr/bin/mpirun',
        mp_size=1,
        ingrid=None,
        outgrid=None,
        ngpus=1,
        device=None,
        libraries=None,
        verbose=False,
        timeout=300,
        sequence=None,
        skiphip=True,
        scalability=False):
    """Run rocFFT bench and return execution times."""
    cmd = [pathlib.Path(bench).resolve()]

    if libraries is not None:
        for library in libraries:
            cmd += ['--lib', pathlib.Path(library).resolve()]
        if len(libraries) > 1:
            # only use different randomizations if using dyna-bench
            if sequence is not None:
                cmd += ['--sequence', str(sequence)]

    if mp_size == 1:
        if skiphip:
            cmd += ['--ignore_runtime_failures']
        else:
            cmd += ['--no-ignore_runtime_failures']

    if isinstance(length, int):
        cmd += ['--length', length]
    else:
        cmd += ['--length'] + list(length)

    if len(ingrid) > 0:
        if isinstance(ingrid, int):
            if (mp_size == 1):
                cmd += ['--ingrid', ingrid]
            else:
                cmd += ['--imgrid', ingrid]
        else:
            if (mp_size == 1):
                cmd += ['--ingrid'] + list(ingrid)
            else:
                cmd += ['--imgrid'] + list(ingrid)

    if len(outgrid) > 0:
        if isinstance(outgrid, int):
            if (mp_size == 1):
                cmd += ['--outgrid', outgrid]
            else:
                cmd += ['--omgrid', outgrid]
        else:
            if (mp_size == 1):
                cmd += ['--outgrid'] + list(outgrid)
            else:
                cmd += ['--omgrid'] + list(outgrid)

    if (ngpus > 1):
        cmd += ['--ngpus', ngpus]

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
    if mp_size == 1 and (device is not None):
        cmd += ['--device', device]

    # default to slab decomposition for scalability experiments,
    # which grants the least number of transpositions
    # TODO: extend to further decompositions
    if (scalability):
        if (ngpus > 1):
            if (len(length) == 3):
                cmd += ['--ingrid'] + list([1, 1, ngpus])
                cmd += ['--outgrid'] + list([ngpus, 1, 1])
            elif (len(length) == 2):
                cmd += ['--ingrid'] + list([1, ngpus])
                cmd += ['--outgrid'] + list([ngpus, 1])

        if (mp_size > 1):
            if (len(length) == 3):
                cmd += ['--imgrid'] + list([1, 1, mp_size])
                cmd += ['--omgrid'] + list([mp_size, 1, 1])
            elif (len(length) == 2):
                cmd += ['--imgrid'] + list([1, mp_size])
                cmd += ['--omgrid'] + list([mp_size, 1])

    itype, otype = 0, 0
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

    if verbose:
        cmd += ['--verbose']

    if (mp_size > 1):
        cmd.insert(0, str(mp_size))
        cmd.insert(
            0, "-n"
        )  # flag to set the number of MPI processes for mpirun or equivalent
        cmd.insert(0, mp_exec)
        cmd += ['--benchmark']

    cmd = [str(x) for x in cmd]

    logging.info('running: ' + ' '.join(cmd))
    if verbose:
        print('running: ' + ' '.join(cmd))
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")

    time_start = time.time()
    proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
    try:
        proc.wait(timeout=None if timeout == 0 else timeout)
    except subprocess.TimeoutExpired:
        logging.info("killed")
        proc.kill()
    time_end = time.time()
    logging.info("elapsed time in seconds: " + str(time_end - time_start))

    fout.seek(0)
    ferr.seek(0)
    cout = fout.read()
    cerr = ferr.read()

    logging.debug(cout)
    logging.debug(cerr)

    tokentoken = "Token: "
    token = ""
    times = []

    soltokenTag = "[SolToken]: "
    soltoken = ""
    matchTag = "[TokenMatch]: "
    match = ""

    for line in cout.splitlines():
        if line.startswith(tokentoken):
            token = line[len(tokentoken):]

    for line in cerr.splitlines():
        if line.startswith(soltokenTag):
            soltoken = line[len(soltokenTag):]
        elif line.startswith(matchTag):
            match = line[len(matchTag):]

    if proc.returncode == 0:
        for m in re.finditer(
                r'(?:Max rank time|Execution gpu time):\s*([0-9. ]+)\s*ms',
                cout, re.MULTILINE):
            raw = m.group(1)
            t = [float(x) for x in raw.split() if x.strip()]
            times.append(t)
    else:
        logging.info("PROCESS FAILED with return code " + str(proc.returncode))

    if verbose:
        print('finished: ' + ' '.join(cmd))

    if proc.returncode == 0:
        if "SKIPPED" in cout:
            print('s', end='', flush=True)
        elif "HIP_V_THROWERROR" in cout:
            print('h', end='', flush=True)
            # TODO: print hip runtime failed cases?
        else:
            print('.', end='', flush=True)
    else:
        print('x', end='', flush=True)

    success = proc.returncode == 0

    return token, times, success, soltoken, match
