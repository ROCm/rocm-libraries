# Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
"""Timing utilities."""

import collections

import perflib
import logging

from dataclasses import dataclass, field
from pathlib import Path as path
from typing import List
from typing import Set

from numpy import ceil


@dataclass
class Timer:
    bench: str = ""
    accutest: str = ""
    active_tests_tokens: Set[bytes] = field(default_factory=set)
    lib: List[str] = field(default_factory=list)
    out: List[str] = field(default_factory=list)
    device: int = 0
    ntrial: int = 10
    mp_size: int = 1
    mp_exec: str = ""
    ingrid: List[int] = field(default_factory=list)
    outgrid: List[int] = field(default_factory=list)
    ngpus: int = 1
    verbose: bool = False
    timeout: float = 0
    sequence: int = None
    hipskip: bool = True

    def run_cases(self, generator):

        bench = path(self.bench)
        if not bench.is_file():
            raise RuntimeError(f"Unable to find (dyna-)bench: {self.bench}")

        if self.mp_size > 1 and self.mp_exec == None:
            raise RuntimeError(f"Unable to find mpi executable {self.bench}")

        failed_tokens = []

        # get a list of powers of two less or equal than the number of requested resources,
        # to be used for scalability experiments
        gpu_list_pow2 = lambda n: [
            2**i for i in range(int(ceil(n**0.5)) + 1) if 2**i <= n
        ]

        total_prob_count = 0
        no_accutest_prob_count = 0
        for prob in generator.generate_problems():
            total_prob_count += 1

            n_resources = 1
            # scalability for single-proc multi-GPU:
            if self.ngpus > 1 and self.mp_size == 1:
                n_resources = self.ngpus
            # scalability for single-proc using 1-GPU per MPI:
            elif self.mp_size > 1 and self.ngpus == 1:
                n_resources = self.mp_size

            scaling = prob.meta.get('scaling')
            if scaling != None:
                list_of_gpus = gpu_list_pow2(n_resources)
            else:
                list_of_gpus = [n_resources]

            ws_factor = 1

            for g in list_of_gpus:
                token, seconds, success, __, __ = perflib.bench.run(
                    self.bench,
                    tuple([ws_factor * l for l in prob.length]),
                    direction=prob.direction,
                    real=prob.real,
                    inplace=prob.inplace,
                    precision=prob.precision,
                    nbatch=prob.nbatch,
                    mp_size=g if self.mp_size > 1 else 1,
                    mp_exec=self.mp_exec,
                    ingrid=self.ingrid,
                    outgrid=self.outgrid,
                    ngpus=g if self.ngpus > 1 else 1,
                    ntrial=self.ntrial,
                    device=self.device,
                    libraries=self.lib,
                    verbose=self.verbose,
                    timeout=self.timeout,
                    sequence=self.sequence,
                    skiphip=self.hipskip,
                    scalability=(scaling != None))

                if scaling == 'weak':
                    ws_factor *= 2

                if success:
                    for idx, vals in enumerate(seconds):
                        out = path(self.out[idx])
                        logging.info("output: " + str(out))
                        meta = {'title': prob.tag}
                        meta.update(prob.meta)
                        perflib.utils.write_dat(out, token, seconds[idx], meta)
                else:
                    failed_tokens.append(token)

                if self.active_tests_tokens and token.encode(
                ) not in self.active_tests_tokens:
                    no_accutest_prob_count += 1
                    logging.info(f'No accuracy test coverage for: ' + token)

        if no_accutest_prob_count > 0:
            print('\t')
            logging.warning(
                str(no_accutest_prob_count) + f' out of ' +
                str(total_prob_count) +
                f' problems do not have accuracy coverage.' +
                f' Refer to rocfft-perf.log for details.')

        return failed_tokens


@dataclass
class GroupedTimer:
    bench: str = ""
    accutest: str = ""
    active_tests_tokens: Set[bytes] = field(default_factory=set)
    lib: List[str] = field(default_factory=list)
    out: List[str] = field(default_factory=list)
    device: int = 0
    ntrial: int = 10
    mp_size: int = 1
    mp_exec: str = ""
    ngpus: int = 1
    verbose: bool = False
    timeout: float = 0

    def run_cases(self, generator):
        failed_tokens = []
        all_problems = collections.defaultdict(list)
        for problem in generator.generate_problems():
            all_problems[problem.tag].append(problem)

        total_problems = sum([len(v) for v in all_problems.values()])
        print(
            f'Timing {total_problems} problems in {len(all_problems)} groups')

        if self.accutest:
            accutest = path(self.accutest)
            if not accutest.is_file():
                raise RuntimeError(
                    f'Unable to find accuracy test: {self.accutest}')
            self.active_tests_tokens = perflib.accutest.get_active_tests_tokens(
                accutest)

        for i, (tag, problems) in enumerate(all_problems.items()):
            print(
                f'\n{tag} (group {i} of {len(all_problems)}): {len(problems)} problems'
            )
            timer = Timer(**self.__dict__)
            timer.out = [path(x) / (tag + '.dat') for x in self.out]
            failed_tokens += timer.run_cases(
                perflib.generators.VerbatimGenerator(problems))
        return failed_tokens
