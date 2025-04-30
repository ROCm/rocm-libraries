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
"""Performance analysis routines."""

import random

import numpy as np
import statistics

from perflib.utils import Run
from dataclasses import dataclass
from typing import List


def confidence_interval(vals, measure, confidence, alpha=0.95, nboot=2000):
    """Compute the alpha-confidence interval for the given values using boot-strap resampling."""
    if confidence == "bootstrap":
        medians = []
        for iboot in range(nboot):
            resample = []
            for i in range(len(vals)):
                resample.append(vals[random.randrange(len(vals))])
            if measure == "median":
                medians.append(np.median(resample))
            elif measure == "mean":
                medians.append(np.mean(resample))
        medians = sorted(medians)
        low = medians[int(np.floor(nboot * 0.5 * (1.0 - alpha)))]
        high = medians[int(np.ceil(nboot * (1.0 - 0.5 * (1.0 - alpha))))]
    elif confidence == "stdev":
        mean = np.mean(vals)
        stdev = np.mean(vals)
        # NB: assumes alpha \approx 95
        low = mean - 2 * stdev
        high = mean + 2 * stdev
    else:
        print("invalid value for confidence:", confidence)
        import sys
        sys.exit(1)
    return low, high


def ratio_confidence_interval(Avals, Bvals, alpha=0.95, nboot=2000):
    """Compute the alpha-confidence interval for the ratio of the given sets of values using boot-strap resampling."""
    ratios = []
    for i in range(nboot):
        ratios.append(Avals[random.randrange(len(Avals))] /
                      Bvals[random.randrange(len(Bvals))])
    ratios = sorted(ratios)
    low = ratios[int(np.floor(len(ratios) * 0.5 * (1.0 - alpha)))]
    high = ratios[int(np.ceil(len(ratios) * (1.0 - 0.5 * (1.0 - alpha))))]
    return low, high


@dataclass
class MoodsResult:
    pval: float
    medians: List[float]


def moods(reference: Run, others: List[Run]):
    """Perform Moods analysis..."""
    import scipy.stats
    import numpy
    pvals = {}
    for rname, rdat in reference.dats.items():
        for other in others:
            odat = other.dats[rname]
            for length in rdat.samples.keys():
                s1 = rdat.samples[length].times
                s2 = odat.samples[length].times

                if arguments.method == 'median':
                    m1 = statistics.median(s1)
                    m2 = statistics.median(s2)
                elif arguments.mesaure == "mean":
                    m1 = numpy.mean(s1)
                    m2 = numpy.mean(s2)

                if arguments.method == 'moods':
                    _, pval, _, _ = scipy.stats.median_test(s1, s2)
                elif arguments.method == 'ttest':
                    _, pval = scipy.stats.ttest_ind(s1, s2)
                elif arguments.method == 'mwu':
                    _, pval = scipy.stats.mannwhitneyu(s1, s2)
                else:
                    print("unsupported statistical method")
                    sys.exit(1)

                pvals[other.path.name, rname,
                      length] = MoodsResult(pval, [m1, m2])
    return pvals
