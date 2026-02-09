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
"""A few small utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from functools import reduce

import sys

import re
from collections import defaultdict

#
# Join shortcuts
#


def join(sep, s):
    """Return 's' joined with 'sep'.  Coerces to str."""
    return sep.join(str(x) for x in list(s))


def sjoin(s):
    """Return 's' joined with spaces."""
    return join(' ', [str(x) for x in s])


def njoin(s):
    """Return 's' joined with newlines."""
    return join('\n', s)


def cjoin(s):
    """Return 's' joined with commas."""
    return join(',', s)


def tjoin(s):
    """Return 's' joined with tabs."""
    return join('\t', s)


#
# Misc
#


def shape(n, nbatch):
    """Return NumPy shape."""
    if isinstance(n, (list, tuple)):
        return [nbatch] + list(n)
    return [nbatch, n]


def product(xs):
    """Return product of factors."""
    return reduce(lambda x, y: x * y, xs, 1)


def flatten(xs):
    """Flatten list of lists to a list."""
    return sum(xs, [])


def write_tsv(path, records, meta={}, overwrite=False):
    """Write tab separated file."""
    path = Path(path)
    dat = []
    with open(path, 'a') as f:
        if overwrite:
            f.truncate(0)
        if f.tell() == 0:
            if meta is not None:
                for k, v in meta.items():
                    dat.append(f'# {k}: {v}')
        dat += [tjoin([str(x) for x in r]) for r in records]
        f.write(njoin(dat))
        f.write('\n')


def write_csv(path, records, meta={}, overwrite=False):
    """Write commas separated file."""
    path = Path(path)
    dat = []
    with open(path, 'a') as f:
        if overwrite:
            f.truncate(0)
            if meta is not None:
                for k, v in meta.items():
                    dat.append(f'# {k}: {v}')
        dat += [cjoin([str(x) for x in r]) for r in records]
        f.write(njoin(dat))
        f.write('\n')


# Find the number of matching test tokens.
def find_ncompare(runs):

    import perflib.utils

    outdirs = [Path(outdir) for outdir in runs]
    ncompare = 0
    if len(outdirs) == 2:
        refdir, testdir = outdirs
        all_runs = perflib.utils.read_runs(outdirs)
        runs = perflib.utils.by_dat(all_runs)
        for dat_name, dat_runs in runs.items():
            if (refdir in dat_runs.keys() and testdir in dat_runs.keys()):
                refdat = dat_runs[refdir]
                testdat = dat_runs[testdir]
                for token, sample in refdat.get_samples():
                    if token not in testdat.samples:
                        continue
                    ncompare += 1
    return ncompare


def find_geomean(outdirs, verbose):
    import perflib.utils
    all_runs = perflib.utils.read_runs(outdirs, verbose)
    if len(all_runs) != 2:
        return None
    runs = perflib.utils.by_dat(all_runs)
    refdir, testdir = outdirs

    ratios = []

    import statistics
    from dataclasses import dataclass

    for dat_name, dat_runs in runs.items():
        if (refdir in dat_runs.keys() and testdir in dat_runs.keys()):
            refdat = dat_runs[refdir]
            testdat = dat_runs[testdir]
            for token, sample in refdat.get_samples():
                if token in testdat.samples:
                    Avals = refdat.samples[token].times
                    Bvals = testdat.samples[token].times
                    ratios.append(
                        statistics.median(Avals) / statistics.median(Bvals))

    import scipy
    return scipy.stats.mstats.gmean(ratios)


def find_slower_faster(outdirs, method, multitest, significance, ncompare,
                       verbose):
    # Takes exactly two outdirs; the first is the reference, the
    # second is the values to be compared.

    import perflib.utils

    import statistics

    slower = []
    faster = []

    all_runs = perflib.utils.read_runs(outdirs, verbose)
    if len(all_runs) != 2:
        return slower, faster, significance

    import numpy as np
    import scipy

    token_p_measures = []

    new_significance = significance

    runs = perflib.utils.by_dat(all_runs)
    refdir, testdir = outdirs

    from dataclasses import dataclass

    @dataclass
    class tokendata:
        token: str
        pval: float
        measure_a: float
        measure_b: float

    for dat_name, dat_runs in runs.items():
        if (refdir in dat_runs.keys() and testdir in dat_runs.keys()):
            refdat = dat_runs[refdir]
            testdat = dat_runs[testdir]
            for token, sample in refdat.get_samples():
                if token in testdat.samples:
                    Avals = refdat.samples[token].times
                    Bvals = testdat.samples[token].times

                    pval = None
                    measure_a = None
                    measure_b = None
                    if method == 'moods':
                        _, pval, _, _ = scipy.stats.median_test(Avals, Bvals)
                        measure_a = statistics.median(Avals)
                        measure_b = statistics.median(Bvals)

                    elif method == 'ttest':
                        _, pval = scipy.stats.ttest_ind(Avals, Bvals)
                        measure_a = np.mean(Avals)
                        measure_b = np.mean(Bvals)
                    elif method == 'mwu':
                        _, pval = scipy.stats.mannwhitneyu(Avals, Bvals)
                        measure_a = statistics.median(Avals)
                        measure_b = statistics.median(Bvals)
                    else:
                        print("unsupported statistical method")
                        sys.exit(1)

                    thistokendata = tokendata(token, pval, measure_a,
                                              measure_b)
                    dats = [token, pval, measure_a, measure_b]

                    token_p_measures.append(thistokendata)

    if multitest == "bonferroni" and ncompare > 0:
        new_significance /= ncompare
    if multitest == "bh":
        pvals = []
        for stuff in token_p_measures:
            pvals.append(stuff.pval)

        pvals.sort()

        #print(pvals)

        new_significance = None

        # Find the largest index
        for idx, pval in enumerate(pvals):
            j_alpha = (idx + 1) * significance / ncompare
            if pval < j_alpha:
                new_significance = pval

        # if new_significance == None:
        #     print("Warning: didn't find cutoff alpha for bh multi-hypothesis testing")
        #     new_significance = significance

    # Now that we have the new significance, decide on cases.
    for dat in token_p_measures:
        if dat.pval < new_significance:
            #print(measure_a, measure_b)
            if dat.measure_a > dat.measure_b:
                faster.append([dat.token, dat.measure_a, dat.measure_b])
            else:
                #print(dat.token, dat.measure_a, dat.measure_b)
                slower.append([dat.token, dat.measure_a, dat.measure_b])

    return slower, faster, new_significance


# Decomposition types for multi-process rocFFT
SPLIT_TYPES = {
    (True, True, False): "SLOW_INOUT",
    (True, False, False): "SLOW_IN",
    (False, True, False): "SLOW_OUT",
    (True, False, True): "SLOW_IN_FAST_OUT",
    (True, True, True): "PENCIL_3D"
}


def get_split_dims(bricks):
    coords = list(zip(*(b['lower'] for b in bricks)))
    return [len(set(dim_vals)) > 1 for dim_vals in coords]  # X, Y, Z


def get_decomposition_type(bricks, label="unknown"):
    if not bricks:
        return "UNKNOWN"

    split_x, split_y, split_z = get_split_dims(bricks)

    # Handle full 3D block case
    if split_x and split_y and split_z:
        return "PENCIL_3D"

    if label == "ifield":
        key = (split_x, False, False)
    elif label == "ofield":
        key = (False, split_z, split_x)
    else:
        return "UNKNOWN"

    return SPLIT_TYPES.get(key, "UNKNOWN")


def get_proc_grid(bricks):
    if not bricks:
        return (1, 1, 1)
    lowers = [b['lower'] for b in bricks]
    grid_dims = []
    for i in range(3):  # X, Y, Z
        unique_coords = sorted(set(coord[i] for coord in lowers))
        grid_dims.append(len(unique_coords))
    return tuple(grid_dims)


def print_mgpu_data_layout(bricks):
    print("Layout per device/rank:")
    for b in bricks:
        rank_str = f"{b['rank']}" if b['rank'] is not None else "N/A"
        print(
            f"  Rank {rank_str:<3} | Dev {b['dev']:<2} | Lower {b['lower']} Upper {b['upper']}"
        )


#
# DAT files
#


@dataclass
class Sample:
    """Dyna-bench/bench timing sample: list of times for a given token.

    This corresponds to a single line of a dat file.
    """

    token: str
    times: List[float]
    label: str = None

    def __post_init__(self):
        self.label = self.token


@dataclass
class DAT:
    """Dyna-bench/bench DAT.

    This corresponds to a single .dat file.
    """

    tag: str
    path: Path
    samples: Dict[str, Sample]
    meta: Dict[str, str]

    def get_samples(self):
        keys = self.samples.keys()
        for key in keys:
            yield key, self.samples[key]

    def print(self):
        print("tag:", self.tag)
        print("path:", self.path)
        print("meta:", self.meta)
        print("samples:", self.samples)


@dataclass
class Run:
    """Dyna-bench/bench runs.

    This corresponds to a directory of .dat files.
    """

    title: str
    path: Path
    dats: Dict[Path, DAT]


def write_dat(fname, token, seconds, meta={}):
    """Append record to dyna-bench/bench .dat file."""
    record = [token, len(seconds)] + seconds
    write_tsv(fname, [record], meta=meta, overwrite=False)


def parse_token(token):
    words = token.split("_")

    precision = None
    length = []
    transform_type = None
    batch = None
    placeness = None

    if words[0] not in {"complex", "real"}:
        print("Error parsing token:", token)
        sys.exit(1)
    if words[1] not in {"forward", "inverse"}:
        print("Error parsing token:", token)
        sys.exit(1)
    transform_type = ("forward" if words[1] == "forward" else
                      "backward") + "_" + words[0]

    for idx in range(len(words)):
        if words[idx] == "len":
            lenidx = idx
            break

    for idx in range(lenidx + 1, len(words)):
        if words[idx].isnumeric():
            length.append(int(words[idx]))
        else:
            # Now we have the precision and placeness
            precision = words[idx]
            placeness = "out-of-place" if words[idx +
                                                1] == "op" else "in-place"
            break

    batchidx = -1
    for idx in range(len(words)):
        if words[idx] == "batch":
            batchidx = idx
            break
    batch = []
    for idx in range(batchidx + 1, len(words)):
        if words[idx].isnumeric():
            batch.append(int(words[idx]))
        else:
            break

    bricks = defaultdict(list)
    ranks = set()
    gpus = set()
    current_field = None

    # Regex to get brick indices
    pattern = r"(?:(ifield|ofield)_)?brick_lower_([0-9_]+)_upper_([0-9_]+)_stride_[0-9_]+_(?:rank_(\d+)_)?dev_(\d+)"
    matches = re.findall(pattern, token)

    for field_type, lower_str, upper_str, rank_str, dev_str in matches:
        if field_type:
            current_field = field_type

        lower = tuple(map(int, lower_str.split('_')))
        upper = tuple(map(int, upper_str.split('_')))
        dev = int(dev_str)
        rank = int(rank_str) if rank_str else None

        gpus.add(dev)
        if rank is not None:
            ranks.add(rank)

        bricks[current_field].append({
            'dev': dev,
            'rank': rank,
            'lower': lower[1:],  # skip batch dimension
            'upper': upper[1:]
        })

    return transform_type, placeness, length, batch, precision, bricks, sorted(
        gpus), sorted(ranks)


def read_dat(fname):
    """Read dyna-bench/bench .dat file."""
    path = Path(fname)
    records, meta = {}, {}
    for line in path.read_text().splitlines():
        if line.startswith('# '):
            k, v = [x.strip() for x in line[2:].split(':', 1)]
            meta[k] = v
            continue
        words = line.split("\t")
        token = words[0]
        times = list(map(float, words[2:]))
        records[token] = Sample(token, times)
    tag = meta['title'].replace(' ', '_')
    return DAT(tag, path, records, meta)


def read_run(dname, verbose=False):
    """Read all .dat files in a directory."""
    path = Path(dname)
    if verbose:
        print("reading", path)
    dats = {}
    for dat in list_runs(dname):
        dats[dat.stem] = read_dat(dat)
    return Run(path.stem, path, dats)


def list_runs(dname):
    """List all .dat files in a directory."""
    path = Path(dname)
    return sorted(list(path.glob('*.dat')))


def read_runs(dnames, verbose=False):
    """Read all .dat files in directories."""
    return [read_run(dname, verbose) for dname in dnames]


def get_post_processed(dname, docdir, outdirs, ngroup):
    """Return file names of post-processed performance data.

    The 'primary' files contain median confidence intervals for each
    DAT file.

    The 'secondary' files contain XXX.
    """
    primary = []
    for outdir in outdirs:
        path = (Path(outdir) / dname).with_suffix('.mdat')
        if path.exists():
            primary.append(path)

    import os

    secondary = []
    if ngroup == None:
        totalgroups = 1
    else:
        totalgroups = (len(outdirs) + ngroup - 1) // ngroup

    # FIXME: ngroup needs to be here as well.
    for gidx in range(totalgroups):
        sdatname = 'group_' + str(gidx) + "-" + dname + ".sdat"
        path = os.path.join(docdir, sdatname)
        if os.path.isfile(path):
            secondary.append(path)

    return primary, secondary


def by_dat(runs):
    r = {}
    for dat in runs[0].dats.values():
        dstem = dat.path.stem
        r[dstem] = {
            run.path: run.dats[dstem]
            for run in runs if dstem in run.dats
        }
    return r


def to_data_frames(primaries, secondaries):
    import pandas
    data_frames = []
    for primary in primaries:
        df = pandas.read_csv(primary, delimiter='\t', comment='#')
        data_frames.append(df)

    for i, secondary in enumerate(secondaries):
        df = pandas.read_csv(secondary, delimiter='\t', comment='#')
        data_frames[i + 1] = data_frames[i + 1].merge(df,
                                                      how='left',
                                                      on='token',
                                                      suffixes=('', '_y'))

    return data_frames


def write_pts_dat(fname, records, meta={}):
    """Write data to *.ptsdat"""
    write_csv(fname, records, meta=meta, overwrite=True)
