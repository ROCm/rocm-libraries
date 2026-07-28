################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import functools
import math
import os
import sys
import time
import re

from inspect import currentframe, getframeinfo
from copy import deepcopy
from enum import Enum
from math import log
from pathlib import Path
from typing import Sequence, Tuple, Optional

from Tensile import __version__

from rocisa import rocIsa

import pickle

def fastdeepcopy(x):
    # Note: Some object can't be pickled
    return pickle.loads(pickle.dumps(x))

def isSubtileMultiDU(kernel) -> bool:
    """True when a subtile kernel runs in multi-DU mode.

    Multi-DU means a data tensor's per-uid DepthU (_DepthUA/_DepthUB) is
    smaller than the loop DepthU, i.e. the unroll is split into sub-iterations
    (currently the MXFP8 swizzle path). Single helper so the detection is not
    re-derived inline across the codegen (AsmStoreState, GlobalWriteBatch,
    KernelWriterAssembly).
    """
    du = kernel["DepthU"]
    return kernel.get("_DepthUA", du) < du or kernel.get("_DepthUB", du) < du

# Global
_global_ti = rocIsa.getInstance()

_verbosity = 1

def setVerbosity(v: int):
    global _verbosity
    _verbosity = v

def getVerbosity():
    return _verbosity

################################################################################
# Printing
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints
################################################################################
def print1(message):
    if getVerbosity() >= 1:
        print(message)
        sys.stdout.flush()


def print2(message):
    if getVerbosity() >= 2:
        print(message)
        sys.stdout.flush()


def printWarning(message):
    print("Tensile::WARNING: %s" % message)
    sys.stdout.flush()


def printExit(message):
    print("Tensile::FATAL: %s" % message)
    sys.stdout.flush()
    sys.exit(-1)

# get param values from structures.
def hasParam(name, structure):
    if isinstance(structure, list):
        for l in structure:
            if hasParam(name, l):
                return True
        return False
    elif isinstance(structure, dict):
        return name in structure
    else:
        return name == structure


def isExe(filePath):
    return os.path.isfile(filePath) and os.access(filePath, os.X_OK)


def locateExe(defaultPath, exeName):  # /opt/rocm/bin, hip-clang
    # look in defaultPath first
    if defaultPath:
        exePath = os.path.join(defaultPath, exeName)
        if isExe(exePath):
            return exePath
    # look in PATH second
    for path in os.environ["PATH"].split(os.pathsep):
        exePath = os.path.join(path, exeName)
        if isExe(exePath):
            return exePath

    raise OSError(f"Failed to locate {exeName}")


def ensurePath(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError:
        raise OSError('Failed to create directory "%s" ' % (path))
    return path


def roundUp(f):
    return (int)(math.ceil(f))


def elineno():
    """
    Return the file name and line number of the caller.
    """
    frame = getframeinfo(currentframe().f_back)
    return f"{Path(frame.filename).name}:{frame.lineno}"


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
    (qMajor, qMinor, qStep) = queryVersionString.split(".")
    (tMajor, tMinor, tStep) = __version__.split(".")

    # major version must match exactly
    if qMajor != tMajor:
        return False

    # minor.patch version must be >=
    if int(qMinor) > int(tMinor):
        return False
    if qMinor == tMinor:
        if int(qStep) > int(tStep):
            return False
    return True


################################################################################
# Progress Bar Printing
# prints "||||" up to width
################################################################################
class ProgressBar:
    def __init__(self, maxValue, width=80):
        self.char = "|"
        self.maxValue = maxValue
        self.width = width
        self.maxTicks = self.width - 7

        self.priorValue = 0
        self.fraction = 0
        self.numTicks = 0
        self.createTime = time.time()

    def increment(self, value=1):
        self.update(self.priorValue + value)

    def update(self, value):
        currentFraction = 1.0 * value / self.maxValue
        currentNumTicks = int(currentFraction * self.maxTicks)
        if currentNumTicks > self.numTicks:
            self.numTicks = currentNumTicks
            self.fraction = currentFraction
            self.printStatus()
        self.priorValue = value

    def printStatus(self):
        sys.stdout.write("\r")
        sys.stdout.write(
            "[%-*s] %3d%%" % (self.maxTicks, self.char * self.numTicks, self.fraction * 100)
        )
        if self.numTicks == self.maxTicks:
            stopTime = time.time()
            sys.stdout.write(" (%-.1f secs elapsed)\n" % (stopTime - self.createTime))
        sys.stdout.flush()

    def finish(self):
        pass


class DataDirection(Enum):
    NONE = (0,)
    READ = (1,)
    WRITE = 2


class SpinnyThing:
    def __init__(self):
        self.chars = ["|", "/", "-", "\\"]
        self.index = 0

    def increment(self, value=1):
        sys.stdout.write("\b" + self.chars[self.index])  # pragma: no mutate
        sys.stdout.flush()
        self.index = (self.index + value) % len(self.chars)

    def finish(self):
        sys.stdout.write("\b*\n")
        sys.stdout.flush()


def iterate_progress(obj, *args, **kwargs):
    try:
        progress = ProgressBar(len(obj))
    except TypeError:
        progress = SpinnyThing()
    for o in obj:
        yield o
        progress.increment()
    progress.finish()


try:
    from tqdm import tqdm
except ImportError:
    tqdm = iterate_progress


def state(obj):
    if hasattr(obj, "state"):
        return obj.state()

    if hasattr(obj.__class__, "StateKeys"):
        rv = {}
        for key in obj.__class__.StateKeys:
            attr = key
            if isinstance(key, tuple):
                (key, attr) = key
            rv[key] = state(getattr(obj, attr))
        return rv

    if isinstance(obj, dict):
        return {k: state(v) for k, v in obj.items()}

    if isinstance(obj, (str, int, float)):
        return obj

    try:
        return [state(i) for i in obj]
    except TypeError:
        pass

    return obj


def state_key_ordering(cls):
    def tup(obj):
        return tuple([getattr(obj, k) for k in cls.StateKeys])

    def lt(a, b):
        return tup(a) < tup(b)

    def eq(a, b):
        return tup(a) == tup(b)

    cls.__lt__ = lt
    cls.__eq__ = eq

    return functools.total_ordering(cls)


def hash_combine(*objs, **kwargs):
    shift = 1
    if "shift" in kwargs:
        shift = kwargs["shift"]

    if len(objs) == 1:
        objs = objs[0]

    rv = 0
    try:
        it = iter(objs)
        rv = next(it)
        for value in it:
            rv = (rv << shift) ^ value
    except TypeError:
        return objs
    except StopIteration:
        pass
    return rv


def hash_objs(*objs, **kwargs):
    return hash(tuple(objs))


def ClientExecutionLock(lockPath: str):
    if not lockPath:
        return open(os.devnull)

    import filelock

    return filelock.FileLock(lockPath)


def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, defaultDictionary):
    if key in sourceDictionary:
        destinationDictionary[key] = deepcopy(sourceDictionary[key])
    else:
        destinationDictionary[key] = deepcopy(defaultDictionary[key])


def isRhel8() -> bool:
    """
    Check if the current OS is Red Hat Enterprise Linux 8 by reading the /etc/os-release file.

    Returns:
        True if the current OS is RHEL 8, False otherwise
    """
    file = Path("/etc/os-release")
    pattern = r'NAME="Red Hat Enterprise Linux".*VERSION_ID="8\.\d+"'
    if not file.exists():
        return False
    with open(file, "r") as f:
        content = f.read()
    match = re.search(pattern, content, re.DOTALL)
    if match:
        printWarning("Rhel8 environments may not support all tools for system queries such as amd-smi.")
        return True
    return False

########################################
# Math
########################################

def clusterEnabled(clusterDim):
    """True when a workgroup cluster is requested (ClusterDim [x, y] is not [1, 1])."""
    return (clusterDim[0] * clusterDim[1]) != 1

def streamKClusterFactors(d):
    """Return (Cs, Ck, C, is2D) for a StreamK workgroup cluster (ClusterDim-driven).

    The StreamK cluster is fully described by ClusterDim = [Cs, Ck]; there are no
    user factoring/reduction knobs. Cs = ClusterDim[0] is the spatial B-multicast
    axis (X), Ck = ClusterDim[1] is the K-split reduction axis (Y), and the total
    cluster is C = Cs * Ck. ``is2D`` is True exactly when Ck > 1, i.e. when the
    launch grid must be genuinely 2-D ([skGrid/Ck, Ck, 1]) and the linear StreamK
    index folds the cluster Y rank in (StreamKIdx = WorkGroup0*Ck + WorkGroup1).

    Config expressions:
      * [C, 1] -> Cs=C, Ck=1  : pure multicast   (1-D launch)
      * [1, C] -> Cs=1, Ck=C  : pure reduction    (2-D launch)
      * [Cs,Ck]-> both > 1     : factored          (2-D launch; B-multicast along Cs
                                                    AND K-split reduction along Ck)

    ``d`` may be a kernel or a solution ``state`` dict; both expose "ClusterDim".
    See docs/design/streamk-wg-clusters.md.
    """
    cd = d["ClusterDim"]
    cs, ck = cd[0], cd[1]
    return cs, ck, cs * ck, (ck > 1)

def streamKForceDP2DMulticast(d):
    """True for the ForceDPOnly 2-D dual-operand multicast cluster.

    This is a StreamK==3 ``StreamKForceDPOnly`` (dense data-parallel, no K-split
    reduction) kernel given a GENUINE 2-D cluster ClusterDim = [Cs, Ck] with BOTH
    axes > 1. Unlike the factored K-split cluster (where Ck is a reduction axis),
    here the Ck (Y) axis maps to N-ADJACENT output tiles so the Y-peers reuse the
    A operand (A-multicast), while the Cs (X) peers reuse B on M-adjacent tiles
    exactly as in the shipped 1-D [C,1] ForceDPOnly multicast. Both operands are
    multicast via the DENSE ClusterLoad 2-D masks.

    Detected purely structurally (ForceDPOnly + ClusterDim[0]>1 + ClusterDim[1]>1):
    the K-split reduction interpretation of Ck>1 is rejected for ForceDPOnly
    (_validateStreamKClusterReduction), so this shape is unambiguous and needs no
    extra serialized/derived flag. ``d`` may be a kernel or a solution ``state``
    dict; both expose "StreamKForceDPOnly" and "ClusterDim".
    See docs/design/streamk-wg-clusters.md.
    """
    return bool(d.get("StreamKForceDPOnly", 0)) \
        and d["ClusterDim"][0] > 1 and d["ClusterDim"][1] > 1

def streamKDual2DMulticast(d):
    """True for a 2-D DUAL-operand multicast cluster (generalized detector).

    This GENERALIZES ``streamKForceDP2DMulticast`` to cover BOTH:

      * the ForceDPOnly 2-D dual-operand multicast (``StreamKForceDPOnly`` + 2-D cluster);
        and
      * the STANDARD two-tile StreamK path (``StreamKForceDPOnly == 0``) opted in
        via ``StreamKDualMulticast``. Here the DP (full-tile) round
        does the same 2-D dual multicast (Cs/X peers share B on M-adjacent tiles,
        Ck/Y peers share A on N-adjacent tiles) while the SK (partial-tile) round
        reduces 1-D via the workspace exactly as today. It is temporal reuse of
        ONE physical cluster: a 2-D mask grouping during DP, a 1-D reduction
        grouping during SK.

    Both cases require a GENUINE 2-D cluster ClusterDim = [Cs, Ck] (both axes > 1)
    where Ck (Y) is an N-tiling / A-multicast axis, NOT a K-split reduction axis.
    It is therefore kept DISTINCT from the factored [Cs,Ck] path (where Ck IS the
    K-reduction axis and ``StreamKClusterReduction`` is derived): a factored config
    sets neither ``StreamKForceDPOnly`` nor ``StreamKDualMulticast``, so this
    returns False for it. The opt-in (``StreamKDualMulticast``) is what a
    StreamKForceDPOnly=0 [Cs,Ck] config uses to select dual-2D multicast INSTEAD
    of the factored K-reduction interpretation (mutual exclusion; see the
    Solution.py collapse). ``d`` may be a kernel or a solution ``state`` dict.
    See docs/design/streamk-wg-clusters.md.
    """
    if not (d["ClusterDim"][0] > 1 and d["ClusterDim"][1] > 1):
        return False
    return bool(d.get("StreamKForceDPOnly", 0)) or bool(d.get("StreamKDualMulticast", 0))

def log2(x):
    return int(log(x, 2) + 0.5)

def effectiveMatrixInstMN(matrixInstM, matrixInstN, sourceSwap):
    # Effective per-instruction M/N extents for tiling/layout. SourceSwap on a
    # non-square MatrixInstruction transposes the accumulator, so the M/N tiling
    # extents swap; the physical MatrixInstM/N (opcode / accumulator-layout source
    # of truth) are unchanged. Square MI or SS0 return the inputs unchanged.
    if sourceSwap and matrixInstM != matrixInstN:
        return matrixInstN, matrixInstM
    return matrixInstM, matrixInstN

def ceilDivide(numerator, denominator):
    # import pdb
    # pdb.set_trace()
    try:
        if numerator < 0 or denominator < 0:
            raise ValueError
    except ValueError:
        print("ERROR: Can't have a negative register value")  # pragma: no mutate
        return 0
    try:
        div = int((numerator+denominator-1) // denominator)
    except ZeroDivisionError:
        print("ERROR: Divide by 0")  # pragma: no mutate
        return 0
    return div

def roundUpToNearestMultiple(numerator, denominator):
    return ceilDivide(numerator,denominator)*int(denominator)

# Given a divisor, this routine computes the corresponding multiplicative constant
# and required post shifts.
#
# Algorithm based on: https://dl.acm.org/doi/pdf/10.1145/178243.178249
#
# Inputs:
#   d: divisor
#   N: Number of bits integers are represented in
#   p: precision in bits (usually N = P)
#
# Output:
#   mhigh: multiplicative constant
#   shPost: amount to right shift after multiplication
def choose_multiplier(d, N, p):
    l = int(math.ceil(math.log(d, 2)))
    shPost = l
    mlow = 2**(N+l) // d
    mhigh = (2**(N+l) + 2 ** (N + l - p )) // d
    while ((mlow // 2) < (mhigh // 2)) and shPost > 0:
        mlow //= 2
        mhigh //= 2
        shPost -=1
    return mhigh, shPost, l

def wmmaV3InputVgprLayout(wmma: Sequence[int], dtypeBitWidth: Optional[int] = None) -> Tuple[int]:
    # wmmaV3InputVgprLayout: (numReadsUnroll, numVecTile, numVecUnroll, NumElementPerRead)
    wmma = tuple(wmma)
    if wmma == (16, 16, 4, 1):
        return (1, 16, 2, 2)
    elif wmma == (16, 16, 32, 1):
        return (2, 16, 2, 8)
    elif wmma == (16, 16, 64, 1):
        return (2, 16, 2, 16)
    elif wmma == (16, 16, 128, 1) or wmma == (32, 16, 128, 1):
        assert dtypeBitWidth
        if dtypeBitWidth == 8:
            return (4, 16, 2, 16)
        if dtypeBitWidth == 4 or dtypeBitWidth == 6:
            return (2, 16, 2, 32)
        assert False, f"Unsupported datatype bitwidth: {dtypeBitWidth}"
    else:
        assert False, f"Unhandled WMMA: {wmma}"
