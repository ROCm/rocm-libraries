#!/usr/bin/python

################################################################################
#
# Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function
import sys, argparse, re

verbose = 1

class GemmConfig:
  def __init__(self, m, n ,k, tA, tB):
    self.m = m
    self.n = n
    self.k = k
    self.batchCnt = 1
    self.tA = tA
    self.tB = tB

#---
class TensileExactParser:
  def __init__(self):
    self.globalTA = False
    self.globalTB = False

  def printHeader(self):
    None
    #print "# %s%s Case" % \
    #  ("T" if self.globalTA else "N", "T" if self.globalTB else "N")

  def parse(self,line):
    line = line.lstrip()
    if not line or line.startswith("#"):
      return (0, None)

    #m = re.search(".*Exact\w*:\w\[(.*)\]", line)
    m = re.search(".*Exact.*\[(.*)\]", line)
    try:
      #print "match=", m.group(1)
      nums = [int(x) for x in m.group(1).split(',')]
      #print "nums=", nums
      if len(nums) == 3:
        g = GemmConfig(nums[0], nums[1], nums[2], self.globalTA, self.globalTB)
      elif len(nums) == 4:
        g = GemmConfig(nums[0], nums[1], nums[3], self.globalTA, self.globalTB)
        g.batchCnt = nums[2]
      return (1,g)
    except AttributeError:
      if verbose:
        print("inparser ignored: ", l,)
      return (0, None)

  def emit(self, g):
    # ignores GemmConfig
    print("          - Exact: [ %u , %u , %u, %u ]" % \
        (g.m, g.n, g.batchCnt, g.k))

#---
class DeepBenchGemmParser :
  # m, n, k, a_t, b_t
  # Example: std::make_tuple(4224, 1, 128, false, false)

  def printHeader(self):
    print("// m, n, k, a_t, b_t")

  def parse(self,line):
    line = line.lstrip()
    if not line or line.startswith("//"):
      return (0, None)

    m = re.search("\s*std::make_tuple\((.*)\)", line)
    try:
      #print "match=", m.group(1)
      fields = [x.strip() for x in m.group(1).split(',')]
      assert(len(fields) == 5)
      #print "nums=", nums
      g = GemmConfig(int(fields[0]), int(fields[1]), int(fields[2]), fields[3]=='true', fields[4]=='true')
      return (1,g)
    except AttributeError:
      if verbose:
        sys.stderr.write ("inparser ignored: %s" % l)
      return (0, None)

  def emit(self, g):
    assert (g.batchCnt==1)
    print("std::make_tuple(%u, %u, %u, %s, %s)" % \
      (g.m, g.n, g.k, "true" if g.tA else"false", "true" if g.tB else "false"))

#---
class CsvParser :
  def printHeader(self):
    print("tA, tB, m, n, k, batchCnt")

  def parse(self,line):
    line = line.lstrip()
    if not line :
      return (0, None)

    try:
      fields = [x.strip() for x in line.split(',')]
      g = GemmConfig(int(fields[2]), int(fields[3]), int(fields[4]), fields[0]=='T', fields[1]=='T')
      return (1,g)
    except AttributeError:
      if verbose:
        sys.stderr.write ("inparser ignored: %s" % l)
      return (0, None)

  def emit(self, g):
    print("%s, %s, %u, %u, %u, %u" % \
      ("T" if g.tA else "N", "T" if g.tB else "N", g.m, g.n, g.k, g.batchCnt))



g_formats = {
  "tensile_exact"  : TensileExactParser(),
  "deepbench_gemm" : DeepBenchGemmParser(),
  "csv"            : CsvParser()
  }

parser = argparse.ArgumentParser(description='convert between GEMM formats (Tensile, rocblas-bench, deepbench, CSV')

parser.add_argument('infile', type=argparse.FileType('r'),
        help="Input file to process")

parser.add_argument('-i', '--in-format', action="store",
        default="tensile_exact",
        help="Input format")

parser.add_argument('-o', '--out-format', action="store",
        default="csv",
        help="Output format")

args = parser.parse_args(sys.argv[1:])


#print args

iformat = g_formats[args.in_format]
oformat = g_formats[args.out_format]

oformat.printHeader()

lineNum = 0
for l in args.infile:
    lineNum = lineNum+1
    #print "L:",l,
    (valid, g) = iformat.parse(l)
    if valid:
      oformat.emit(g)
