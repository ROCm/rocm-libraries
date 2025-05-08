################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import csv
import os
from subprocess import Popen, PIPE

################################################################################
# Benchmark Problem Size
################################################################################
def BenchmarkProblemSize(cmdPrefix, row):
  cmd = cmdPrefix
  cmd += " --sizes"
  for size in row:
    cmd += " "
    cmd += size.lstrip()
  sys.stderr.write(cmd)
  sys.stderr.write("\n")
  process = Popen(cmd, stdout=PIPE, shell=True)
  stdout = process.communicate()[0]
  sys.stderr.write(stdout)

  # find beginning of data
  initializingIdx = stdout.find("Initializing")
  newLineIdx = stdout.find("\n", initializingIdx)
  stdout = stdout[newLineIdx+1:]

  gflopList = []
  msList = []

  # skip first b/c warmup/lookup
  newLineIdx = stdout.find("\n")
  stdout = stdout[newLineIdx+1:]

  # parse every line of data
  while "\n" in stdout:
    newLineIdx = stdout.find("\n")
    lineString = stdout[:newLineIdx]
    splits = lineString.split(",")
    if len(splits) < 3:
      break
    gflopsString = splits[1].lstrip()
    msString = splits[3].lstrip()
    gflops = float(gflopsString)
    ms = float(msString)
    gflopList.append(gflops)
    msList.append(ms)

    # next line
    stdout = stdout[newLineIdx+1:]
  return (gflopList, msList)


################################################################################
# Print Stats
################################################################################
def PrintStats(header, row, gflopList, msList):
  meanGFlops = mean(gflopList)
  meanMs = mean(msList)
  medianGFlops = median(gflopList)
  medianMs = median(msList)
  stddevGFlops = stddev(gflopList) / meanGFlops
  stddevMs = stddev(msList) / meanMs

  # format output
  line = ""
  for size in row:
    line += "%6u, " % int(size)
  line += "%9.2f, %9.4f, " % (medianGFlops, medianMs)
  line += "%9.2f, %9.4f, " % (meanGFlops, meanMs)
  line += "%9.5f, %9.5f" % (stddevGFlops, stddevMs)
  sys.stdout.write(line)
  sys.stdout.write("\n")
  sys.stdout.flush()
  sys.stderr.write("[STDOUT] %s\n" % header)
  sys.stderr.write("[STDOUT] %s\n" % line)
  sys.stderr.write("[END]\n\n\n")
  sys.stderr.flush()


################################################################################
# Benchmark Problem Sizes
################################################################################
def TensileBenchmarkLibraryClient(userArgs):
  if len(userArgs) < 2:
    line = "USAGE:   python TensileBenchmarkLibraryClient.py sizes.csv library_client_command > name.txt 2> name.raw.txt \n"
    sys.stderr.write(line)
    line = "Example: python TensileBenchmarkLibraryClient.py sizes.csv ./4_LibraryClient/build/client --function-idx 1 --num-benchmarks 100 --use-gpu-timer 0 > nn.txt 2> nn.raw.txt\n"
    sys.stderr.write(line)
    line = "[BE SURE TO PIN CLOCKS]\n"
    sys.stderr.write(line)
    exit(-1)

  # parse problem sizes path
  problemSizesPath = os.path.realpath( userArgs[0] )
  line = "ProblemSizesPath: %s\n" % problemSizesPath
  sys.stdout.write(line)
  sys.stderr.write(line)

  # parse library client command
  libraryClientCommandList = userArgs[1:]
  libraryClientCommand = " ".join(libraryClientCommandList)
  line = "LibraryClientCommand: %s\n" % libraryClientCommand
  sys.stdout.write(line)
  sys.stdout.write("\n")
  sys.stderr.write(line)
  sys.stderr.write("\n\n")

  # read problem sizes file
  csvFileRaw = open(problemSizesPath, "r")
  csvFile = csv.reader(csvFileRaw)

  # column headers
  numIndices = -1
  for row in csvFile:
    numIndices = len(row)
    firstRow = row
    break
  header = ""
  for i in range(0, numIndices):
    sizeStr = "size%u" % i
    header += "%6s, " % sizeStr
  header += "%9s, %9s, " % ( "medianGF", "medianMs" )
  header += "%9s, %9s, " % ( "meanGF", "meanMs" )
  header += "%9s, %9s" % ( "rstdGF", "rstdMs" )
  sys.stdout.write(header)
  sys.stdout.write("\n")

  # benchmark each problem size
  for row in [firstRow]:
    (gflopList, msList) = BenchmarkProblemSize(libraryClientCommand, row)
    PrintStats(header, row, gflopList, msList)

  # benchmark each problem size
  for row in csvFile:
    (gflopList, msList) = BenchmarkProblemSize(libraryClientCommand, row)
    PrintStats(header, row, gflopList, msList)

def median(lst):
  sortedList = sorted(lst)
  return sortedList[len(sortedList)/2]

def mean(lst):
  total = 0
  for i in range(len(lst)):
    total += lst[i]
  return (total / len(lst))

def stddev(lst):
  total = 0
  mn = mean(lst)
  for i in range(len(lst)):
    total += pow((lst[i]-mn),2)
  return (total/(len(lst)-1))**0.5

# installed "tensileBenchmarkLibraryClient" command
def main():
  TensileBenchmarkLibraryClient(sys.argv[1:])

if __name__ == "__main__":
  main()

