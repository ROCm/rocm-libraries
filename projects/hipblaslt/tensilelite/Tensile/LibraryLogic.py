################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from pathlib import Path
from typing import Dict
from . import LibraryIO
from . import SolutionSelectionLibrary
from Tensile.Common import print1, print2, HR, printExit, \
  assignParameterWithDefault, ProgressBar, printWarning, ensurePath, \
  LIBRARY_LOGIC_DIR, BENCHMARK_DATA_DIR, getVerbosity, IsaInfo
from Tensile.Common.GlobalParameters import defaultAnalysisParameters, globalParameters, startTime
from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin, getSolutionNameFull

from copy import deepcopy
from sys import stdout

import array
import csv
import os
import time
import math

################################################################################
# Analyze Problem Type
################################################################################
def analyzeProblemType(problemType, problemSizeGroups, inputParameters, libraryLogicPath, splitGSU: bool):
  print2(HR)
  print1("# Analyzing: %s" % problemType)

  enableTileSelection = problemType["TileAwareSelection"]
  solutionsList = []
  problemSizesList = []
  dataFileNameList = []
  selectionFileNameList = []

  for problemSizeGroup in problemSizeGroups:
    problemSizes = problemSizeGroup[0]
    dataFileName = problemSizeGroup[1]
    dataFileNameList.append(dataFileName)
    solutionsFileName = problemSizeGroup[2]
    #print "  problemSizes:", problemSizes
    #print "# DataFileName:", dataFileName
    #print "  solutionsFileName:", solutionsFileName
    if enableTileSelection:
      selectionFileName = problemSizeGroup[3]
      selectionFileNameList.append(selectionFileName)

    ######################################
    # Read Solutions
    # (problemSizes, solutions) are already read and kept in problemSizeGroups, no need to call LibraryIO.readSolutions(solutionsFileName) again
    solutions = problemSizeGroup[4]
    problemSizesList.append(problemSizes)
    solutionsList.append(solutions)
    print1("# Read: %s" % (solutionsFileName))
    print2("# ProblemSizes: %s" % problemSizes)
    print2("# Solutions:")
    solutionIdx = 0
    for solution in solutions:
      print2("#  (%u) %s" % (solutionIdx, getSolutionNameMin(solution, splitGSU)))
      solutionIdx += 1
    print2(HR)

  ######################################
  # Create Logic Analyzer
  logicAnalyzer = LogicAnalyzer( problemType, problemSizesList, solutionsList, \
      dataFileNameList, inputParameters, splitGSU)

  selectionSolutionsIdsList = None
  selectionSolutions = None

  validSelectionSolutions = []

  ######################################
  # Remove invalid solutions
  logicAnalyzer.removeInvalidSolutions()

  ######################################
  # Remove least important solutions
  if inputParameters["LibraryType"] == "FreeSize" or inputParameters["LibraryType"] == "Prediction":
    logicAnalyzer.deReferenceSolutions()
  elif globalParameters["SolutionSelectionAlg"] == 0:
    logicAnalyzer.removeLeastImportantSolutions()
  elif globalParameters["SolutionSelectionAlg"] == 1:
    logicAnalyzer.keepWinnerSolutions()
  else:
    printExit("Bad KeepLogic=%u"%globalParameters["KeepLogic"])

  # print raw data
  if getVerbosity() >= 2:
    line = "After Removals:\n"
    numOther = 1
    for size in logicAnalyzer.numProblemSizes:
      numOther *= size
    numCols = logicAnalyzer.numProblemSizes[1]
    if numCols == 0: numCols = 1
    numOther //= numCols
    for row in range(0, numOther):
      for col in range(0, numCols):
        for sol in range(0, logicAnalyzer.numSolutions):
         line += "% 5.0f" % logicAnalyzer.data[sol + logicAnalyzer.numSolutions*(col + row*numCols)]
        line += "; "
      line += "\n"
    print(line)

  for i in range(0, len(logicAnalyzer.solutions)):
    s = logicAnalyzer.solutions[i]
    s["SolutionIndex"] = i
    s["SolutionNameMin"] = getSolutionNameMin(s, splitGSU)
    s["KernelNameMin"]   = getKernelNameMin(s, splitGSU)
    print1("(%2u) %s : %s" % (i, s["SolutionNameMin"], getSolutionNameFull(s, splitGSU)))

  if enableTileSelection:
    validSelectionSolutions = SolutionSelectionLibrary.analyzeSolutionSelection(problemType, selectionFileNameList, \
        logicAnalyzer.numSolutionsPerGroup,  logicAnalyzer.solutionGroupMap, solutionsList)

    validSelectionSolutionsIncluded = []
    validSelectionSolutionsRemainder = []
    selectionSolutionsIds = set([])
    for validSelectionSolution in validSelectionSolutions:
      (validSolution, validSolutionInfo) = validSelectionSolution
      if validSolution in logicAnalyzer.solutions:
        validExactSolutionIndex = logicAnalyzer.solutions.index(validSolution)
        selectionSolutionsIds.add(validExactSolutionIndex)
        validExactSolution = logicAnalyzer.solutions[validExactSolutionIndex]
        validSelectionSolutionsIncluded.append((validExactSolution, validSolutionInfo))
      else:
        validSelectionSolutionsRemainder.append(validSelectionSolution)

    selectionSolutions = []
    for i in range(0 ,len(validSelectionSolutionsIncluded)):
      validSelectionSolution = validSelectionSolutionsIncluded[i]
      (validSolution, validSolutionInfo) = validSelectionSolution
      validSolution["Ideals"] = validSolutionInfo

    solutionsStartIndex = len(logicAnalyzer.solutions)

    for i in range(0, len(validSelectionSolutionsRemainder)):
      validSelectionSolution = validSelectionSolutionsRemainder[i]
      (validSolution, validSolutionInfo) = validSelectionSolution
      selectionSolutionIndex = solutionsStartIndex + i
      selectionSolutionsIds.add(selectionSolutionIndex)
      validSolution["SolutionNameMin"] = getSolutionNameMin(validSolution, splitGSU)
      validSolution["KernelNameMin"]   = getKernelNameMin(validSolution, splitGSU)
      validSolution["Ideals"] = validSolutionInfo
      selectionSolutions.append(validSolution)

    selectionSolutionsIdsList = list(selectionSolutionsIds)

  ######################################
  # Correct outliers
  """
  if inputParameters["SmoothOutliers"]:
    logicAnalyzer.smoothOutliers()
  """
  numProblemSizes = logicAnalyzer.numProblemSizes
  # print all 2D
  numPermutations = 1
  permutations = []
  for i in range(0, logicAnalyzer.numIndices):
    if i != logicAnalyzer.idx0 and i != logicAnalyzer.idx1:
      numPermutations *= numProblemSizes[i]
  #print numPermutations
  for j in range(0, numPermutations):
    pIdx = j
    permutation = []
    for i in range(0, logicAnalyzer.numIndices):
      if i != logicAnalyzer.idx0 and i != logicAnalyzer.idx1:
        npsi = numProblemSizes[i]
        permutation.append(pIdx%npsi)
        pIdx /= numProblemSizes[i]
    permutations.append(permutation)
  #print permutations
  for permutation in permutations:
    logicAnalyzer.print2D(permutation, libraryLogicPath)

  ######################################
  # Range Logic
  if inputParameters["LibraryType"] != "FreeSize" and inputParameters["LibraryType"] != "Prediction":
    indexOrder = logicAnalyzer.indexOrder
  else:
    indexOrder = None

  ######################################
  # Range Logic
  if inputParameters["LibraryType"] != "FreeSize" and inputParameters["LibraryType"] != "Prediction":
    rangeLogic = logicAnalyzer.enRule(0, logicAnalyzer.globalIndexRange)
    print2("# Final Range Logic:")
    print2(rangeLogic)
    logicComplexity = [0]*logicAnalyzer.numIndices
    logicAnalyzer.scoreLogicComplexity(rangeLogic, logicComplexity)
    print2("# Range Logic Complexity: %s" % logicComplexity)
    score = logicAnalyzer.scoreRangeForLogic( \
        logicAnalyzer.globalIndexRange, rangeLogic)
    print1("\n# Score: %.0f ms" % (score/1000))
    logicAnalyzer.prepareLogic(rangeLogic) # convert indices to sizes, -1
  else:
    rangeLogic = None

  ######################################
  # Exact Logic
  if inputParameters["LibraryType"] != "FreeSize" and inputParameters["LibraryType"] != "Prediction":
    exactLogic = logicAnalyzer.exactWinners
    print1("# Exact Logic:\n")
    print1("%s"%exactLogic)
  else:
    exactLogic = None

  #selectionSolutionsIdsList = list(selectionSolutionsIds)
  return (problemType, logicAnalyzer.solutions, indexOrder, \
       exactLogic, rangeLogic, selectionSolutions, selectionSolutionsIdsList, logicAnalyzer.perfMetric, \
       None)



################################################################################
# LogicAnalyzer
################################################################################
class LogicAnalyzer:

  ##############################################################################
  ##############################################################################
  ###
  ###  Entry / Top-Level Functions
  ###
  ##############################################################################
  ##############################################################################

  ##############################################################################
  # ENTRY: Init
  ##############################################################################
  def __init__(self, problemType, problemSizesList, solutionsList, \
      dataFileNameList, inputParameters, splitGSU: bool):

    # parameters
    self.parameters = inputParameters
    self.splitGSU = splitGSU

    # problem type
    self.problemType = problemType
    self.idx0 = self.problemType["Index0"]
    self.idx1 = self.problemType["Index1"]
    self.idxU = self.problemType["IndexUnroll"]

    # merge solutions from size groups
    # solutions needs to be a set, and offset needs to be mapping
    print1("# Merging Solutions:")
    self.numSolutionsPerGroup = []
    self.solutionGroupMap = []
    self.solutions = []
    solutionsHash = {} # for accelerating lookups

    totalSolutions = 0
    for solutionGroupIdx in range(0, len(solutionsList)):
      solutionGroup = solutionsList[solutionGroupIdx]
      totalSolutions += len(solutionGroup)
    progressBar = ProgressBar(totalSolutions)
    for solutionGroupIdx in range(0, len(solutionsList)):
      solutionGroup = solutionsList[solutionGroupIdx]
      self.numSolutionsPerGroup.append(len(solutionGroup))
      self.solutionGroupMap.append({})
      for solutionIdx in range(0, len(solutionGroup)):
        solution = solutionGroup[solutionIdx]
        if not solution in solutionsHash:
          sIdx = len(self.solutions) # the one we are about to add
          self.solutions.append(solution)
          solutionsHash[solution] = sIdx
        else:
          sIdx = solutionsHash[solution]

        self.solutionGroupMap[solutionGroupIdx][solutionIdx] = sIdx
        progressBar.increment()
    self.numSolutions = len(self.solutions)
    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(getSolutionNameMin(solution, self.splitGSU))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], solution["MacroTile1"]))
    self.flopsPerMac = self.problemType["DataType"].flopsPerMac()

    # merge problem sizes from size groups
    #self.numIndices = len(problemSizesList[0].numProblemSizes)
    self.numIndices = self.problemType["TotalIndices"] + problemType["NumIndicesLD"]
    unifiedProblemSizes = []
    for i in range(0, self.numIndices):
      unifiedProblemSizes.append(set())
    self.exactProblemSizes = set()
    self.rangeProblemSizes = set()
    for problemSizes in problemSizesList:
      # add exacts
      for problem in problemSizes.exacts:
        self.exactProblemSizes.add(tuple(problem.sizes))

      # add ranges
      #print "ProblemSizes", problemSizes.sizes
      #FIXME-problem
      self.rangeProblemSizes.update([tuple(problem.sizes) for problem in problemSizes.problems])
      for rangeSize in problemSizes.ranges:
        # Treat ranges as pile of exacts:
        for rsize in rangeSize.problemSizes:
          self.exactProblemSizes.add(tuple(rsize))
    for i in range(0, len(unifiedProblemSizes)):
      unifiedProblemSizes[i] = sorted(list(unifiedProblemSizes[i]))
    print2("UnifiedProblemSizes: %s" % unifiedProblemSizes)
    print2("ExactProblemSizes: %s" % self.exactProblemSizes)
    print2("RangeProblemSizes: %s" % self.rangeProblemSizes)

    # problem size index <-> size
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    self.numProblemSizes = []
    for i in range(0, self.numIndices):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append([])
      for j in range(0, len(unifiedProblemSizes[i])):
        size = unifiedProblemSizes[i][j]
        self.problemSizeToIndex[i][size] = j
        self.problemIndexToSize[i].append(size)
      self.numProblemSizes.append(len(unifiedProblemSizes[i]))
    print1("# NumProblemSizes: %s" % self.numProblemSizes)

    # total size of data array
    self.totalProblems = 1
    for numProblems in self.numProblemSizes:
      self.totalProblems *= numProblems
    self.totalSize = self.totalProblems * self.numSolutions
    print2("TotalProblems: %u" % self.totalProblems)
    print2("TotalSolutions: %u" % self.numSolutions)
    print2("TotalSize: %u" % self.totalSize)
    # data is a 2D array [problemIdx][solutionIdx] which stores perf data in gflops for
    # the specified solution
    self.data = array.array('f', [-2]*self.totalSize)

    # Each entry in exactWinners is a 2D array [solutionIdx, perf]
    self.exactWinners = {}

    """
    # map problem sizes -> index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, self.numIndices):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append([])
      if self.problemSizes.indexIsSized[i]:
        index = self.problemSizes.indicesSized[sizedIdx]
        sizedIdx += 1
      else:
        index = self.problemSizes.indicesSized[ \
          self.problemSizes.indicesMapped[mappedIdx]]
        mappedIdx += 1
      currentSize = index[0]
      currentStride = index[1]
      idx = 0
      while currentSize <= index[3]:
        self.problemSizeToIndex[i][currentSize] = idx
        self.problemIndexToSize[i].append(currentSize)
        currentSize += currentStride
        currentStride += index[2]
        idx += 1
    """
    #self.rangeIndicesFree = range(0, self.problemType["NumIndicesC"])
    #self.rangeIndicesSummation = range(self.problemType["NumIndicesC"], \
    #    self.problemType["TotalIndices"])
    self.indexOrder = self.recommendedIndexOrder()
    print2("IndexOrder: %s" % self.indexOrder)
    self.globalIndexRange = []
    for i in range(0, self.numIndices):
      self.globalIndexRange.append([0, self.numProblemSizes[i]])
    self.problemIndicesForGlobalRange \
        = self.problemIndicesForRange(self.globalIndexRange)
    self.tab = [""]*self.numIndices

    ######################################
    # Read Data From CSV
    for fileIdx in range(0, len(dataFileNameList)):
      dataFileName = dataFileNameList[fileIdx]
      self.addFromCSV(dataFileName, self.numSolutionsPerGroup[fileIdx], \
          self.solutionGroupMap[fileIdx])



    #print self.data
    # map exact problem sizes to solutions
    print1("# ExactWinners: %s" % self.exactWinners)


  ##############################################################################
  # ENTRY: Add From CSV
  ##############################################################################
  def addFromCSV(self, dataFileName, numSolutions, solutionMap):

    # open file
    print("reading datafile", dataFileName)
    try:
      dataFile = open(dataFileName, "r")
    except IOError:
      printExit("Can't open \"%s\" to get data" % dataFileName )

    # column indices
    csvFile = csv.reader(dataFile)
    problemSizeStartIdx = 1
    # notice that for OperationType != GEMM, the numIndices = 0
    totalSizeIdx = problemSizeStartIdx + self.numIndices

    # need to take care if the loaded csv is the export-winner-version
    csvHasWinner = "_CSVWinner" in dataFileName
    csvHasWinnerColumn = True

    # iterate over rows
    rowIdx = 0
    deviceMaxFreq = None
    for row in csvFile:
      rowIdx+=1
      if rowIdx == 1:
        # get unit (gflops or gflops/cu) of benchmark data
        perfUnit = row[0]
        if perfUnit == "GFlops":
          self.perfMetric = "DeviceEfficiency"
        elif perfUnit == "GFlopsPerCU":
          self.perfMetric = "CUEfficiency"
        else:
          printWarning("Performance unit %s in %s is unrecognized: assuming GFlops (device efficiency)" % (perfUnit, dataFileName))
          self.perfMetric = "DeviceEfficiency"

        if csvHasWinner:
          try:
            # These two columns only appear when using WinnerCSV
            columnOfWinnerGFlops = row.index(" WinnerGFlops")
            columnOfWinnerIdx = row.index(" WinnerIdx")
          except ValueError as e:
            csvHasWinnerColumn = False
            print1(f"Error: Could not find WinnerGFlops or WinnerIdx column in CSV file: {e}")

        # get the length of each row, and derive the first column of the solution instead of using wrong "solutionStartIdx = totalSizeIdx + 1"
        rowLength = len(row)
        solutionStartIdx = rowLength - numSolutions
        continue
      else:
        #if len(row) < rowLength:
        #  printWarning("CSV File %s row %u doesn't have %u elements; ignoring remainer of file." \
        #      % (dataFileName, rowIdx, rowLength) )
        #  break

        # get problem size
        problemSize = []
        for i in range(problemSizeStartIdx, totalSizeIdx):
          problemSize.append(int(row[i]))
        problemSize = tuple(problemSize)
        # Exact Problem Size
        if problemSize in self.exactProblemSizes:

          if csvHasWinner and csvHasWinnerColumn:
            # Faster. Get the winner info from csv directly, avoid an extra loop
            winnerGFlops = float(row[columnOfWinnerGFlops])
            winnerIdx = int(row[columnOfWinnerIdx])
          else:
            # Old code. TODO - Can we get rid of this in the future?
            # solution gflops
            solutionIdx = 0
            winnerIdx = -1
            winnerGFlops = -1
            for i in range(solutionStartIdx, rowLength):
              gflops = float(row[i])
              if gflops > winnerGFlops:
                winnerIdx = solutionIdx
                winnerGFlops = gflops
              solutionIdx += 1

          if globalParameters["UseEffLike"]:
            if not deviceMaxFreq:
              deviceMaxFreq = read_max_freq()

            # calculate effLike
            # effLike = winnerGFlops / Frequency(MHz)
            try:
              if not deviceMaxFreq or deviceMaxFreq <= 0 or math.isnan(deviceMaxFreq):
                performance_metric = round(float(winnerGFlops))
                print("Error when retrieving device frequency, fall back to winnerGFlops.")
              else:
                performance_metric = round(float(winnerGFlops) / deviceMaxFreq, 2)
            except:
              print1("Error: Could not convert winnerGFlops to float.")
              performance_metric = float('nan')
          else:
            try:
              performance_metric = float(winnerGFlops)
            except:
              print1("Error: Could not convert winnerGFlops to float.")
              performance_metric = float('nan')

          if winnerIdx != -1:
            if problemSize in self.exactWinners:
              if winnerGFlops > self.exactWinners[problemSize][1]:
                #print "update exact", problemSize, "CSV index=", winnerIdx, self.exactWinners[problemSize], "->", solutionMap[winnerIdx], winnerGFlops
                self.exactWinners[problemSize] = [solutionMap[winnerIdx], performance_metric]
            else:
              self.exactWinners[problemSize] = [solutionMap[winnerIdx], performance_metric]
              #print "new exact", problemSize, "CSV index=", winnerIdx, self.exactWinners[problemSize]

        # Range Problem Size
        elif problemSize in self.rangeProblemSizes:
          problemIndices = []
          for i in range(0, self.numIndices):
            problemIndices.append(self.problemSizeToIndex[i][problemSize[i]])
          serialIdx = self.indicesToSerial(0, problemIndices)
          # solution gflops
          solutionIdx = 0
          for i in range(solutionStartIdx, rowLength):
            gflops = float(row[i])
            self.data[serialIdx+solutionMap[solutionIdx]] = gflops
            solutionIdx += 1

        # Unknown Problem Size
        else:
          printExit("Huh? %s has ProblemSize %s which isn't in its yaml" \
              % ( dataFileName, list(problemSize)) )
    #print self.data


  ##############################################################################
  # ENTRY: Remove Invalid Solutions
  ##############################################################################
  def removeInvalidSolutions(self):
    #problemIndices = [0]*self.numIndices
    allSolutionValid = False
    while not allSolutionValid:
      invalidIdx = -1
      for problemIndices in self.problemIndicesForGlobalRange:
        problemSerial = self.indicesToSerial(0, problemIndices)
        for solutionIdx in range(0, self.numSolutions):
          gflops = self.data[problemSerial+solutionIdx]
          if gflops == 0:
            invalidIdx = solutionIdx
            break
      if invalidIdx >= 0:
        print1("# Removing Invalid Solution: %u %s" \
            % (invalidIdx, self.solutionNames[invalidIdx]) )
        self.removeSolution(invalidIdx)
      else:
        allSolutionValid = True

  ##############################################################################
  # ENTRY: Dereference ProblemType and Solutions
  ##############################################################################
  def deReferenceSolutions(self):
    self.solutions = deepcopy(self.solutions)

  ##############################################################################
  # ENTRY: Original KeepLogic algorithm: Remove Least Important Solutions,
  # one at a time.  Stop when leastImportantSolution indicates no more
  # solutions can be removed, which appears to be when the solution
  # is used by an exact problem or is the only possible solution for some
  # problem or doesn't improve the a solution by > SolutionImportanceMin%
  ##############################################################################
  def removeLeastImportantSolutions(self):
    # Remove least important solutions
    start = time.time()
    while len(self.solutions) > 1:
      lisTuple = self.leastImportantSolution()
      if lisTuple != None:
        lisIdx = lisTuple[0]
        lisPercSaved = lisTuple[1]
        lisPercWins = lisTuple[2]
        lisPercTime = lisTuple[3]
        if lisPercSaved < self.parameters["SolutionImportanceMin"] or lisPercWins == 0:
          print1("# Removing Unimportant Solution %u/%u: %s ( %f%% wins, %f%% ms time, %f%% ms saved" \
              % (lisIdx, self.numSolutions, self.solutionNames[lisIdx], 100*lisPercWins, 100*lisPercTime, 100*lisPercSaved) )
          self.removeSolution(lisIdx)
          continue
        else:
          break
      else: # no more lis, remainders are exact winner
        break
    stop = time.time()
    print("removeLeastImportantSolutions elapsed time = %.1f secs" % (stop - start))


  ##############################################################################
  # ENTRY: Alternate KeepLogic algorithm that keeps the fastest for each
  #  exact and range.  Other solutions are removed.
  ##############################################################################
  def keepWinnerSolutions(self):

    # solution indexes for the winners:
    winners = set()

    solutionImportance = []
    for i in range(0, self.numSolutions):
      solutionImportance.append([i, 0, 0, 0, False])
    problemSizes = [0]*self.numIndices
    print("problemIndicesForGlobalRange", self.problemIndicesForGlobalRange)
    for problemIndices in self.problemIndicesForGlobalRange:
      for i in range(0, self.numIndices):
        problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
      totalFlops = self.flopsPerMac
      for size in problemSizes:
        totalFlops *= size

      problemSerial = self.indicesToSerial(0, problemIndices)
      winnerIdx = -1
      winnerGFlops = -1e6
      for solutionIdx in range(0, self.numSolutions):
        solutionSerialIdx = problemSerial + solutionIdx
        solutionGFlops = self.data[solutionSerialIdx]
        if solutionGFlops > winnerGFlops:
          winnerIdx = solutionIdx
          winnerGFlops = solutionGFlops

      winners.add(winnerIdx)

    # Always keep the exact sizes:
    for exactProblem in self.exactWinners:
      winnerIdx = self.exactWinners[exactProblem][0]
      #print "keepWinnerSolution adding exact", exactProblem, winnerIdx
      winners.add(winnerIdx)

    print("Winners", winners)
    self.pruneSolutions(winners)



  ##############################################################################
  # ENTRY: En Rule
  # currentIndexIndex = 0, 1, 2, 3...
  # currentIndexRange will have only 1 size for prior indices (unless initial)
  #
  # Rule:
  # [128, [
  #         [64, [
  #                [16, 0],
  #                [2880,1]
  #              ]
  #         ],
  #         [96, [
  #                [16, 0],
  #                [64, 1]
  #              ]
  #         ]
  #       ]
  # ], another
  #
  #
  ##############################################################################
  def enRule(self, currentIndexIndex, currentIndexRange):
    cii = currentIndexIndex
    if currentIndexIndex == 0:
      self.tab[cii] = "[] "
    elif currentIndexIndex == 1:
      self.tab[cii] = "[%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0])
    elif currentIndexIndex == 2:
      self.tab[cii] = "[%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0])
    elif currentIndexIndex == 3:
      self.tab[cii] = "[%2u,%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0], \
          currentIndexRange[self.indexOrder[2]][0])
    elif currentIndexIndex == 4:
      self.tab[cii] = "[%2u,%2u,%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0], \
          currentIndexRange[self.indexOrder[2]][0], \
          currentIndexRange[self.indexOrder[3]][0])
    tab = self.tab[cii]
    if getVerbosity() == 1:
      stdout.write("\n%s"%tab)
    currentIndex = self.indexOrder[currentIndexIndex]
    print2("%senRule(%s)" % (tab, currentIndexRange))
    nextIndexIndex = currentIndexIndex+1
    nextIndexRange = deepcopy(currentIndexRange)
    isLastIndex = currentIndexIndex == self.numIndices-1
    ruleList = []

    ########################################
    # SingleProblem
    ########################################
    if currentIndexRange[currentIndex][1] \
        - currentIndexRange[currentIndex][0] == 1:

      ########################################
      # SingleProblem & LastIndex
      # this is last index, so just return fastest solution
      ########################################
      if isLastIndex:
        print2("%sSingleProblem & LastIndex" % tab)
        winnerIdx = self.winnerForRange(currentIndexRange)
        if winnerIdx < 0:
          print2("%sSingleProblem & LastIndex :: winnerIdx<0; returning" % (tab) )
          return None
        ruleList.append([-1, winnerIdx])
        if getVerbosity() == 1:
          stdout.write("%")

      ########################################
      # SingleProblem & NotLastIndex
      # this isn't last index, so just recursively return next index
      ########################################
      else:
        print2("%sSingleProblem & NotLastIndex" % tab)
        #    % (tab, nextIndexRange) )
        nextRule = self.enRule(nextIndexIndex, nextIndexRange)
        if nextRule == None:
          print2("%sSingleProblem & NotLastIndex :: nextRule==None; returning" % (tab) )
          return None
        rule = [ -1, nextRule ]
        ruleList.append(rule)
        if getVerbosity() == 1:
          stdout.write("%")

    else:
    ########################################
    # MultiProblem
    # Create Initial Rule
    ########################################

      if isLastIndex:
        ########################################
        # MultiProblem & LastIndex
        # InitialRule using winnerForRange()
        ########################################
        print2("%sMultiProblem & LastIndex" % tab)
        winnerIdx = -1
        for problemIndex in range(currentIndexRange[currentIndex][0], \
            currentIndexRange[currentIndex][1]):
          nextIndexRange[currentIndex][0] = problemIndex
          nextIndexRange[currentIndex][1] = problemIndex+1
          winnerIdx = self.winnerForRange(nextIndexRange)
          initialRule = [ currentIndexRange[currentIndex][0], winnerIdx]
          if winnerIdx >= 0:
            break
        if winnerIdx < 0:
          print2("%sMultiProblem & LastIndex :: winnerIdx<0; returning" % (tab) )
          return None

      else:
        ########################################
        # MultiProblem & NotLastIndex
        # InitialRule using enRule()
        ########################################
        print2("%sMultiProblem & NotLastIndex" % tab)

        # create initial rule
        winnerIdx = -1
        nextRule = None
        for problemIndex in range(currentIndexRange[currentIndex][0], \
            currentIndexRange[currentIndex][1]):
          nextIndexRange[currentIndex][0] = problemIndex
          nextIndexRange[currentIndex][1] = problemIndex+1
          nextRule = self.enRule(nextIndexIndex, nextIndexRange)
          # break when found initial rule
          if nextRule != None:
            break

        if nextRule == None:
          printWarning("%sMultiProblem & NotLastIndex :: nextRule==None; returning" % (tab) )
          return None
        initialRule = [ currentIndexRange[currentIndex][0], nextRule ]
      ruleList.append(initialRule)
      print2("%sMultiProblem::InitialRuleList=%s" % (tab, ruleList))
      if getVerbosity() == 1:
        stdout.write("#")

      ########################################
      # MultiProblem
      # Append Rules to Initial Rule
      ########################################
      print2("%sMultiProblem::Improving Rule" % tab)
      for problemIndex in range(currentIndexRange[currentIndex][0]+1, \
          currentIndexRange[currentIndex][1]):
        nextIndexRange[currentIndex][0] = problemIndex
        nextIndexRange[currentIndex][1] = problemIndex+1
        priorRule = ruleList[len(ruleList)-1]
        priorRuleForSize = deepcopy(priorRule)
        priorRuleForSize[0] = problemIndex

        if isLastIndex:
          ########################################
          # nextRule using winnersForRange()
          winnerIdx = self.winnerForRange(nextIndexRange)
          print2("%sMultiProblem::ImproveRule[%u]::LastIndex::WinnerIdx=%u for %s" % (tab, problemIndex, winnerIdx, nextIndexRange))
          # if no solutions benchmarked for this problem size, continue
          if winnerIdx < 0:
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
            print2("%sUpdating range b/c None" % tab)
            if getVerbosity() == 1:
              stdout.write(" ")
            continue
          else:
            candidateRule = [ problemIndex, winnerIdx]
        else:
          ########################################
          # nextRule using enRule()
          nextRule = self.enRule(nextIndexIndex, nextIndexRange)
          print2("%sMultiProblem::ImproveRule[%u]::NotLastIndex::NextRule=%s for %s; %s" % (tab, problemIndex, nextRule, nextIndexIndex, nextIndexRange))
          if nextRule == None:
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
            print2("%sUpdating b/c None" % tab)
            if getVerbosity() == 1:
              stdout.write(" ")
            continue
          else:
            candidateRule = [ problemIndex, nextRule ]

        ########################################
        # candidate same as prior
        if candidateRule[1] == priorRule[1]:
          print2("%sCandidateRule==PriorRule; just updating prior" % (tab))
          ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
          if getVerbosity() == 1:
            stdout.write(" ")
          continue

        ########################################
        # compare candidate vs prior
        else:
          print2("%sCandidateRule!=PriorRule; appending rule assuming its better" % (tab))

          """
          priorRuleScore = self.scoreRangeForLogic(nextIndexRange, \
              [priorRuleForSize])
          logicComplexity = [0]*self.numIndices
          self.scoreLogicComplexity( \
              [priorRuleForSize], logicComplexity)
          priorRuleScore += self.parameters["BranchPenalty"] \
              * sum(logicComplexity)
          # score candidate
          candidateRuleScore = self.scoreRangeForLogic(nextIndexRange, \
              [candidateRule])
          #print "CRS", candidateRuleScore
          logicComplexity = [0]*self.numIndices
          self.scoreLogicComplexity( \
              [candidateRule], logicComplexity)
          candidateRuleScore += self.parameters["BranchPenalty"] \
              * sum(logicComplexity)
          candidateRuleScore += self.parameters["BranchPenalty"] # penalize
          candidateFaster = candidateRuleScore < priorRuleScore
          print2("%sP[%2u]: %s %s~%.0fus < %s~%.0fus" % (tab, problemIndex, \
              "wins" if candidateFaster else "same", \
              candidateRule, candidateRuleScore, priorRuleForSize, \
              priorRuleScore ))
          """

          ########################################
          # candidate wins
          #print candidateRuleScore, priorRuleScore
          if True: # or candidateRuleScore < priorRuleScore:
            ruleList.append(candidateRule)
            print2("%sAppending b/c Different" % tab)
            if getVerbosity() == 1:
              stdout.write("#")

          ########################################
          # prior wins
          else:
            print2("%sPrior Rule Wins" % tab)
            if getVerbosity() == 1:
              stdout.write(".")
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE

    print2("%sReturning RuleList: %s" % (tab, ruleList))
    return ruleList



  ##############################################################################
  ##############################################################################
  ###
  ###  Mid-Level Functions
  ###
  ##############################################################################
  ##############################################################################



  ##############################################################################
  # Prepare Logic
  # convert threshold indices to sizes
  # last threshold = -1
  ##############################################################################
  def prepareLogic(self, logic):
    depth = self.getLogicDepth(logic)
    if depth == 0: return
    indexIndex = self.numIndices - depth
    index = self.indexOrder[indexIndex]
    for i in range(0, len(logic)):
      if i == len(logic)-1:
        logic[i][0] = -1
      else:
        logic[i][0] = self.problemIndexToSize[index][logic[i][0]]
      self.prepareLogic(logic[i][1])


  ##############################################################################
  # Print2D
  ##############################################################################
  def print2D(self, indices, libraryLogicPath):
    indicesIdx = 0
    problemIndices = []
    for i in range(0, self.numIndices):
      if i == self.idx0:
        problemIndices.append(-1)
      elif i == self.idx1:
        problemIndices.append(-1)
      else:
        problemIndices.append(indices[indicesIdx])
        indicesIdx += 1

    winnerIndices = []
    w = "winner"
    g = "gflops"
    f = "faster"
    s = "second"
    sss = []
    for sIdx in range(0, self.numSolutions):
      sss.append("Sol[%u]" % sIdx)
    for j in range(0, self.numProblemSizes[1]):
      w += ",%4u" % self.problemIndexToSize[1][j]
      g += ",%4u" % self.problemIndexToSize[1][j]
      f += ",%4u" % self.problemIndexToSize[1][j]
      s += ",%4u" % self.problemIndexToSize[1][j]
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += ",%4u" % self.problemIndexToSize[1][j]
    w += "\n"
    g += "\n"
    f += "\n"
    s += "\n"
    for sIdx in range(0, self.numSolutions):
      sss[sIdx] += "\n"
    for i in range(0, self.numProblemSizes[0]):
      problemIndices[self.idx0] = i
      w += "%4u" % self.problemIndexToSize[0][i]
      g += "%4u" % self.problemIndexToSize[0][i]
      f += "%4u" % self.problemIndexToSize[0][i]
      s += "%4u" % self.problemIndexToSize[0][i]
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += "%4u" % self.problemIndexToSize[0][i]
      for j in range(0, self.numProblemSizes[1]):
        problemIndices[self.idx1] = j
        problemSerial = self.indicesToSerial(0, problemIndices)
        for sIdx in range(0, self.numSolutions):
          sss[sIdx] += ",%f" % self.data[problemSerial+sIdx]
        winnerIdx = 0
        secondIdx = 1
        winnerGFlops = self.data[problemSerial+0]
        secondGFlops = 1e-9
        for solutionIdx in range(1, self.numSolutions):
          solutionSerialIdx = problemSerial + solutionIdx
          solutionGFlops = self.data[solutionSerialIdx]
          if solutionGFlops > winnerGFlops:
            secondIdx = winnerIdx
            secondGFlops = winnerGFlops
            winnerIdx = solutionIdx
            winnerGFlops = solutionGFlops
          elif solutionGFlops > secondGFlops:
            secondIdx = solutionIdx
            secondGFlops = solutionGFlops

        if winnerIdx not in winnerIndices:
          winnerIndices.append(winnerIdx)
        w += ",%4u" % winnerIdx
        g += ",%f" % winnerGFlops
        f += ",%f" % (winnerGFlops/secondGFlops)
        s += ",%4u" % (secondIdx)
      w += "\n"
      g += "\n"
      f += "\n"
      s += "\n"
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += "\n"

    w += "\n\n"
    g += "\n\n"
    f += "\n\n"
    s += "\n\n"
    for sIdx in range(0, self.numSolutions):
      sss[sIdx] += "\n\n"
    w += "Winners:\n"
    for winnerIdx in winnerIndices:
      w += "%4u, %s, %s\n" % (winnerIdx, self.solutionTiles[winnerIdx], self.solutionNames[winnerIdx])

    printFileName = "Winner2D"
    for idx in indices:
      printFileName += "_%u" % idx
    printFileName += ".csv"
    printFile = open(os.path.join(libraryLogicPath, printFileName), "w")
    printFile.write( w )
    printFile.write( g )
    printFile.write( f )
    printFile.write( s )
    for sIdx in range(0, self.numSolutions):
      printFile.write( sss[sIdx] )
    printFile.close()


  ##############################################################################
  # Least Important Solution
  ##############################################################################
  def leastImportantSolution(self):
    solutionImportance = []
    for i in range(0, self.numSolutions):
      solutionImportance.append([i, 0, 0, 0, False])
    problemSizes = [0]*self.numIndices
    totalSavedMs = 0
    totalExecMs = 0
    totalWins = 0
    for problemIndices in self.problemIndicesForGlobalRange:
      for i in range(0, self.numIndices):
        problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
      totalFlops = self.flopsPerMac
      for size in problemSizes:
        totalFlops *= size

      problemSerial = self.indicesToSerial(0, problemIndices)
      winnerIdx = -1
      winnerGFlops = -1e6
      secondGFlops = -1e9
      for solutionIdx in range(0, self.numSolutions):
        solutionSerialIdx = problemSerial + solutionIdx
        solutionGFlops = self.data[solutionSerialIdx]
        if solutionGFlops > winnerGFlops:
          secondGFlops = winnerGFlops
          winnerIdx = solutionIdx
          winnerGFlops = solutionGFlops
        elif solutionGFlops > secondGFlops:
          secondGFlops = solutionGFlops

      winnerTimeMs = totalFlops / winnerGFlops / 1000000.0
      secondTimeMs = totalFlops / secondGFlops / 1000000.0
      if winnerGFlops > 0 and secondGFlops > 0:
        solutionImportance[winnerIdx][1] += (secondTimeMs - winnerTimeMs)
        totalSavedMs += secondTimeMs - winnerTimeMs
      if winnerGFlops > 0:
        solutionImportance[winnerIdx][2] += 1
        solutionImportance[winnerIdx][3] += winnerTimeMs
        totalExecMs += winnerTimeMs
        totalWins += 1
        if secondGFlops <= 0:
          solutionImportance[winnerIdx][4] = True # this is only valid solution for this problem size, keep it


    # print data before sorting
    for i in range(0, self.numSolutions):
      print2("[%2u] %s: %e saved, %u wins, %u time, %s" \
          % (solutionImportance[i][0], \
          self.solutionNames[solutionImportance[i][0]], \
          solutionImportance[i][1], \
          solutionImportance[i][2], \
          solutionImportance[i][3], \
          "singular" if solutionImportance[i][4] else "" ) )

    totalSavedMs = max(1, totalSavedMs)
    solutionImportance.sort(key=lambda x: x[1])
    for i in range(0, self.numSolutions):
      solutionIdx = solutionImportance[i][0]
      canRemove = not solutionImportance[i][4] # don't remove if is only win for any size
      for exactProblem in self.exactWinners:
        winnerIdx = self.exactWinners[exactProblem][0]
        if solutionIdx == winnerIdx: # exact winners are important
          canRemove = False
          break
      if canRemove:
        idx = solutionImportance[i][0]
        if totalSavedMs > 0:
          percSaved = 1.0 * solutionImportance[i][1] / totalSavedMs
        else:
          percSaved = 0
        if totalWins > 0:
          percWins = 1.0 * solutionImportance[i][2] / totalWins
        else:
          percWins = 0
        if totalExecMs > 0:
          percTime = 1.0 * solutionImportance[i][3] / totalExecMs
        else:
          percTime = 0
        return ( idx, percSaved, percWins, percTime )
    return None


  ##############################################################################
  # Remove Solution
  ##############################################################################
  def removeSolution(self, removeSolutionIdx):

    # temporarily move current to old
    oldSolutions = deepcopy(self.solutions)
    oldNumSolutions = self.numSolutions
    oldData = deepcopy(self.data)

    # update solutions
    self.solutions = []
    for i in range(0, oldNumSolutions):
      if i != removeSolutionIdx:
        self.solutions.append(oldSolutions[i])
    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(getSolutionNameMin(solution, self.splitGSU))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], \
          solution["MacroTile1"]))
    self.numSolutions = len(self.solutions)

    # update data
    self.totalSize = self.totalProblems * self.numSolutions
    self.data = array.array('f', [0]*self.totalSize)
    for problemIndex in range(0, self.totalProblems):
      newSolutionIdx = 0
      for oldSolutionIdx in range(0, oldNumSolutions):
        if oldSolutionIdx != removeSolutionIdx:
          self.data[problemIndex*self.numSolutions+newSolutionIdx] \
              = oldData[problemIndex*oldNumSolutions+oldSolutionIdx]
          newSolutionIdx += 1

    # update exact Winners
    for problemSize in self.exactWinners:
      if self.exactWinners[problemSize][0] >= removeSolutionIdx:
        self.exactWinners[problemSize][0] -= 1


  ##############################################################################
  # Prune a list of solutions, keeping only the indices specified in
  # keepSolutions.  keepSolutions is a set not a list
  ##############################################################################
  def pruneSolutions(self, keepSolutions):

    removeSolutionIdxList = []
    solutionMapNewToOld = [] # dense mapping
    solutionMapOldToNew = [-1] * self.numSolutions

    # temporarily move current to old
    oldSolutions = deepcopy(self.solutions)
    oldNumSolutions = self.numSolutions
    oldData = deepcopy(self.data)
    # update solutions
    self.solutions = []
    for i in range(0, oldNumSolutions):
      if i in keepSolutions:
        solutionMapNewToOld.append(i)
        solutionMapOldToNew[i] = len(self.solutions)
        self.solutions.append(oldSolutions[i])
      else:
        removeSolutionIdxList.append(i)

    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(getSolutionNameMin(solution, self.splitGSU))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], \
          solution["MacroTile1"]))
    self.numSolutions = len(self.solutions)

    # update data
    self.totalSize = self.totalProblems * self.numSolutions
    self.data = array.array('f', [0]*self.totalSize)
    for problemIndex in range(0, self.totalProblems):
      for newSolutionIdx in range(0, self.numSolutions):
        oldSolutionIdx = solutionMapNewToOld[newSolutionIdx]
        self.data[problemIndex*self.numSolutions+newSolutionIdx] \
            = oldData[problemIndex*oldNumSolutions+oldSolutionIdx]

    # update exact Winners
    for problemSize in self.exactWinners:
      #print "prune updating exacWinner", problemSize, \
      #        "from ", self.exactWinners[problemSize][0], \
      #        "to ", solutionMapOldToNew[self.exactWinners[problemSize][0]]
      self.exactWinners[problemSize][0] = \
          solutionMapOldToNew[self.exactWinners[problemSize][0]]
      if self.exactWinners[problemSize][0] == -1:
        print(("warning: exactWinner[", problemSize, "] == -1"))
      if self.exactWinners[problemSize][0] >= self.numSolutions:
        print(("warning: exactWinner[", problemSize, "] "))


  ##############################################################################
  # Score Range For Logic
  ##############################################################################
  def scoreRangeForLogic(self, indexRange, logic):
    depth = self.getLogicDepth(logic)
    depth = self.numIndices - depth
    fullLogic = deepcopy(logic)
    for i in range(0, depth):
      fullLogic = [[-1, fullLogic]]
    return self.scoreRangeForFullLogic(depth, indexRange, fullLogic)

  ##############################################################################
  # Score Range For Full Logic
  ##############################################################################
  def scoreRangeForFullLogic(self, depth, indexRange, logic):
    score = 0
    for problemIndices in self.problemIndicesForRange(indexRange):
      problemSerial = self.indicesToSerial(0, problemIndices)
      totalFlops = self.totalFlopsForProblemIndices(problemIndices)
      solutionIdx = self.getSolutionForProblemIndicesUsingLogic( \
          problemIndices, logic)
      if solutionIdx == None:
        printWarning("SolutionIdx = None. This should never happen.")
        continue
      solutionGFlops = self.data[problemSerial + solutionIdx]
      solutionGFlops = max(1E-9, solutionGFlops)
      timeUs = totalFlops / solutionGFlops / 1000
      score += timeUs
    return score

  ##############################################################################
  # Get Solution For Problem Indices Using Logic
  ##############################################################################
  def getSolutionForProblemIndicesUsingLogic(self, problemIndices, logic):
    currentProblemIndices = self.toIndexOrder(problemIndices)
    currentLogic = logic
    for i in range(0, self.numIndices):
      currentSizeIndex = currentProblemIndices[0]
      for j in range(0, len(currentLogic)):
        if currentLogic[j][0] < 0:
          currentProblemIndices = currentProblemIndices[1:]
          currentLogic = currentLogic[j][1]
          break
        if currentLogic[j][0] >= 0:
          if currentSizeIndex <= currentLogic[j][0]:
            currentProblemIndices = currentProblemIndices[1:]
            currentLogic = currentLogic[j][1]
            break
    return currentLogic


  ##############################################################################
  ##############################################################################
  ###
  ###  Helper / Low-Level Functions
  ###
  ##############################################################################
  ##############################################################################


  ##############################################################################
  # Get Winner For Problem
  def getWinnerForProblem(self, problemIndices):
    problemSerial = self.indicesToSerial(0, problemIndices)
    winnerIdx = -1
    winnerGFlops = -1
    for solutionIdx in range(0, self.numSolutions):
      solutionSerialIdx = problemSerial + solutionIdx
      solutionGFlops = self.data[solutionSerialIdx]
      solutionGFlops = max(1E-9, solutionGFlops)
      if solutionGFlops > winnerGFlops:
        winnerIdx = solutionIdx
        winnerGFlops = solutionGFlops
    # print "Winner %u %f" % (winnerIdx, winnerGFlops)
    return (winnerIdx, winnerGFlops)


  ##############################################################################
  # Winner For Range, -1 if nothing benchmarked
  def winnerForRange(self, indexRange):
    if self.numSolutions == 1:
      return 0
    else:
      scores = self.scoreRangeForSolutions(indexRange)
      winnerIdx = 0
      hasWinner = False
      for solutionIdx in range(1, self.numSolutions):
        if scores[solutionIdx] < scores[winnerIdx]:
          winnerIdx = solutionIdx
          hasWinner = True
        elif scores[solutionIdx] > scores[winnerIdx]:
          hasWinner = True
        else:
          pass # still no winner

      return winnerIdx if hasWinner else -1


  ##############################################################################
  # Score (microseconds) Range For Solutions
  def scoreRangeForSolutions(self, indexRange):
    scores = [0]*self.numSolutions
    for problemIndices in self.problemIndicesForRange(indexRange):
      problemSerial = self.indicesToSerial(0, problemIndices)
      totalFlops = self.totalFlopsForProblemIndices(problemIndices)
      for solutionIdx in range(0, self.numSolutions):
        gflops = self.data[problemSerial+solutionIdx]
        if gflops > 0:
          timeUs = totalFlops / gflops / 1000
        else: # this solution not benchmarked for this size, therefore set
          # its score to +inf so that its automatically disqualified
          timeUs = float("inf")
        scores[solutionIdx] += timeUs
    return scores


  ##############################################################################
  # Score Logic Complexity
  def scoreLogicComplexity(self, logic, logicComplexity):
    depth = self.getLogicDepth(logic)
    if depth == 0: return
    depth = self.numIndices - depth
    for i in range(0, len(logic)):
      logicComplexity[depth] += 1
      self.scoreLogicComplexity(logic[i][1], logicComplexity)


  ##############################################################################
  # Get Logic Depth
  def getLogicDepth(self, logic):
    obj = logic
    depth = 0
    while isinstance(obj, list):
      obj = obj[0][1]
      depth += 1
    return depth


  ##############################################################################
  # To Index Order
  def toIndexOrder(self, problemIndices):
    ordered = []
    for i in self.indexOrder:
      ordered.append(problemIndices[i])
    return ordered


  ##############################################################################
  # Total Flops For Problem Indices
  def totalFlopsForProblemIndices(self, problemIndices):
    totalFlops = self.flopsPerMac
    for i in range(0, self.numIndices):
      totalFlops *= self.problemIndexToSize[i][problemIndices[i]]
    return totalFlops


  ##############################################################################
  # Recommended Index Order
  # TODO, this may depend on transposes
  def recommendedIndexOrder(self):
    order = []
    for i in range(0, self.problemType["TotalIndices"]):
      if i != self.idxU and i != self.idx1 and i != self.idx0:
        order.append(i)
    order.append(self.idxU)
    order.append(self.idx0)
    order.append(self.idx1)
    return order

  ##############################################################################
  # Problem Indices For Range
  def problemIndicesForRange(self, indexRange):
    problemIndexList = []
    # early return for empty set
    for i in range(0, self.numIndices):
      if indexRange[i][0] == indexRange[i][1]:
        return []
    problemIndices = []
    for idx in indexRange:
      problemIndices.append(idx[0])
    moreProblems = True
    while moreProblems:
      problemIndexList.append(deepcopy(problemIndices))
      # next problem
      problemIndices[0] += 1
      for i in range(0, self.numIndices):
        if problemIndices[i] >= indexRange[i][1]:
          if i == self.numIndices-1:
            moreProblems = False
            break
          else:
            problemIndices[i] = indexRange[i][0]
            problemIndices[i+1] += 1
        else:
          break
    return problemIndexList


  ##############################################################################
  # Get Item
  def __getitem__(self, indexTuple):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices)
    return self.data[serial]


  ##############################################################################
  # Set Item
  def __setitem__(self, indexTuple, value):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices )
    self.data[serial] = value


  ##############################################################################
  # Indices -> Serial
  def indicesToSerial(self, solutionIdx, indices ):
    serial = 0
    stride = 1
    serial += solutionIdx * stride
    stride *= self.numSolutions
    for i in range(0, self.numIndices):
      serial += indices[i] * stride
      stride *= self.numProblemSizes[i]
    return serial


def generateLogic(
    config,
    benchmarkDataPath,
    libraryLogicPath,
    cxxCompiler: str,
    splitGSU: bool,
    printSolutionRejectionReason: bool,
    printIndexAssignmentInfo: bool,
    isaInfoMap: Dict[str, IsaInfo]
  ):

  libraryLogicPath = ensurePath(libraryLogicPath)

  print2("# LibraryLogic config: %s" % config)
  print2("# DefaultAnalysisParameters: " % defaultAnalysisParameters)

  # Assign Defaults
  analysisParameters = {}
  for parameter in defaultAnalysisParameters:
    assignParameterWithDefault(analysisParameters, parameter, config, \
        defaultAnalysisParameters)

  print1("")
  print1(HR)
  currentTime = time.time()
  elapsedTime = currentTime - startTime
  print1("# Analysing data in %s - %.3fs" % (benchmarkDataPath, elapsedTime) )
  for parameter in analysisParameters:
    print2("#   %s: %s" % (parameter, analysisParameters[parameter]))
  print1(HR)
  print1("")

  ##############################################################################
  # Determine Which Problem Types
  ##############################################################################
  problemTypes = {}
  if not os.path.exists(benchmarkDataPath):
    printExit("Path doesn't exist: %s" % benchmarkDataPath)
  fileNames = os.listdir(benchmarkDataPath)
  fileNames = sorted(fileNames)
  for fileName in fileNames:
    if os.path.splitext(fileName)[1] == ".csv":
      fileBase = os.path.splitext( \
          os.path.join(benchmarkDataPath, \
          fileName))[0]
      dataFileName = fileBase + ".csv"
      solutionsFileName = fileBase + ".yaml"
      selectionFileName = fileBase + ".gsp"
      if not os.path.exists(dataFileName):
        printExit("%s doesn't exist for %s" % (dataFileName, fileBase) )
      if not os.path.exists(solutionsFileName):
        printExit("%s doesn't exist for %s" % (solutionsFileName, fileBase) )
      (problemSizes, solutions) = LibraryIO.parseSolutionsFile(
                                      solutionsFileName,
                                      cxxCompiler,
                                      splitGSU,
                                      printSolutionRejectionReason,
                                      printIndexAssignmentInfo,
                                      isaInfoMap
                                  )
      if len(solutions) == 0:
        printExit("%s doesn't contains any solutions." % (solutionsFileName) )
      problemType = solutions[0]["ProblemType"]
      if problemType not in problemTypes:
        problemTypes[problemType] = []
      problemTypes[problemType].append( (problemSizes, \
          dataFileName, solutionsFileName, selectionFileName, solutions) )

  for problemType in problemTypes:
    logicTuple = analyzeProblemType(problemType, problemTypes[problemType], analysisParameters, libraryLogicPath, splitGSU)

    filename = os.path.join(libraryLogicPath, \
        "{}_{}".format(analysisParameters["ScheduleName"], str(problemType)))

    print2("# writing library logic YAML {}".format(filename))
    data = LibraryIO.createLibraryLogic(analysisParameters["ScheduleName"], \
        analysisParameters["ArchitectureName"], analysisParameters["DeviceNames"], analysisParameters["LibraryType"], logicTuple)

    if globalParameters["LogicFormat"] == "yaml":
      LibraryIO.writeYAML(filename + ".yaml", data, explicit_start=False, explicit_end=False)
    elif globalParameters["LogicFormat"] == "json":
      LibraryIO.write(filename, data, "json")
    else:
      printExit("Unrecognized LogicFormat", globalParameters["LogicFormat"])

  currentTime = time.time()
  elapsedTime = currentTime - startTime
  print1("%s\n# Finish Analysing data to %s in %.3fs\n%s" % (HR, os.path.split(libraryLogicPath)[0], elapsedTime, HR) )


##############################################################################
# Error handling for frequency issues
##############################################################################
def handle_frequency_issue(message):
    print1(message)
    print1("Input the frequency manually to proceed.")

    while True:
      frequency_input = input("Frequency: ").strip()
      if frequency_input == "":
          print1("Frequency cannot be empty")
          continue
      try:
          frequency = float(frequency_input)
          if frequency > 0:
            return frequency
          else:
            print1("Frequency cannot be negative or zero.")
      except ValueError:
          print1("Invalid frequency.Please input a valid frequency.")

def read_max_freq():
    try:
        max_freq = os.environ.get("MAX_FREQ")
        if max_freq is None or max_freq.strip() == "":
          raise ValueError("Environment variable 'MAX_FREQ' is not set or is empty.")

        return float(max_freq)
    except ValueError as ve:
        print(f"Error: {ve}")  # Handle invalid or missing value
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")  # Handle any other unexpected errors
        return None

################################################################################
################################################################################
###
###   Main
###
################################################################################
################################################################################
def main(
      config,
      cxxCompiler: str,
      outputPath: Path,
      splitGSU: bool,
      printSolutionRejectionReason: bool,
      printIndexAssignmentInfo: bool,
      isaInfoMap: Dict[str, IsaInfo]
    ):
  benchmarkDataPath = outputPath / BENCHMARK_DATA_DIR
  libraryLogicPath = outputPath / LIBRARY_LOGIC_DIR
  generateLogic(
    config,
    benchmarkDataPath,
    libraryLogicPath,
    cxxCompiler,
    splitGSU,
    printSolutionRejectionReason,
    printIndexAssignmentInfo,
    isaInfoMap
  )
