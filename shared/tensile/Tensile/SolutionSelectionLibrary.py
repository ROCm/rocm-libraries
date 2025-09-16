################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .SolutionStructs import Solution

import csv

def getSummationKeys(header):
  keys=[]
  for i in range(7, len(header)):
    keystr = header[i].split("=")[1].strip()
    key = int(keystr)
    keys.append(key)
  return keys

def makeKey(row):
  key=row[3]
  for i in range(4, 7):
    key += "_%s" % row[i].strip()
  return key

def getSolutionBaseKey (solution):

  macroTile0 = solution["MacroTile0"]
  macroTile1 = solution["MacroTile1"]
  globalSplitU = solution["GlobalSplitU"]
  localSplitU = solution["WorkGroup"][2]

  key = "%s_%s_%s_%s" % (macroTile0, macroTile1, localSplitU, globalSplitU)

  return key

def updateIfGT(theDictionary, theKey, theValue):
  if not theKey in theDictionary:
    theDictionary[theKey] = theValue
  else:
    theOldValue = theDictionary[theKey]
    if theValue > theOldValue:
      theDictionary[theKey] = theValue


def updateValidSolutions(validSolutions, analyzerSolutions, solutionMinNaming):
  solutionsStartIndex = len(analyzerSolutions)
  validSelectionSolutionsIncluded = []
  validSelectionSolutionsRemainder = []
  selectionSolutionsIds = set([])
  for validSelectionSolution in validSolutions:
    (validSolution, validSolutionInfo) = validSelectionSolution
    if validSolution in analyzerSolutions:
      validExactSolutionIndex = analyzerSolutions.index(validSolution)
      selectionSolutionsIds.add(validExactSolutionIndex)
      validExactSolution = analyzerSolutions[validExactSolutionIndex]
      validSelectionSolutionsIncluded.append((validExactSolution, validSolutionInfo))
    else:
      validSelectionSolutionsRemainder.append(validSelectionSolution)

  selectionSolutions = []
  for i in range(0 ,len(validSelectionSolutionsIncluded)):
    validSelectionSolution = validSelectionSolutionsIncluded[i]
    (validSolution, validSolutionInfo) = validSelectionSolution
    validSolution["Ideals"] = validSolutionInfo
    analyzerSolutions.append(validSolution)

  solutionsStartIndex = len(analyzerSolutions)

  for i in range(0, len(validSelectionSolutionsRemainder)):
    validSelectionSolution = validSelectionSolutionsRemainder[i]
    (validSolution, validSolutionInfo) = validSelectionSolution
    selectionSolutionIndex = solutionsStartIndex + i
    selectionSolutionsIds.add(selectionSolutionIndex)
    validSolution["SolutionNameMin"] = Solution.getNameMin(validSolution, solutionMinNaming)
    validSolution["Ideals"] = validSolutionInfo
    selectionSolutions.append(validSolution)

  selectionSolutionsIdsList = list(selectionSolutionsIds)

  return selectionSolutionsIdsList


def analyzeSolutionSelection(problemType, selectionFileNameList, numSolutionsPerGroup, solutionGroupMap, solutionsList):

  performanceMap = {}
  solutionsHash = {}

  totalIndices = problemType["TotalIndices"]
  summationIndex = totalIndices
  numIndices = totalIndices + problemType["NumIndicesLD"]
  problemSizeStartIdx = 1
  totalSizeIdx = problemSizeStartIdx + numIndices
  solutionStartIdx = totalSizeIdx + 1
  for fileIdx in range(0, len(selectionFileNameList)):
    solutions = solutionsList[fileIdx]
    selectionFileName = selectionFileNameList[fileIdx]
    numSolutions = numSolutionsPerGroup[fileIdx]
    rowLength = solutionStartIdx + numSolutions
    solutionBaseKeys = []

    for solution in solutions:
      baseKey = getSolutionBaseKey(solution)
      solutionBaseKeys.append(baseKey)

    selectionfFile = open(selectionFileName, "r")
    csvFile = csv.reader(selectionfFile)

    firstRow = 0
    for row in csvFile:
      if firstRow == 0:
        firstRow += 1
      else:
        sumationId = row[summationIndex].strip()

        solutionIndex = 0
        for i in range(solutionStartIdx, rowLength):
          baseKey = solutionBaseKeys[solutionIndex]
          key = "%s_%s" % (baseKey, sumationId)
          solution = solutions[solutionIndex]
          solutionIndex += 1
          value = float(row[i])
          if not solution in solutionsHash:
            dataMap = {}
            solutionsHash[solution] = dataMap

          updateIfGT(solutionsHash[solution], sumationId, value)
          if not key in performanceMap:
            performanceMap[key] = (solution, value)
          else:
            _,valueOld = performanceMap[key]
            if value > valueOld:
              performanceMap[key] = (solution, value)


  validSolutions = []
  validSolutionSet = set([])

  for key in performanceMap:
    solution, _ = performanceMap[key]
    validSolutionSet.add(solution)

  for validSolution in validSolutionSet:
    dataMap = solutionsHash[validSolution]
    validSolutions.append((validSolution,dataMap))

  return validSolutions
