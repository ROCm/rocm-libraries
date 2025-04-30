"""Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import os
import sys
import argparse
import copy

HR = "################################################################################"

################################################################################
# Print Debug
################################################################################

def printWarning(message):
  print("Tensile::WARNING: %s" % message)
  sys.stdout.flush()

def printExit(message):
  print("Tensile::FATAL: %s" % message)
  sys.stdout.flush()
  sys.exit(-1)

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See https://pyyaml.org/wiki/PyYAML for installation instructions.")

def ensurePath( path ):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

################################################################################
# Library Logic Container
################################################################################
class LibraryLogic:

  def __init__(self,filename=None):

    if filename is not None:
      print ("# Reading Library Logic: " + filename)
      try:
        stream = open(filename, "r")
      except IOError:
        printExit("Cannot open file: %s" % filename )
      data = yaml.load(stream, yaml.SafeLoader)

      if isinstance(data, list):

        length = len(data)

        if (length > 0):
          self.__set_versionString(data[0]["MinimumRequiredVersion"])
        else:
          self.__set_versionString(None)

        if (length > 1):
          self.__set_scheduleName(data[1])
        else:
          self.__set_scheduleName(None)

        if (length > 2):
          self.__set_architectureName(data[2])
        else:
          self.__set_architectureName(None)

        if (length > 3):
          self.__set_deviceNames(data[3])
        else:
          self.__set_deviceNames(None)

        if (length > 4):
          self.__set_problemType(data[4])
        else:
          self.__set_problemType(None)

        if (length > 5):
          self.__set_solutionStates(data[5])
        else:
          self.__set_solutionStates(None)

        if (length > 6):
          self.__set_indexOrder(data[6])
        else:
          self.__set_indexOrder(None)

        if (length > 7):
          exactData = data[7]
          exactList = list()
          for exact in exactData:
            size = exact[0]
            if (len(size) > 4):
              exactOut = [size[:4],exact[1]]
              exactList.append(exactOut)
            else:
              exactList.append(exact)
          self.__set_exactLogic(exactList)
        else:
          self.__set_exactLogic(None)

        if (length > 8):
          self.__set_rangeLogic(data[8])
        else:
          self.__set_rangeLogic(None)

      else:
        printExit("Invalid Logic file: %s" % filename)

      stream.close()

    else:
      self.__set_versionString(None)
      self.__set_scheduleName(None)
      self.__set_architectureName(None)
      self.__set_deviceNames(None)
      self.__set_problemType(None)
      self.__set_solutionStates(None)
      self.__set_indexOrder(None)
      self.__set_exactLogic(None)
      self.__set_rangeLogic(None)

  #versionString
  def __get_versionString(self):
    return self.__versionString

  def __set_versionString(self,value):
    self.__versionString = value

  versionString = property(__get_versionString,__set_versionString)

  #scheduleName
  def __get_scheduleName(self):
    return self.__scheduleName

  def __set_scheduleName(self, value):
    self.__scheduleName = value

  scheduleName = property(__get_scheduleName,__set_scheduleName)

  #architectureName
  def __get_architectureName(self):
    return self.__architectureName

  def __set_architectureName(self,value):
    self.__architectureName = value

  architectureName = property(__get_architectureName,__set_architectureName)

  #deviceNames
  def __get_deviceNames(self):
    return self.__deviceNames

  def __set_deviceNames(self,value):
    self.__deviceNames = value

  deviceNames = property(__get_deviceNames,__set_deviceNames)


  #problemTypeState
  def __get_problemType(self):
    return self.__problemType

  def __set_problemType(self,value):
    self.__problemType = value

  problemType = property(__get_problemType,__set_problemType)

  #solutionStates
  def __get_solutionStates(self):
    return self.__solutionStates

  def __set_solutionStates(self,value):
    self.__solutionStates = value

  solutionStates = property(__get_solutionStates,__set_solutionStates)

  #indexOrder
  def __get_indexOrder(self):
    return self.__indexOrder

  def __set_indexOrder(self,value):
    self.__indexOrder = value

  indexOrder = property(__get_indexOrder,__set_indexOrder)


  #exactLogic
  def __get_exactLogic(self):
    return self.__exactLogic

  def __set_exactLogic(self,value):
    self.__exactLogic = value

  exactLogic = property(__get_exactLogic,__set_exactLogic)

  #rangeLogic
  def __get_rangeLogic(self):
    return self.__rangeLogic

  def __set_rangeLogic(self,value):
    self.__rangeLogic = value

  rangeLogic = property(__get_rangeLogic,__set_rangeLogic)

  def writeLibraryLogic(self,filename):

    data = []

    data.append({"MinimumRequiredVersion":self.versionString})
    data.append(self.scheduleName)
    data.append(self.architectureName)
    data.append(self.deviceNames)
    data.append(self.problemType)
    data.append(self.solutionStates)
    data.append(self.indexOrder)
    data.append(self.exactLogic)
    data.append(self.rangeLogic)

    if not data:
      printExit("No data to output")
    else:
      try:
        stream = open(filename, "w")
        yaml.safe_dump(data, stream, default_flow_style=None)
        stream.close()
      except IOError:
        printExit("Cannot open file: %s" % filename)

def MassageTensileLogic(origionalLibraryLogic):

  ouputLibraryLogic = copy.deepcopy(origionalLibraryLogic)

  inputSolutionList = origionalLibraryLogic.solutionStates
  outputSolutionList = ouputLibraryLogic.solutionStates

  solutionIndexKey = "SolutionIndex"
  lastSolutionIndex = 0
  for solution in inputSolutionList:
    solutionIndex = solution[solutionIndexKey]
    if solutionIndex > lastSolutionIndex:
      lastSolutionIndex = solutionIndex

  numSolutions = len(inputSolutionList)

  if numSolutions != (lastSolutionIndex + 1):
    raise Exception("SolutionIndex mismatch. The maximal solution index should match the number of solutions. There may be a formatting issue in the logic file.")

  solutionIndexCounter = lastSolutionIndex + 1

  outputSolutionList = []
  solutionIndexMapper = {}
  for solution in inputSolutionList:
    deepSolution = copy.deepcopy(solution)
    outputSolutionList.append(deepSolution)

  for solution in inputSolutionList:

    if "PackBatchDims" not in solution or solution["PackBatchDims"] != 1:
      newSolution = copy.deepcopy(solution)
      oldSolutionIndex = solution[solutionIndexKey]
      solutionIndexMapper[oldSolutionIndex] = solutionIndexCounter
      newSolution[solutionIndexKey] = solutionIndexCounter

      newSolution["LdcEqualsLdd"] = False
      if "ReplacementKernel" in newSolution:
        newSolution["ReplacementKernel"] = False
      solutionIndexCounter = solutionIndexCounter + 1
      outputSolutionList.append(newSolution)

  ouputLibraryLogic.solutionStates = outputSolutionList

  for exact in origionalLibraryLogic.exactLogic:
    # example exact entry [[123,124,1,123], [5, 4312.3]]
    # the first fiedl in [5, 4312.3] is the mapping to the
    # kernel configuration
    oldSolutionIndex = exact[1][0]
    if oldSolutionIndex in solutionIndexMapper:
      newExact = copy.deepcopy(exact)
      newSolutionIndex = solutionIndexMapper[oldSolutionIndex]
      newExact[1][0] = newSolutionIndex
      ouputLibraryLogic.exactLogic.append(newExact)

  return ouputLibraryLogic

def MassageLogicFile(inputFileName, outputFileName):

  _, fileName = os.path.split(inputFileName)
  print ("processing file: " + fileName)
  libraryLogic = LibraryLogic(inputFileName)
  massagedLibraryLogic = MassageTensileLogic(libraryLogic)
  massagedLibraryLogic.writeLibraryLogic(outputFileName)

def RunMassage():

  print("")
  print(HR)
  print("# Merge Library Logic")
  print(HR)
  print("")

  ##############################################################################
  # Parse Command Line Arguments
  ##############################################################################

  argParser = argparse.ArgumentParser()
  argParser.add_argument("InputPath", help="Path to the un massaged LibraryLogic.yaml files.")
  argParser.add_argument("OutputPath", help="Where to write the massaged files?")

  args = argParser.parse_args()

  inputPath = args.InputPath
  outputPath = args.OutputPath

  print ("Exact Logic Path: " + inputPath)
  print ("OutputPath: " + outputPath)

  print("")
  ensurePath(outputPath)
  if not os.path.exists(inputPath):
    printExit("input logic path %s doesn't exist" % inputPath)

  inputLogicFiles = [os.path.join(inputPath, f) for f in os.listdir(inputPath) \
      if (os.path.isfile(os.path.join(inputPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]

  for unmassagedLogicFilePath in inputLogicFiles:
    _, fileName = os.path.split(unmassagedLogicFilePath)


    outputLogicFilePath = os.path.join(outputPath, fileName)

    try:
      MassageLogicFile(unmassagedLogicFilePath, outputLogicFilePath)
    except Exception as ex:
      print("Exception: {0}".format(ex))

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    RunMassage()
