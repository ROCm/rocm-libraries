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

from __future__ import print_function
import os
import sys
import argparse

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
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

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


def MergeTensileLogicFiles(origionalLibraryLogic, exactLibraryLogic):

  mergedLibraryLogic = LibraryLogic()

  solutionList = origionalLibraryLogic.solutionStates
  solutionListExact = exactLibraryLogic.solutionStates

  indexKey = "SolutionIndex"
  # zero out solution indexes
  for solution in solutionList:
    solution[indexKey] = 0

  for solution in solutionListExact:
    solution[indexKey] = 0

  newSolutionOffset = len(solutionList)

  filterdSolutionExactList = []
  replicationMapping = {}
  idx = 0
  idxMapping = newSolutionOffset

  mergedSolutionList = []
  for solution in solutionList:
    mergedSolutionList.append(solution)

  # construct the mappings from the old exact kernal configurations
  # to their definitions in the merged files
  for solution in solutionListExact:
    if solution in solutionList:
      # if solution exists in the origional configuration the
      # its placement in the merged kernel configurations list
      # gets mapped to the pre-existing configuration
      idxOrg = solutionList.index(solution)
      replicationMapping[idx] = idxOrg

    else:
      filterdSolutionExactList.append(solution)
      # if the solution does not exist in the origional configurations
      # it gets mapped to the new offset
      replicationMapping[idx] = idxMapping
      mergedSolutionList.append(solution)
      idxMapping += 1

    idx += 1

  exactLogic = origionalLibraryLogic.exactLogic
  exactLogicExact = exactLibraryLogic.exactLogic

  filteredExactLogicExact = []

  # use the mapping from above to remap the exact logic
  # in the merged file
  for exact in exactLogicExact:
    # example exact entry [[123,124,1,123], [5, 4312.3]]
    # the first fiedl in [5, 4312.3] is the mapping to the
    # kernel configuration
    kernelIndex = exact[1][0]

    if kernelIndex in replicationMapping:
      exact[1][0] = replicationMapping[kernelIndex]

    filteredExactLogicExact.append(exact)

  sizeList, _ = zip(*filteredExactLogicExact)

  mergedExactLogic = []
  for logicMapping in exactLogic:
    if logicMapping[0] not in sizeList:
      mergedExactLogic.append(logicMapping)

  for logicMapping in filteredExactLogicExact:
    mergedExactLogic.append(logicMapping)

  # re index solutions
  index = 0
  for solution in mergedSolutionList:
    solution[indexKey] = index
    index += 1


  mergedLibraryLogic.versionString = origionalLibraryLogic.versionString
  mergedLibraryLogic.scheduleName = origionalLibraryLogic.scheduleName
  mergedLibraryLogic.architectureName = origionalLibraryLogic.architectureName
  mergedLibraryLogic.deviceNames = origionalLibraryLogic.deviceNames
  mergedLibraryLogic.problemType = origionalLibraryLogic.problemType
  mergedLibraryLogic.solutionStates = mergedSolutionList
  mergedLibraryLogic.indexOrder = origionalLibraryLogic.indexOrder
  mergedLibraryLogic.exactLogic = mergedExactLogic
  mergedLibraryLogic.rangeLogic  = origionalLibraryLogic.rangeLogic

  return mergedLibraryLogic


def ProcessMergeLogicFile(exactFileName, origionalFileName, outputFileName):

  _, fileName = os.path.split(exactFileName)

  print ("processing file: " + fileName)

  libraryLogic = LibraryLogic(origionalFileName)
  libraryLogicExact = LibraryLogic(exactFileName)

  mergedLibraryLogic = MergeTensileLogicFiles(libraryLogic,libraryLogicExact)

  mergedLibraryLogic.writeLibraryLogic(outputFileName)

def RunMergeTensileLogicFiles():

  print("")
  print(HR)
  print("# Merge Library Logic")
  print(HR)
  print("")

  ##############################################################################
  # Parse Command Line Arguments
  ##############################################################################

  argParser = argparse.ArgumentParser()
  argParser.add_argument("OrigionalLogicPath", help="Path to the origional LibraryLogic.yaml input files.")
  argParser.add_argument("ExactLogicPath", help="Path to the exact LibraryLogic.yaml input files.")
  argParser.add_argument("OutputPath", help="Where to write library files?")

  args = argParser.parse_args()

  origionalLogicPath = args.OrigionalLogicPath
  exactLogicPath = args.ExactLogicPath
  outputPath = args.OutputPath
  print ("Origional Logic Path: " + origionalLogicPath)
  print ("Exact Logic Path: " + exactLogicPath)
  print ("OutputPath: " + outputPath)

  print("")
  ensurePath(outputPath)
  if not os.path.exists(exactLogicPath):
    printExit("LogicPath %s doesn't exist" % exactLogicPath)

  exactLogicFiles = [os.path.join(exactLogicPath, f) for f in os.listdir(exactLogicPath) \
      if (os.path.isfile(os.path.join(exactLogicPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]

  for exactLogicFilePath in exactLogicFiles:
    _, fileName = os.path.split(exactLogicFilePath)

    origionalLogicFilePath = os.path.join(origionalLogicPath, fileName)
    if os.path.isfile(origionalLogicFilePath):

      outputLogicFilePath = os.path.join(outputPath, fileName)

      try:
        ProcessMergeLogicFile(exactLogicFilePath, origionalLogicFilePath, outputLogicFilePath)
      except Exception as ex:
        print("Exception: {0}".format(ex))

    else:
      print ("# file does not exist in origional directory " + origionalLogicFilePath)


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    RunMergeTensileLogicFiles()
