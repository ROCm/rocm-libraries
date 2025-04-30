#!/usr/bin/python

"""Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

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

import yaml
import sys
import os
import subprocess
from subprocess import PIPE
from tempfile import mkdtemp
from shutil import rmtree

#
minimumVersionIdx       = 0
prefixIdx               = 1
architectureIdx         = 2
deviceNamesIdx          = 3
problemTypeStateIdx     = 4
solutionListIdx         = 5
indexOrderIdx           = 6
exactLogicListIdx       = 7       #[[m, n, batch_size, k, ...], [kernelIdx, GFLOPS]]
mIdx        = 0
nIdx        = 1
batchIdx    = 2
kIdx        = 3
rangeLogicIdx           = 8
tileSelectionLogicIdx   = 9       #Only if tile aware selection is enabled

helpMessage = """Script for determining which problem sizes in a log file have been pre-tuned for
by Tensile.
Usage:
python3 check-for-pretuned-sizes.py -f logfilePath
                                      [-c commit] [-a architecture] [-u] [-l] [--help]
Options:
-f       Log file containing benchmark tests.
-a       Architecture. Matches any problem containing this argument.
         ex: vega10
-c       Specify commit of rocBLAS to check. If unspecified, will use most recent master.
-l       Attempt to detect the installed version of rocBLAS
         (Currently just works when installing from cloned repository)
-u       Compares detected problem sizes to those found in the most recent develop commit.
--help   Displays this message.
NOTE: Flags cannot currently be combined together as in -lu, they must be specified
separately in the form -l -u
"""

usageMessage = """Usage:
python3 check-for-pretuned-sizes.py -f logfilePath
                                      [-c commit] [-a architecture] [-u] [-l] [--help]"""


def removeChar(text, removedChar):
    return "".join(filter(lambda c : c != removedChar, text))

class ParseOptionError(Exception):
    pass

#Same usage as getopt but doesn't fail on unknown args and returns dict
def parseOptions(textList, shortArgs, longArgs=[]):
    hasArg = []
    lastChar = ':'
    optionList = []
    #Set keys for short arguments
    for c in shortArgs:
        #':' indicates c takes an argument
        if c == ':':
            if not lastChar.isalpha():
                raise ParseOptionError("Error: Colon must follow alphabetic character")
            hasArg[-1] = True
        else:
            optionList.append('-' + c)
            hasArg.append(False)
        lastChar = c

    #Set keys for long arguments
    for s in longArgs:
        #'=' indicates s takes an argument
        if s[-1] == '=':
            hasArg.append(True)
            optionList.append("--" + s[:-1])
        else:
            hasArg.append(False)
            optionList.append("--" + s)

    hasArg = dict(zip(optionList,
        hasArg))

    #Extract options from textList
    optionDict = dict([])
    needsArg = None
    for item in textList:
        #If this is an argument of a previous option...
        if needsArg != None:
            #Store associated with it
            optionDict[needsArg] = item
            needsArg = None
        #Otherwise check if it's a valid option
        elif item in optionList:
            optionDict[item] = None     #Set as none for now
            if hasArg[item]:
                needsArg = item         #Set to capture next item

    return dict(zip(
        list(map(lambda s : removeChar(s, '-'), optionDict.keys())),
        optionDict.values()))

#assumes b has at least as many periods as a
def isVersionGreaterThanOrEqual(versionA, versionB):
    for a, b in zip(versionA.split('.'), versionB.split('.')):
        if int(a) != int(b):
            return int(a) > int(b)
    return True     #Is equal


def shellCmd(command):
    spReturn = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE)
    return spReturn.stdout.decode('utf-8').rstrip('\n')


def getInstalledRocBLASCommitHash():
    versionString = shellCmd("dpkg -s rocblas | grep Version") #Format "Version X.XX.X.XXXX-CommitHash"
    spaceSplit = versionString.split()
    if len(spaceSplit) > 1:
        dashSplit = spaceSplit[1].split('-')
        if len(dashSplit) > 1:
            return dashSplit[1]
        else:
            print("Error: No commit hash found in \"dpkg -s rocblas | grep Version\" command")
            return ""
    else:
        print("Error: Could not parse output from \"dpkg -s rocblas | grep Version\" command")
        return ""


def cloneRepository(destinationPath):
    print("Cloning repository...")
    shellCmd("git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git %s" % destinationPath)

def checkoutSpecifiedCommit(destinationPath, commit):
    originalPath = shellCmd("pwd").rstrip('\n')
    os.chdir(destinationPath)
    shellCmd("git checkout %s" % commit)
    os.chdir(originalPath)

def checkoutInstalledBranch(destinationPath):
    rocBLASCommitHash = getInstalledRocBLASCommitHash()
    checkoutSpecifiedCommit(destinationPath, rocBLASCommitHash)

def checkoutMostRecentBranch(destinationPath, branchname):
    originalPath = shellCmd("pwd").rstrip('\n')
    os.chdir(destinationPath)
    shellCmd("git checkout %s" % branchname)
    os.chdir(originalPath)

def convertToExplicitType(t):
    typeMap = {
        'h': 'f16_r',
        's': 'f32_r',
        'd': 'f64_r',
        'c': 'f32_r',
        'z': 'f64_r'}
    if t in typeMap:
        return typeMap[t]
    else:
        return t

def convertArgumentTypesToKernelIdentifier(aType, bType, cType, dType, computeType):
    aType       = convertToExplicitType(aType)
    bType       = convertToExplicitType(bType)
    cType       = convertToExplicitType(cType)
    dType       = convertToExplicitType(dType)
    computeType = convertToExplicitType(computeType)

    argumentsToRocblasType = {
        ('f16_r',  'f16_r',  'f16_r',  'f16_r',  'f16_r') : 'half_precision',
        ('f16_r',  'f16_r',  'f16_r',  'f16_r',  'f32_r') : 'hpa_half_precision',
        ('f32_r',  'f32_r',  'f32_r',  'f32_r',  'f32_r') : 'single_precision',
        ('f64_r',  'f64_r',  'f64_r',  'f64_r',  'f64_r') : 'double_precision',
        ('i8_r',   'i8_r',   'i8_r',   'i8_r',   'i32_r') : 'int8_precision',
        ('bf16_r', 'bf16_r', 'bf16_r', 'bf16_r', 'bf16_r'): 'bf16_precision',
        ('bf16_r', 'bf16_r', 'bf16_r', 'bf16_r', 'f32_r') : 'hpa_bf16_precision',
        ('f16_c',  'f16_c',  'f16_c',  'f16_c',  'f16_c') : 'half_precision_complex',
        ('f16_c',  'f16_c',  'f16_c',  'f16_c',  'f32_c') : 'hpa_half_precision_complex',
        ('f32_c',  'f32_c',  'f32_c',  'f32_c',  'f32_c') : 'single_precision_complex',
        ('f64_c',  'f64_c',  'f64_c',  'f64_c',  'f64_c') : 'double_precision_complex',
        ('i8_c',   'i8_c',   'i8_c',   'i8_c',   'i32_c') : 'int8_precision_complex',
        ('bf16_c', 'bf16_c', 'bf16_c', 'bf16_c', 'bf16_c'): 'bf16_precision_complex',
        ('bf16_c', 'bf16_c', 'bf16_c', 'bf16_c', 'f32_c') : 'hpa_bf16_precision_complex'}
    rocblasTypeToKernelIdentifier = {
        'half_precision'          : 'HB.',
        'hpa_half_precision'      : 'HBH.',
        'single_precision'        : 'SB.',
        'double_precision'        : 'DB.',
        'int8_precision'          : '4xi8BH.',
        'bf16_precision'          : 'BB.',
        'hpa_bf16_precision'      : 'BBH.',
        'single_precision_complex': 'CB.',
        'double_precision_complex': 'ZB.'}

    try:
        rocblasType = argumentsToRocblasType[(aType, bType, cType, dType, computeType)]
        kernelIdentifier = rocblasTypeToKernelIdentifier[rocblasType]
    except KeyError:
        print("Error: Unrecognized argument type combination (a_type %s b_type %s c_type %s d_type %s compute_type %s)" %(aType, bType, cType, dType, computeType))
        return 'NOT VALID'
    return kernelIdentifier
    #Comes from the roblas_common.yaml file

#Gets one of the first or second keys provided
def getOne(d, key1, key2):
    if key1 in d:
        return d[key1]
    else:
        return d[key2]

def supportedProblemType(problemType):
    return 'gemm' in problemType


#Bind as member variable to dict

class ProblemDescription:
    def __init__(self, benchmarkText):
        self.m = 1
        self.n = 1
        self.k = 1
        self.batch_count = 1

        try:
            optDict= parseOptions(
                benchmarkText.split(),
                "m:n:k:f:r:",
                [
                    "batch_count=", "transposeA=", "transposeB=", \
                    "a_type=", "b_type=", "c_type=", "d_type=", "compute_type=",\
                    "precision=", "sizem=", "sizen=", "sizek="\
                ]
            )
            self.gemmType     =     getOne(optDict, 'f','function')
            if not supportedProblemType(self.gemmType):
                return
            self.m            = int(getOne(optDict, 'm', 'sizem'))
            self.n            = int(getOne(optDict, 'n', 'sizen'))
            self.k            = int(getOne(optDict, 'k', 'sizek'))
            transposeA        =            optDict['transposeA'] == 'T'
            transposeB        =            optDict['transposeB'] == 'T'
            if 'ex' in self.gemmType:
                aType         =            optDict['a_type']
                bType         =            optDict['b_type']
                cType         =            optDict['c_type']
                dType         =            optDict['d_type']
                computeType   =            optDict['compute_type']
            else:
                aType         =\
                bType         =\
                cType         =\
                dType         =\
                computeType   =     getOne(optDict, 'r', 'precision')
            self.kernel_flags = convertArgumentTypesToKernelIdentifier(
                aType, bType, cType, dType, computeType)

        except ValueError:
            print("Error: Problem description parameters have invalid type")
        except ParseOptionError:
            print("Error: Input arguments ill formed")

        if 'batch_count' in optDict:
            self.batch_count = int(optDict['batch_count'])

        self.matrix_A = 'Alik' if transposeA else 'Ailk'
        self.matrix_B = 'Bjlk' if transposeB else 'Bljk'

    def __str__(self):
        return "(m=%d n=%d k=%d batch_count=%d transpose_A=%r transpose_B=%r kernel_flags=%s)" \
        % (self.m, self.n, self.k, self.batch_count, self.matrix_A == 'Alik', self.matrix_B == 'Bjlk', self.kernel_flags)

def loadBenchmarkDescriptions(logfilePath):
    try:
        f = open(logfilePath)
    except OSError:
        print("%s file not found." % logfilePath)
        return []

    lines = f.read().split('\n')
    benchmarkList = []

    for line in lines:
        pieces = line.split(maxsplit=1)
        if pieces and "rocblas-bench" in pieces[0]:
            try:
                problem = ProblemDescription(line)
            except KeyError:
                raise KeyError(line)
            if supportedProblemType(problem.gemmType):
                benchmarkList.append(problem)
    return benchmarkList


def matchBetween(matrixFormSet, filename):
    for matrixForm in matrixFormSet:
        if matrixForm[0] in filename and matrixForm[1] in filename and matrixForm[2] in filename:
            return True
    return False

#Search through library logic files in directory to find a matching kernel
def findMatchingKernel(benchDescriptions, architecture, directoryPath):
    #Whether a benchmark description has found a match
    print("Searching %s for logic files..." % directoryPath)
    foundMatch = [False for x in range(len(benchDescriptions))]

    problemSet = set()
    for bench in benchDescriptions:
        problemSet.add((bench.matrix_A, bench.matrix_B, bench.kernel_flags))

    #Filter files by problem types in benchmarks
    fileList = []
    for filename in os.listdir(directoryPath):
        if filename.endswith(".yaml") \
        and architecture in filename \
        and matchBetween(problemSet, filename):
            fileList.append(filename)

    success = False
    #Find a kernel with a matching problem size in the file list
    for filename in fileList:
        foundMatchingKernel = False
        f = open(directoryPath + filename, 'r')
        if f.closed:
            sys.exit("File %s could not be opened" % directoryPath + filename)
        libraryLogic = yaml.safe_load(f)
        f.close()
        logicList = libraryLogic[exactLogicListIdx]
        for i in range(len(benchDescriptions)):
            bench = benchDescriptions[i]
            if bench.matrix_A in filename and bench.matrix_B in filename and bench.kernel_flags in filename:
                #Iterate through kernels to find one with matching size
                for [problemInfo, winningKernelInfo] in logicList:
                    foundMatchingKernel = \
                        problemInfo[mIdx]     == bench.m and \
                        problemInfo[nIdx]     == bench.n and \
                        problemInfo[kIdx]     == bench.k and \
                        problemInfo[batchIdx] == bench.batch_count
                    if foundMatchingKernel: break

                if foundMatchingKernel:
                    print("Match found for %s in file %s" % \
                        (                  bench,      filename))
                    foundMatch[i] = True
    return foundMatch

def findBenchmarkInFile(problemDescriptions):
    pass


def main(argv):
    try:
        optdict = parseOptions(argv, "f:a:c:ul", ["help"])
        if 'help' in optdict.keys():
            print(helpMessage)
            sys.exit()
    except KeyError:
        sys.exit(usageMessage)
    except ParseOptionError:
        sys.exit(usageMessage)

    logpath = ""
    if 'f' in optdict:
        logpath = optdict['f']
    else:
        sys.exit(usageMessage)
    architecture = ""
    if 'a' in optdict:
        architecture = optdict['a']

    directoryPath = mkdtemp()   #"./rocBLASTemp"     #Default path
    libraryPathExtension="/library/src/blas3/Tensile/Logic/asm_full/"
    print("Created temporary directory %s" % directoryPath)
    cloneRepository(directoryPath)

    #Benchmarked problem sizes in log file
    benchDescriptions = loadBenchmarkDescriptions(logpath)
    print("\
-------------------------------------------------------------\n\
-- Finding pre-tuned sizes in specified version of rocblas --\n\
-------------------------------------------------------------")
    if not ('c' in optdict or 'l' in optdict):
        checkoutMostRecentBranch(directoryPath, "master")
    else:
        specifiedCommitHash = optdict['c'] if 'c' in optdict \
                              else getInstalledRocBLASCommitHash()
        checkoutSpecifiedCommit(directoryPath, specifiedCommitHash)
    localMatches = findMatchingKernel(
        benchDescriptions,
        architecture,
        directoryPath+libraryPathExtension)

    if 'u' in optdict:
        print("\
---------------------------------------------------------------\n\
-- Finding pre-tuned sizes in most recent version of rocblas --\n\
---------------------------------------------------------------")
        checkoutMostRecentBranch(directoryPath, "develop")
        recentMatches = findMatchingKernel(
            benchDescriptions,
            architecture,
            directoryPath+libraryPathExtension)

        print("\nMatches in most recent version of rocBLAS but not specified version:")
        for i in range(len(benchDescriptions)):
            if recentMatches[i] and not localMatches[i]:
                print("\t%s" % benchDescriptions[i])


    #Remove temporary directory only if it wasn't provided by user
    print("Removing temporary directory")
    rmtree(directoryPath) #shellCmd("rm -rf %s" % directoryPath)

if __name__=="__main__":
    main(sys.argv[1:])
