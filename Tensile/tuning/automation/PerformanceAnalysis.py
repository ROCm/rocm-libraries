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

import os
import sys
import argparse
import re
from distutils.util import strtobool

import pandas as pd
from ExtractSizes import *
from TuningConfiguration import *

headers = ""

def MatchLine(headerPattern, linePattern, line):

    global headers

    if not headers:
        matched = headerPattern.match(line)

        if matched:
            headers = line

        return matched
    else:
        matched = linePattern.match(line)

        return matched

def ResultsFilesList(inputPath, resultsName):

    resultsFilePattern = re.compile(resultsName + "\.[0-9]*")
    resultsFiles = [f for f in os.listdir(inputPath)]

    filteredFiles = [f for f in resultsFiles if resultsFilePattern.match(f)]

    return filteredFiles

def ParseResults(inputPath, outputPath, resultsName):

    global headers

    headers = ""

    filteredFiles = ResultsFilesList(inputPath, resultsName)

    headerPattern = re.compile("transA,transB")
    linePattern = re.compile(r"(N|T),(N|T).*")

    outfilename = resultsName + ".csv"

    outputFilePath = os.path.join(outputPath, outfilename)
    outfile = open(outputFilePath,'w')

    for fl in filteredFiles:
        flPath = os.path.join(inputPath,fl)
        filteredLines = [ line for line in open(flPath) if MatchLine(headerPattern, linePattern, line)]
        outfile.writelines(filteredLines)
    outfile.flush()
    outfile.close()

def getMultiplier(xdl):

    if xdl:
        return 2

    return 1

def getCuCount(gpu):

    gpuMap = {'vega10':64, 'mi25':64, 'vega20':64, 'v340l':56,'mi50':60,'arcturus':120,'mi60':64}
    for key in gpuMap.keys():
        if gpu == key:
            return gpuMap[key]

    return 64

def fillCallCounts(problemMapper, callCounts, callCount, callCountStrided, isOne):

    for i in problemMapper:
        for klist in i:
            midList = list()
            for key in klist:
                if key == "transposeA" or key == "transposeB" or key == "f" or key == "i":
                    if klist[key] == 10 and isOne:
                        klist[key] = 1
                    midList.append(klist[key])
                if len(midList) == 4:
                    callCounts.append(midList)

    for line in callCounts:
        if line[0] == "gemm" or line[0] == "gemm_ex":
            callCount.append(line[3])
        elif "gemm_strided_batched" in line[0]:
            callCountStrided.append(line[3])

def chooseCallCount(resultsName, callCount, callCountStrided):
    if "strided" in resultsName:
        return callCountStrided
    return callCount

def ProcessResults(outputPath, resultsName, freqM, sz, call_count, gpu = 'vega20', xdl = False):

    global headers
    resultsFilename = resultsName + ".csv"

    resultsFilePath = os.path.join(outputPath, resultsFilename)

    data = None
    data = pd.read_csv(resultsFilePath)
    multiplier = getMultiplier(xdl)
    cus = getCuCount(gpu)

    headerValues = headers.strip().split(",")
    headerLength = len(headerValues)
    key = headerValues[0:headerLength-2]
    key.append('us')

    performanceField = "rocblas-Gflops"
    timingField = "us"

    df = data.groupby(key,sort=False)

    results = df[performanceField].mean().to_frame()
    timingResults = df[timingField].mean().to_frame()

    freq=freqM
    factor=sz * 64 * multiplier * cus
    results['eff'] = 100*1e3*results['rocblas-Gflops'] / (factor * freq)
    results['us_w'] = timingResults['us']*call_count

    aggregateFileName = resultsName + "-aggregated.csv"
    aggregateFilePath = os.path.join(outputPath, aggregateFileName)

    results.to_csv(aggregateFilePath, header=True)

    resultsBad = results[results['eff'] < 70]
    badResultsFileName = resultsName + "-bad.csv"
    badResultsFilePath = os.path.join(outputPath, badResultsFileName)
    resultsBad.sort_values(by='us_w',ascending=False).to_csv(badResultsFilePath, header=True)

    large1 = data
    large1['N'] = pd.to_numeric(large1['N'])
    large1['M'] = pd.to_numeric(large1['M'])
    large2 = large1[large1['N']>1000]
    large = large2[large2['M']>1000]

    largeAgg = large.groupby(key)
    largeResults = largeAgg[performanceField].mean().to_frame()
    largeResultsTime = largeAgg[timingField].mean().to_frame()
    largeResults['eff'] = 100*1e3*largeResults['rocblas-Gflops'] / (factor * freq)
    largeResults['us_w'] = largeResultsTime['us']

    resultsFileName = resultsName + "-large.csv"
    resultsFilePath = os.path.join(outputPath, resultsFileName)
    largeResults.sort_values(by='us_w',ascending=False).to_csv(resultsFilePath, header=True)

    resultsBad = largeResults[largeResults['eff'] < 70]
    badResultsFileName = resultsName + "-bad-large.csv"
    badResultsFilePath = os.path.join(outputPath, badResultsFileName)
    resultsBad.sort_values(by='eff',ascending=True).to_csv(badResultsFilePath, header=True)

def RunMain():

    userArgs = sys.argv[1:]

    def strbool(arg):
        return bool(strtobool(arg))

    argParser = argparse.ArgumentParser()
    argParser.add_argument("input_path", help="path where the results are located")
    argParser.add_argument("output_path", help="path where the processed files are to go")
    argParser.add_argument("frequency", help="frequecy in megahertz used in testing", type=int,default=1301)
    argParser.add_argument("data_size", help="data size",type=int,default=2)
    argParser.add_argument("input_file_name", help="configuration file path")
    argParser.add_argument("gpu", help="which gpu was used", type=str,default="vega20")
    argParser.add_argument("mfma", help="were mfma instructions enabled", type=strbool,default=False)
    argParser.add_argument("is_count_1", help="were mfma instructions enabled", type=strbool,default=False) # duplicated parameter?

    args = argParser.parse_args(userArgs)

    inputPath = args.input_path
    outputPath = args.output_path
    freqM = args.frequency
    sz = args.data_size
    inputFileName = args.input_file_name
    cu = args.gpu
    xdl = args.mfma
    isOne = args.is_count_1

    problemMapper = list(ProcessFile(inputFileName).values())
    callCounts = list(list())
    callCount = list()
    callCountStrided = list()

    fillCallCounts(problemMapper, callCounts, callCount, callCountStrided, isOne)

    resultsFiles = [f for f in os.listdir(inputPath) if (os.path.isfile(os.path.join(inputPath, f)))]
    resultsNameSet = set()

    for resultsFile in resultsFiles:
        resultsName, _ = os.path.splitext(resultsFile)
        resultsNameSet.add(resultsName)

    resultsNames = list(resultsNameSet)

    for resultsName in resultsNames:
        ParseResults(inputPath, outputPath, resultsName)
        callCountChoice = chooseCallCount(resultsName, callCount, callCountStrided)
        ProcessResults(outputPath, resultsName, freqM, sz, callCountChoice, cu, xdl)


if __name__ == "__main__":
    RunMain()
