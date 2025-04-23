# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import datetime
import csv
import sys
import os
import argparse
from distutils.dir_util import copy_tree
import shutil
import logging
import numpy as np
import json

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD Dataset Analysis Tool"
__credits__ = ["Mike Schmit"]
__license__ = "MIT"
__version__ = "0.9.5"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Alpha"


def readLabelFile(labelFile):
    """Reads the label file and creates a label list

    Args:
        labelFile (string): path of the label file

    Returns:
        tuple: A list of label lines and the length of the list
    """
    with open(labelFile, 'r') as labels:
        labelLines = [l.strip().split(' ', 1)[1] for l in labels]
    return labelLines, len(labelLines)


def readHierarchyFile(hierarchyFile):
    """Reads a hierarchy csv file and converts it to list

    Args:
        hierarchyFile (string): full path of the hierarchy file to read

    Returns:
        tuple: list of elements in hierarchy file and length of the list
    """
    if hierarchyFile != '':
        hierarchyElements = 0
        with open(hierarchyFile) as hierarchy:
            hierarchyCSV = csv.reader(hierarchy)
            hierarchyDataBase = [r for r in hierarchyCSV]
            hierarchyElements = len(hierarchyDataBase)
        return hierarchyDataBase, hierarchyElements


def readInferenceResultFile(inputCSVFile):
    """Reads inference csv file

    Args:
        inputCSVFile (string): path of the file

    Returns:
        tuple: elements of file as list and length of the list
    """
    numElements = 0
    with open(inputCSVFile) as resultFile:
        resultCSV = csv.reader(resultFile)
        next(resultCSV)  # skip header
        resultDataBase = [r for r in resultCSV]
        numElements = len(resultDataBase)
    return resultDataBase, numElements


def copyHtmlAssets(outputDirectory, fileName):
    """Makes required output directories and copies all required html files.
    Copies icons, tablesort file inside util directory, all scripts, styles and
    main html file  inside the output directory

    Args:
        outputDirectory (string): root directory path to copy files to
        fileName (string): name of output directory where the files will be copied to

    Returns:
        tuple: (created output directory path, results directory path)
    """
    # create toolkit with icons and images
    toolKit_Dir = os.path.join(outputDirectory, fileName + '-toolKit')
    toolKit_dir = os.path.expanduser(toolKit_Dir)
    if not os.path.exists(toolKit_dir):
        os.makedirs(toolKit_dir)
    # copy scripts and icons
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fromDirectory = os.path.join(dir_path, 'icons')
    toDirectory = os.path.join(toolKit_dir, 'icons')
    copy_tree(fromDirectory, toDirectory)

    new_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fromDirectory = os.path.join(new_path, 'utils')
    toDirectory = os.path.join(toolKit_dir, 'utils')
    copy_tree(fromDirectory, toDirectory)

    resultsDirectory = os.path.join(toolKit_dir, 'results')
    if not os.path.exists(resultsDirectory):
        os.makedirs(resultsDirectory)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    dirsToCopy = ['scripts', 'styles']
    for srcDir in dirsToCopy:
        fromDirectory = os.path.join(dir_path, 'assets', srcDir)
        toDirectory = os.path.join(toolKit_Dir, srcDir)
        copy_tree(fromDirectory, toDirectory)

    # COpy the main template files
    shutil.copy(os.path.join(dir_path, 'assets', 'templates',
                             'index.html'), os.path.join(toolKit_Dir, 'index.html'))

    return toolKit_dir, resultsDirectory


def copyImages(imagesSource, toolkit_dir):

    dest = os.path.join(toolkit_dir, 'images')
    copy_tree(imagesSource, dest)
    return dest


def generateTop1Result(resultsDirectory, resultDataBase, labelLines):
    logger.debug("Generating top 1 result file")
    outputCsvFile = os.path.join(resultsDirectory, 'results.csv')

    result = {}

    headerText = [
        'Image', 'Ground Truth', 'Top 1' 'Label', 'Match', 'Top 1 Confidence', 'Ground Truth Text', 'Top 1 Label Text']
    fieldNames = ['image', 'gt', 'top1', 'match',
                  'top1_prob', 'gt_text', 'top1_text']

    with open(outputCsvFile, 'w') as csvFile:
        dictWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
        dictWriter.writer.writerow(headerText)

        for imgResult in resultDataBase:
            result = {}
            result['image'] = imgResult[0]
            result['gt'] = int(imgResult[1])
            result['top1'] = int(imgResult[2])
            result['match'] = 'no'
            if result['gt'] == result['top1']:
                result['match'] = 'yes'

            result['gt_text'] = labelLines[result['gt']]
            result['top1_text'] = labelLines[result['top1']]
            result['top1_prob'] = imgResult[7]
            dictWriter.writerow(result)

    logger.debug("Written file %s", outputCsvFile)


def generateComprehensiveResults(resultsDirectory, resultDataBase, labelLines, imageDir, modelName):

    # 0 to 4 are top1 to top5 data
    topk = 5
    logger.debug("Comprehenive csv generation")
    labelSummary = np.zeros(shape=(len(labelLines), topk+2), dtype=int)

    topCounts = [0] * topk
    topTotProb = [0] * topk
    totalMismatch = 0
    totalFailProb = 0
    totalNoGroundTruth = 0
    allImageResults = []

    with open(os.path.join(resultsDirectory, 'resultsComprehensive.csv'), 'w') as csvFile:
        writer = csv.writer(csvFile)

        for img in resultDataBase:
            # imgName = img[0]
            gt = int(img[1])
            labels = [int(x) for x in img[2:2+topk]]
            labelTexts = [labelLines[l] for l in labels]
            probs = [float(x) for x in img[2+topk: 2+topk+topk]]

            if gt >= 0:
                gtLabelText = labelLines[gt]
                match = 0
                for i in range(topk):
                    if gt == labels[i]:
                        match = i+1
                        topCounts[i] += 1
                        topTotProb[i] += probs[i]
                        labelSummary[gt][i+1] += 1
                        labelSummary[gt][0] += 1
                        break
                if match == 0:
                    totalMismatch += 1
                    totalFailProb += probs[0]
                    labelSummary[gt][0] += 1

                if gt != labels[0]:
                    labelSummary[labels[0]][topk+1] += 1
            else:
                match = -1
                gtLabelText = 'Unknown'
                totalNoGroundTruth += 1

            wr = [img]
            wr.extend(labels)
            wr.extend([gt, match])
            wr.extend(labelTexts)
            wr.append(gtLabelText)
            wr.extend(probs)

            # Follwoing syntax can be used in python > 3.5
            # writer.writerow([img[0], *labels, gt, match,
            #                  *labelTexts, gtLabelText, *probs])

            writer.writerow(wr)

            imgRes = {}

            imagePath = 'images/' if imageDir else 'file://'
            imgRes['imageName'] = img[0]
            imgRes['filePath'] = imagePath+img[0]
            imgRes['labels'] = labels
            imgRes['gt'] = gt
            imgRes['match'] = match
            imgRes['labelTexts'] = labelTexts
            imgRes['gtText'] = gtLabelText
            imgRes['probs'] = probs

            allImageResults.append(imgRes)

            # imgRes['filename'] = img[0]
            # if useAbsolutePath == True:
            #     imgRes'filename'

    with open(os.path.join(resultsDirectory, 'imageSummary.js'), 'w') as imageSummaryJson:
        imageSummaryJson.write('var imageSummary = ' +
                               json.dumps(allImageResults))
    logger.debug("Comprehenive csv generation completed")
    logger.debug("Label summary generation started")
    labelSummaryComplete = []

    with open(os.path.join(resultsDirectory, 'labelSummary.csv'), 'w') as labelSummaryCsv:
        writer = csv.writer(labelSummaryCsv)
        for i, label in enumerate(labelLines):

            wr = [i]
            wr.extend(labelSummary[i])
            wr.append(label)
            writer.writerow(wr)

            # writer.writerow([i, *labelSummary[i], label])

            if sum(labelSummary[i]) > 0:
                labelDict = {}
                labelDict['id'] = int(i)
                labelDict['modelName'] = modelName
                labelDict['label'] = label
                labelDict['totalImages'] = int(labelSummary[i][0])
                labelDict['match1'] = int(labelSummary[i][1])
                labelDict['match2'] = int(labelSummary[i][2])
                labelDict['match3'] = int(labelSummary[i][3])
                labelDict['match4'] = int(labelSummary[i][4])
                labelDict['match5'] = int(labelSummary[i][5])
                labelDict['misclassifiedTop1'] = int(labelSummary[i][6])
                labelDict['matchedTop1Per'] = float(float(
                    labelSummary[i][0]) / labelSummary[i][0]) * 100.0 if labelSummary[i][0] > 0 else 0
                labelDict['matchedTop5Per'] = float(float(
                    sum(labelSummary[i][1:6]))/labelSummary[i][0]) * 100.0 if labelSummary[i][0] > 0 else 0
                labelDict['check'] = False

                labelSummaryComplete.append(labelDict)

    with open(os.path.join(resultsDirectory, 'labelSummary.js'), 'w') as labelSummaryJson:
        labelSummaryJson.write('var labelSummary = ' +
                               json.dumps(labelSummaryComplete))

    logger.debug("Label summary generation completed")

    stats = {}
    netSummaryImages = len(resultDataBase) - totalNoGroundTruth
    passProb = sum(topTotProb)
    passCount = sum(topCounts)
    avgPassProb = float(passProb/passCount) if passCount > 0 else 0.
    accuracyPer = (float(passCount) / netSummaryImages) * 100.
    mismatchPer = (float(totalMismatch)/netSummaryImages) * 100.
    avgMismatchProb = 0.0
    if totalMismatch > 0:
        avgMismatchProb = float(totalFailProb)/totalMismatch

    stats['totalImages'] = len(resultDataBase)
    stats['totalMismatch'] = totalMismatch
    stats['passProb'] = passProb
    stats['passCount'] = passCount
    stats['avgPassProb'] = avgPassProb
    stats['accuracyPer'] = accuracyPer
    stats['mismatchPer'] = mismatchPer
    stats['avgMismatchProb'] = avgMismatchProb
    stats['totalFailProb'] = totalFailProb
    stats['netSummaryImages'] = netSummaryImages
    stats['totalNoGroundTruth'] = totalNoGroundTruth

    topKStats = []
    for i in range(topk):
        stat = {}
        stat['matches'] = topCounts[i]
        stat['accuracyPer'] = (float(topCounts[i]) / netSummaryImages) * 100.
        stat['avgPassProb'] = 0
        if topCounts[i] > 0:
            stat['avgPassProb'] = topTotProb[i] / topCounts[i]
        topKStats.append(stat)

    return stats, topCounts, topKStats


def writeResultsJson(resultsDirectory, stats, topCounts, topKStats, modelScores, matchCounts, methodScores, chartData):

    with open(os.path.join(resultsDirectory, 'results_1.js'), 'w') as resJson:
        currentDateString = (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        scores = {}
        scores['modelScores'] = modelScores
        scores['matchCounts'] = matchCounts

        hasHierarchy = False

        if(methodScores):
            scores['method1Scores'] = methodScores[0]
            scores['method2Scores'] = methodScores[1]
            scores['method3Scores'] = methodScores[2]
            hasHierarchy = True

        chartDataDict = {}

        if chartData:
            chartDataDict['passFailData'] = chartData[0]
            chartDataDict['lnPassFailData'] = chartData[1]
            chartDataDict['lnPassFailCombinedData'] = chartData[2]

            chartDataDict['modelScoreChartData'] = methodScores[3]
            chartDataDict['methodScoreChartData'] = methodScores[4]

        jsonString = json.dumps({'stats': stats,
                                 'topCounts': topCounts,
                                 'topKStats': topKStats,
                                 'summaryGenerationDate': currentDateString,
                                 'scores': scores,
                                 'hasHierarchy': hasHierarchy,
                                 'chartData': chartDataDict})

        jsScript = "var data = " + jsonString + ';'
        resJson.write(jsScript)


def writeHierarchyJson(resultsDirectory, topKPassFail, topKHierarchyPassFail):
    with open(os.path.join(resultsDirectory, 'hierarchySummary.js'), 'w') as resJson:
        result = {}
        if(topKPassFail is not None and topKHierarchyPassFail is not None):
            result['topKPassFail'] = topKPassFail.tolist()
            result['topKHierarchyPassFail'] = topKHierarchyPassFail.tolist()

        jsScript = "var hierarchyData = " + json.dumps(result) + ';'
        resJson.write(jsScript)


def generateCompareResultSummary(toolKit_dir, modelName, dataFolder, stats):
    # Compare result summary
    summaryFileName = ''
    folderName = os.path.expanduser("~/.adatCompare")
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    modelFolderName = os.path.join(folderName, modelName)

    if not os.path.exists(modelFolderName):
        os.makedirs(modelFolderName)

    summaryFileName = os.path.join(folderName, 'modelRunHistoryList.json')
    stats['modelName'] = modelName or 'Generic Model'
    stats['genDate'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(summaryFileName):
        with open(summaryFileName, 'r') as f:
            data = json.load(f)
            data.append(stats)
    else:
        data = [stats]

    with open(summaryFileName, 'w') as f:
        json.dump(data, f)

    with open(os.path.join(toolKit_dir, 'results', 'resultHistory.js'), 'w') as f:
        f.write("var modelHistories="+json.dumps(data))


def processHierarchy(resultDataBase, labelLines, hierarchyDataBase):
    topk = 5
    topKPassFail = np.zeros(shape=(100, 2))
    topKHierarchyPassFail = np.zeros(shape=(100, 12))
    for img in resultDataBase:
        # imgName = img[0]
        gt = int(img[1])
        labels = [int(x) for x in img[2:2+topk]]
        # labelTexts = [labelLines[l] for l in labels]
        probs = [float(x) for x in img[2+topk: 2+topk+topk]]

        if gt >= 0:
            if gt == labels[0]:
                count = 0
                f = 0.0
                while f < 1:
                    if probs[0] < (f+0.01) and probs[0] > f:
                        topKPassFail[count][0] += 1
                        topKHierarchyPassFail[count][0] += 1
                        topKHierarchyPassFail[count][2] += 1
                        topKHierarchyPassFail[count][4] += 1
                        topKHierarchyPassFail[count][6] += 1
                        topKHierarchyPassFail[count][8] += 1
                        topKHierarchyPassFail[count][10] += 1
                    count += 1
                    f += 0.01
            else:
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] < (f + 0.01)) and probs[0] > f):
                        topKPassFail[count][1] += 1
                        truthHierarchy = hierarchyDataBase[gt]
                        resultHierarchy = hierarchyDataBase[labels[0]]
                        token_result = ''
                        token_truth = ''
                        previousTruth = 0
                        catCount = 0
                        while catCount < 6:
                            token_truth = truthHierarchy[catCount]
                            token_result = resultHierarchy[catCount]
                            if((token_truth != '') and (token_truth == token_result)):
                                topKHierarchyPassFail[count][catCount*2] += 1
                                previousTruth = 1
                            elif((previousTruth == 1) and (token_truth == '' and token_result == '')):
                                topKHierarchyPassFail[count][catCount*2] += 1
                            else:
                                topKHierarchyPassFail[count][catCount*2 + 1] += 1
                                previousTruth = 0
                            catCount += 1
                    count += 1
                    f += 0.01
    return topKPassFail, topKHierarchyPassFail


def generateHierarchySummary(resultsDirectory, topKPassFail, topKHierarchyPassFail):
    logger.debug("hierarchySummary.csv generation ..")
    orig_stdout = sys.stdout
    sys.stdout = open(resultsDirectory+'/hierarchySummary.csv', 'w')
    print("Probability,Pass,Fail,cat-1 pass,cat-1 fail,cat-2 pass, cat-2 fail,"
          "cat-3 pass,cat-3 fail,cat-4 pass,cat-4 fail,cat-5 pass,cat-5 fail,cat-6 pass,cat-6 fail")
    i = 99
    f = 0.99
    while i >= 0:
        print("%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (f, topKPassFail[i][0], topKPassFail[i][1],
                                                                                          topKHierarchyPassFail[i][0], topKHierarchyPassFail[
                                                                                              i][1], topKHierarchyPassFail[i][2],
                                                                                          topKHierarchyPassFail[i][3], topKHierarchyPassFail[
                                                                                              i][4], topKHierarchyPassFail[i][5],
                                                                                          topKHierarchyPassFail[i][6], topKHierarchyPassFail[
                                                                                              i][7], topKHierarchyPassFail[i][8],
                                                                                          topKHierarchyPassFail[i][9], topKHierarchyPassFail[i][10], topKHierarchyPassFail[i][11]))
        f = f - 0.01
        i = i - 1

    sys.stdout = orig_stdout
    logger.debug("hierarchySummary.csv generated ..")


def calculateHierarchyPenalty(truth, result, hierarchyDataBase):
    if hierarchyDataBase == None:
        return 0

    penaltyValue = 0
    penaltyMultiplier = 0
    truthHierarchy = hierarchyDataBase[truth]
    resultHierarchy = hierarchyDataBase[result]
    token_result = ''
    token_truth = ''
    previousTruth = 0
    catCount = 0

    while catCount < 6:
        token_truth = truthHierarchy[catCount]
        token_result = resultHierarchy[catCount]
        if((token_truth != '') and (token_truth == token_result)):
            previousTruth = 1
        elif((previousTruth == 1) and (token_truth == '' and token_result == '')):
            previousTruth = 1
        else:
            previousTruth = 0
            penaltyMultiplier += 1
        catCount += 1

    penaltyMultiplier = float(penaltyMultiplier - 1)
    penaltyValue = (0.2 * penaltyMultiplier)

    return penaltyValue


def createScoreSummary(stats, topCounts):
    logger.debug('Calculating scores')

    topk = 5
    topScores = [float(t) for t in topCounts]
    modelScores = [0.0] * topk
    matchCounts = [0] * topk

    for i in range(topk):
        modelScores[i] = (sum(topScores[:i+1]) /
                          stats['netSummaryImages']) * 100.0
        matchCounts[i] = sum(topCounts[:i+1])

    return modelScores, matchCounts


def createHirerchySummaryScore(stats, topCounts, resultDataBase, labelLines, hierarchyDataBase, topKPassFail):
    topk = 5

    hierarchyPenalty = np.zeros(shape=(100, topk))
    top5PassFail = np.zeros(shape=(100, topk*2))

    # method1ModelScores = []
    # method2ModelScores = []
    # method3ModelScores = []
    # methodStandardaModelScores = []

    for img in resultDataBase:
        # imgName = img[0]
        gt = int(img[1])
        labels = [int(x) for x in img[2:2+topk]]
        # labelTexts = [labelLines[l] for l in labels]
        probs = [float(x) for x in img[2+topk: 2+topk+topk]]

        if(gt >= 0):
            if(gt == labels[0]):
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][0] += 1
                    count += 1
                    f += 0.01

            elif(gt == labels[1]):
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][1] += 1
                        hierarchyPenalty[count][0] += calculateHierarchyPenalty(
                            gt, labels[0], hierarchyDataBase)
                    if((probs[1] <= (f + 0.01)) and probs[1] > f):
                        top5PassFail[count][2] += 1
                    count += 1
                    f += 0.01
            elif(gt == labels[2]):
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][1] += 1
                        hierarchyPenalty[count][0] += calculateHierarchyPenalty(
                            gt, labels[0], hierarchyDataBase)
                    if((probs[1] <= (f + 0.01)) and probs[1] > f):
                        top5PassFail[count][3] += 1
                        hierarchyPenalty[count][1] += calculateHierarchyPenalty(
                            gt, labels[1], hierarchyDataBase)
                    if((probs[2] <= (f + 0.01)) and probs[2] > f):
                        top5PassFail[count][4] += 1
                    count += 1
                    f += 0.01

            elif(gt == labels[3]):
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][1] += 1
                        hierarchyPenalty[count][0] += calculateHierarchyPenalty(
                            gt, labels[0], hierarchyDataBase)
                    if((probs[1] <= (f + 0.01)) and probs[1] > f):
                        top5PassFail[count][3] += 1
                        hierarchyPenalty[count][1] += calculateHierarchyPenalty(
                            gt, labels[1], hierarchyDataBase)
                    if((probs[2] <= (f + 0.01)) and probs[2] > f):
                        top5PassFail[count][5] += 1
                        hierarchyPenalty[count][2] += calculateHierarchyPenalty(
                            gt, labels[2], hierarchyDataBase)
                    if((probs[3] <= (f + 0.01)) and probs[3] > f):
                        top5PassFail[count][6] += 1
                    count += 1
                    f += 0.01
            elif(gt == labels[4]):
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][1] += 1
                        hierarchyPenalty[count][0] += calculateHierarchyPenalty(
                            gt, labels[0], hierarchyDataBase)
                    if((probs[1] <= (f + 0.01)) and probs[1] > f):
                        top5PassFail[count][3] += 1
                        hierarchyPenalty[count][1] += calculateHierarchyPenalty(
                            gt, labels[1], hierarchyDataBase)
                    if((probs[2] <= (f + 0.01)) and probs[2] > f):
                        top5PassFail[count][5] += 1
                        hierarchyPenalty[count][2] += calculateHierarchyPenalty(
                            gt, labels[2], hierarchyDataBase)
                    if((probs[3] <= (f + 0.01)) and probs[3] > f):
                        top5PassFail[count][7] += 1
                        hierarchyPenalty[count][3] += calculateHierarchyPenalty(
                            gt, labels[3], hierarchyDataBase)
                    if((probs[4] <= (f + 0.01)) and probs[4] > f):
                        top5PassFail[count][8] += 1
                    count += 1
                    f += 0.01
            else:
                count = 0
                f = 0
                while f < 1:
                    if((probs[0] <= (f + 0.01)) and probs[0] > f):
                        top5PassFail[count][1] += 1
                        hierarchyPenalty[count][0] += calculateHierarchyPenalty(
                            gt, labels[0], hierarchyDataBase)
                    if((probs[1] <= (f + 0.01)) and probs[1] > f):
                        top5PassFail[count][3] += 1
                        hierarchyPenalty[count][1] += calculateHierarchyPenalty(
                            gt, labels[1], hierarchyDataBase)
                    if((probs[2] <= (f + 0.01)) and probs[2] > f):
                        top5PassFail[count][5] += 1
                        hierarchyPenalty[count][2] += calculateHierarchyPenalty(
                            gt, labels[2], hierarchyDataBase)
                    if((probs[3] <= (f + 0.01)) and probs[3] > f):
                        top5PassFail[count][7] += 1
                        hierarchyPenalty[count][3] += calculateHierarchyPenalty(
                            gt, labels[3], hierarchyDataBase)
                    if((probs[4] <= (f + 0.01)) and probs[4] > f):
                        top5PassFail[count][9] += 1
                        hierarchyPenalty[count][4] += calculateHierarchyPenalty(
                            gt, labels[4], hierarchyDataBase)
                    count += 1
                    f += 0.01

    topKPassScore = [0] * topk
    topKFailScore = [0] * topk
    topKHierarchyPenalty = [0] * topk

    confID = 0.99
    i = 99
    # passIndex = 0
    # failIndex = 1
    while confID >= 0:
        for j in range(topk):
            # Every even index is pass score, every odd is fail score
            topKPassScore[j] += confID * top5PassFail[i][j*2]
            topKFailScore[j] += confID * top5PassFail[i][j*2+1]

            topKHierarchyPenalty[j] += hierarchyPenalty[i][j]

        confID -= 0.01
        i = i-1

    netSummaryImages = stats['netSummaryImages']
    # Method 1 scores
    method1Score = [0]*topk
    method2Score = [0]*topk
    method3Score = [0]*topk

    for i in range(topk):
        topKPassScoreSum = float(sum(topKPassScore[:i+1]))
        method1Score[i] = (topKPassScoreSum / netSummaryImages)*100.0
        method2Score[i] = (
            (topKPassScoreSum - topKFailScore[i])/netSummaryImages) * 100.0
        method3Score[i] = ((topKPassScoreSum -
                            (topKFailScore[i] + topKHierarchyPenalty[i]))/netSummaryImages) * 100.0

    # TODO: Refactor This ------------------------
    # Scores here are same as calculated above
    # This section is kept as it is-------------
    top5ModelScore = np.zeros(shape=(100, 20))

    standardPassTop1 = 0
    standardPassTop2 = 0
    standardPassTop3 = 0
    standardPassTop4 = 0
    standardPassTop5 = 0

    Top1PassScore = 0
    Top1FailScore = 0
    Top2PassScore = 0
    Top2FailScore = 0
    Top3PassScore = 0
    Top3FailScore = 0
    Top4PassScore = 0
    Top4FailScore = 0
    Top5PassScore = 0
    Top5FailScore = 0
    Top1HierarchyPenalty = 0
    Top2HierarchyPenalty = 0
    Top3HierarchyPenalty = 0
    Top4HierarchyPenalty = 0
    Top5HierarchyPenalty = 0

    confID = 0.99
    i = 99
    while i >= 0:
        Top1PassScore += confID * topKPassFail[i][0]
        Top1FailScore += confID * topKPassFail[i][1]
        Top2PassScore += confID * top5PassFail[i][2]
        Top2FailScore += confID * top5PassFail[i][3]
        Top3PassScore += confID * top5PassFail[i][4]
        Top3FailScore += confID * top5PassFail[i][5]
        Top4PassScore += confID * top5PassFail[i][6]
        Top4FailScore += confID * top5PassFail[i][7]
        Top5PassScore += confID * top5PassFail[i][8]
        Top5FailScore += confID * top5PassFail[i][9]

        Top1HierarchyPenalty += hierarchyPenalty[i][0]
        Top2HierarchyPenalty += hierarchyPenalty[i][1]
        Top3HierarchyPenalty += hierarchyPenalty[i][2]
        Top4HierarchyPenalty += hierarchyPenalty[i][3]
        Top5HierarchyPenalty += hierarchyPenalty[i][4]

        # method 1
        top5ModelScore[i][1] = (float(Top1PassScore)/netSummaryImages)*100
        top5ModelScore[i][3] = (
            float(Top1PassScore + Top2PassScore)/netSummaryImages)*100
        top5ModelScore[i][5] = (
            float(Top1PassScore + Top2PassScore + Top3PassScore)/netSummaryImages)*100
        top5ModelScore[i][7] = (float(
            Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore)/netSummaryImages)*100
        top5ModelScore[i][9] = (float(Top1PassScore + Top2PassScore +
                                      Top3PassScore + Top4PassScore + Top5PassScore)/netSummaryImages)*100

        # method 2
        top5ModelScore[i][0] = (
            float(Top1PassScore - Top1FailScore)/netSummaryImages)*100
        top5ModelScore[i][4] = (float(
            (Top1PassScore + Top2PassScore + Top3PassScore) - Top3FailScore)/netSummaryImages)*100
        top5ModelScore[i][6] = (float((Top1PassScore + Top2PassScore +
                                       Top3PassScore + Top4PassScore) - Top4FailScore)/netSummaryImages)*100
        top5ModelScore[i][8] = (float((Top1PassScore + Top2PassScore + Top3PassScore +
                                       Top4PassScore + Top5PassScore) - Top5FailScore)/netSummaryImages)*100

        # method 3
        top5ModelScore[i][15] = (float(
            Top1PassScore - (Top1FailScore + Top1HierarchyPenalty))/netSummaryImages)*100
        top5ModelScore[i][16] = (float((Top1PassScore + Top2PassScore) -
                                       (Top2FailScore + Top2HierarchyPenalty))/netSummaryImages)*100
        top5ModelScore[i][17] = (float((Top1PassScore + Top2PassScore + Top3PassScore) - (
            Top3FailScore + Top3HierarchyPenalty))/netSummaryImages)*100
        top5ModelScore[i][18] = (float((Top1PassScore + Top2PassScore + Top3PassScore +
                                        Top4PassScore) - (Top4FailScore + Top4HierarchyPenalty))/netSummaryImages)*100
        top5ModelScore[i][19] = (float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore +
                                        Top5PassScore) - (Top5FailScore + Top5HierarchyPenalty))/netSummaryImages)*100

        # standard method
        standardPassTop1 += float(topKPassFail[i][0])
        standardPassTop2 += float(topKPassFail[i][0] + top5PassFail[i][2])
        standardPassTop3 += float(topKPassFail[i][0] +
                                  top5PassFail[i][2] + top5PassFail[i][4])
        standardPassTop4 += float(topKPassFail[i][0] + top5PassFail[i]
                                  [2] + top5PassFail[i][4] + top5PassFail[i][6])
        standardPassTop5 += float(topKPassFail[i][0] + top5PassFail[i][2] +
                                  top5PassFail[i][4] + top5PassFail[i][6] + top5PassFail[i][8])
        top5ModelScore[i][10] = (standardPassTop1/netSummaryImages)*100
        top5ModelScore[i][11] = (standardPassTop2/netSummaryImages)*100
        top5ModelScore[i][12] = (standardPassTop3/netSummaryImages)*100
        top5ModelScore[i][13] = (standardPassTop4/netSummaryImages)*100
        top5ModelScore[i][14] = (standardPassTop5/netSummaryImages)*100
        confID = confID - 0.01
        i = i - 1
    # end of section graph data calculation

    # Make arrays for chart plotting ----------------------------
    modelScoreChartData = [[[1, 0, 0, 0, 0]] for i in range(topk)]
    modelChartDataIndices = [[10, 1, 0, 15],
                             [11, 3, 2, 16],
                             [12, 5, 4, 17],
                             [13, 7, 6, 18],
                             [14, 9, 8, 19]]

    scoreMethodChartData = [[[1, 0, 0, 0, 0, 0]] for i in range(4)]
    scoreMethodChartIndices = [[10, 11, 12, 13, 14],  # Standard
                               [1, 3, 5, 7, 9],  # method 1
                               [0, 2, 4, 6, 8],  # method 2
                               [15, 16, 17, 18, 19]]  # method3

    fVal = 0.99
    i = 99
    while i >= 0:
        for j in range(topk):
            modelScoreDataFromIndices = [top5ModelScore[i][k]
                                         for k in modelChartDataIndices[j]]

            modelScoreChartData[j].append(
                [fVal].extend(modelScoreDataFromIndices))

            # modelScoreChartData[j].append([fVal, *modelScoreDataFromIndices])

        # DIfferent methods now - Standard, method1, method2, method3

        for j in range(4):
            scoreMethodFromIndices = [top5ModelScore[i][k]
                                      for k in scoreMethodChartIndices[j]]

            scoreMethodChartData[j].append(
                [fVal].extend(scoreMethodFromIndices))

            # scoreMethodChartData[j].append([fVal, *scoreMethodFromIndices])

        fVal = round(fVal-0.01, 2)
        i = i - 1

    return method1Score, method2Score, method3Score, modelScoreChartData, scoreMethodChartData


def getSuccessFailureChartData(stats, topKPassFail, topKHierarchyPassFail):

    fVal = 0.99
    sumPass = 0.0
    sumFail = 0.0
    netSummaryImages = stats['netSummaryImages']
    i = 99
    passFailData = [[1, 0, 0]]

    lnPassFailData = [[[1, 0, 0]], [[1, 0, 0]], [
        [1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]]
    lnSumPass = [0.0]*6
    lnSumFail = [0.0]*6

    lnCombinedPassFailData = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    while i >= 0:
        sumPass = float(sumPass + topKPassFail[i][0])
        sumFail = float(sumFail + topKPassFail[i][1])
        passFailData.append(
            [fVal, sumPass/netSummaryImages, sumFail/netSummaryImages])
        dTemp = [fVal]

        for j in range(6):
            lnSumPass[j] = float(lnSumPass[j] + topKHierarchyPassFail[i][j*2])
            lnSumFail[j] = float(
                lnSumFail[j] + topKHierarchyPassFail[i][j*2+1])

            lnPassFailData[j].append(
                [fVal, lnSumPass[j]/netSummaryImages, lnSumFail[j]/netSummaryImages])

            dTemp.append(float(lnSumPass[j]/netSummaryImages))
            dTemp.append(float(lnSumFail[j]/netSummaryImages))

        lnCombinedPassFailData.append(dTemp)

        fVal = round(fVal - 0.01, 2)
        i = i-1

    return passFailData, lnPassFailData, lnCombinedPassFailData


def main():
    # AMD Data Analysis Toolkit - Classification
    logger.debug("AMD Data Analysis Toolkit - Classification")

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_results',  type=str, required=True,
                        help='input inference results CSV file          [required] (File Format:ImgFileName, GroundTruth, L1, L2, L3, L4, L5, P1, P2, P3, P4, P5)')
    parser.add_argument('--image_dir',          type=str, required=False,
                        help='input image directory used in inference   [optional]')
    parser.add_argument('--label',              type=str, required=True,
                        help='input labels text file                    [required]')
    parser.add_argument('--hierarchy',          type=str, default='',
                        help='input AMD proprietary hierarchical file   [optional]')
    parser.add_argument('--model_name',         type=str, default='',
                        help='input inferece model name                 [optional]')
    parser.add_argument('--output_dir',         type=str, required=True,
                        help='output dir to store ADAT results          [required]')
    parser.add_argument('--output_name',        type=str, required=True,
                        help='output ADAT file name                     [required]')
    args = parser.parse_args()

    modelName = args.model_name if args.model_name else 'Generic Model'

    inputCSVFile = args.inference_results
    inputImageDirectory = args.image_dir
    labelFile = args.label
    hierarchyFile = args.hierarchy
    modelName = args.model_name
    outputDirectory = args.output_dir
    fileName = args.output_name

    if inputImageDirectory:
        if not os.path.exists(args.image_dir):
            print("ERROR: Invalid Input Image Directory")
            exit()

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # read inference results file
    resultDataBase, _ = readInferenceResultFile(inputCSVFile)

    # read label file
    labelLines, labelElements = readLabelFile(labelFile)

    if args.hierarchy:
        hierarchyDataBase, hierarchyElements = readHierarchyFile(
            hierarchyFile)
        if hierarchyElements != labelElements:
            print("ERROR Invalid Hierarchy file / label File")
            exit()

    toolkit_dir, resultsDirectory = copyHtmlAssets(outputDirectory, fileName)
    imageDir = None
    if inputImageDirectory:
        imageDir = copyImages(inputImageDirectory, toolkit_dir)

    generateTop1Result(resultsDirectory, resultDataBase, labelLines)

    stats, topCounts, topKStats = generateComprehensiveResults(
        resultsDirectory, resultDataBase, labelLines, imageDir, modelName)

    generateCompareResultSummary(toolkit_dir, modelName, 'images', stats)

    methodScores = None
    chartData = None

    # Check if hierarchy option is given, if it is given process hierarchy and generate script files as needed
    if args.hierarchy:
        hierarchyDataBase, hierarchyElements = readHierarchyFile(hierarchyFile)
        if hierarchyElements != labelElements:
            print("ERROR Invalid Hierarchy file / label File")
            exit()

        topKPassFail, topKHierarchyPassFail = processHierarchy(
            resultDataBase, labelLines, hierarchyDataBase)

        generateHierarchySummary(
            resultsDirectory, topKPassFail, topKHierarchyPassFail)

        methodScores = createHirerchySummaryScore(
            stats, topCounts, resultDataBase, labelLines, hierarchyDataBase, topKPassFail)

        chartData = getSuccessFailureChartData(
            stats, topKPassFail, topKHierarchyPassFail)


             # Write hierarchy json, if no hierarchy creates an empty file
        writeHierarchyJson(resultsDirectory, topKPassFail, topKHierarchyPassFail)
    else:
         writeHierarchyJson(resultsDirectory, None, None)

    modelScores, matchCounts = createScoreSummary(stats, topCounts)

    # Write to result json
    writeResultsJson(resultsDirectory, stats, topCounts, topKStats,
                     modelScores, matchCounts, methodScores, chartData, )

   

    logger.debug("HTML generation complete")


if __name__ == '__main__':
    main()
