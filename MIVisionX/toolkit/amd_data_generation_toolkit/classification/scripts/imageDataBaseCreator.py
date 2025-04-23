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

import os
import getopt
import sys
#from PIL import Image

opts, args = getopt.getopt(sys.argv[1:], 'd:o:f:w:h:p:c:')
 
directory = ''
outputDir = ''
fileName = ''
width = ''
height = ''
padVal = ''
count = ''

# get user variables
for opt, arg in opts:
    if opt == '-d':
        directory = arg
    elif opt == '-o':
        outputDir = arg
    elif opt == '-f':
        fileName = arg
    elif opt == '-w':
        width = arg
    elif opt == '-h':
        height = arg
    elif opt == '-p':
        padVal = arg
    elif opt == '-c':
        count = arg

# set defaults
if width == '':
    width = '-1'
if height == '':
    height = '-1'
if padVal == '':
    padVal = '-1'
if count == '':
    count = '-1'

# error check and script help
if fileName == '' or directory == '' or outputDir == '':
    print('Invalid command line arguments.\nUsage python imageDataBaseCreator.py \t\t\t\t-d [input image directory - required]\n ' \
          '\t\t\t\t-o [output image directory - required] \n\t\t\t\t-f [new image file name - required]\n '\
	      '\t\t\t\t-w [resize width - optional]\n \t\t\t\t-h [resize height  - optional]\n '\
          '\t\t\t\t-p [padding value  - optional]\n \t\t\t\t-c [image start number  - optional]\n')
    exit()

# set log output directory
logDir = fileName+'-scriptOutput'

# run step 1, 2, & 3 scripts to generate the dataBase
os.system('python step-1.py -d '+directory+' -o '+outputDir+' -f '+fileName +' -w '+ width +' -h '+height+' -p ' +padVal+' -c ' +count);
os.system('python step-2.py -d '+outputDir+' -f tag_file_name.txt');
os.system('python step-3.py -l script-labels.txt -t csv_tag_file_name.txt >> '+fileName+'-val.txt');

# remove  and copy system files and duplicates
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    print('imageDataBaseCreator.py Linux Detected')
    os.system('rm -rf tag_file_name.txt output.log');
    os.system('mv csv_tag_file_name.txt '+logDir+'/csv_tag_file_name.csv');
    os.system('cp '+fileName+'-val.txt '+logDir+'/'+fileName+'-val.txt');
elif _platform == "win32" or _platform == "win64":
    print('imageDataBaseCreator.py Windows Detected')
    os.system('DEL tag_file_name.txt output.log');
    os.system('rm -rf tag_file_name.txt output.log');
    os.system('mv csv_tag_file_name.txt '+logDir+'/csv_tag_file_name.csv');
    os.system('cp '+fileName+'-val.txt '+logDir+'/'+fileName+'-val.txt');

# generate error reports
orig_stdout = sys.stdout
sys.stdout = open(logDir+'/'+fileName+'-fileNameWithLabels.csv','a')
print('Old FileName, New FileName, Labels')
sys.stdout = open(logDir+'/'+fileName+'-fileNameWithErrors.csv','a')
print('Old FileName, New FileName')
sys.stdout = open(logDir+'/'+fileName+'-multipleLabelsFile.csv','a')
print('Old FileName, New FileName, Labels')
sys.stdout = open(logDir+'/'+fileName+'-invalidLabelsFile.csv','a')
print('Old FileName, New FileName, Labels')


# generate reports for images with and without labels
with open(logDir+'/'+fileName+'-FileNameTranslation.csv') as orig_filename:
    for file in orig_filename:
        fileList = file.strip().split(",")
        imgFileCount = 0
        sys.stdout = open(logDir+'/'+fileName+'-fileNameWithLabels.csv','a')
        with open(logDir+'/csv_tag_file_name.csv') as tagFile:
            for tag in tagFile:
                tagList = tag.strip().split(",")
                fileList[1] = fileList[1].strip()
                tagList[0] = tagList[0].strip()

                if fileList[1] == tagList[0]:
                    imgFileCount += 1
                    tag = tag.strip()
                    print(fileList[0]+','+tag)

        if imgFileCount == 0:
            sys.stdout = open(logDir+'/'+fileName+'-fileNameWithErrors.csv','a')
            print(fileList[0]+','+fileList[1])


# generate report for multiple labels for single image
with open(logDir+'/'+fileName+'-FileNameTranslation.csv') as orig_filename:
    for file in orig_filename:
        fileList = file.strip().split(",")
        multipleLabelFlag = 0
        sys.stdout = open(logDir+'/'+fileName+'-fileNameWithLabels.csv','a')
        with open(logDir+'/'+fileName+'-val.txt') as tagFile:
            for tag in tagFile:
                tagList = tag.strip().split(" ")
                fileList[0] = fileList[0].strip()
                fileList[1] = fileList[1].strip()
                tagList[0] = tagList[0].strip()
                tagList[1] = tagList[1].strip()

                if fileList[1] == tagList[0]:
                    multipleLabelFlag += 1

                if multipleLabelFlag > 1:
                    multipleLabelFlag = 0
                    sys.stdout = open(logDir+'/'+fileName+'-multipleLabelsFile.csv','a')
                    print(fileList[0]+','+tagList[0]+','+tagList[1])


# generate report for invalid labels for single image
sys.stdout = open(logDir+'/'+fileName+'-invalidLabelsFile.csv','a')
with open(logDir+'/'+fileName+'-FileNameTranslation.csv') as orig_filename:
    for file in orig_filename:
        fileList = file.strip().split(",")
        
        with open(logDir+'/'+fileName+'-val.txt') as tagFile:
            for tag in tagFile:
                tagList = tag.strip().split(" ")
                fileList[0] = fileList[0].strip()
                fileList[1] = fileList[1].strip()
                tagList[0] = tagList[0].strip()
                tagList[1] = tagList[1].strip()

                if fileList[1] == tagList[0] and tagList[1] == '-1':
                    with open(logDir+'/csv_tag_file_name.csv') as labelfilename:
                        for labelfile in labelfilename:
                            LABELfile = labelfile.strip().split(",")
                            if LABELfile[0] == tagList[0]:
                                labelfile = labelfile.strip()
                                print(fileList[0]+','+labelfile)

# remove  and copy system files and duplicates
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    os.system('rm -rf '+logDir+'/csv_tag_file_name.csv');
    os.system('rm -rf '+logDir+'/'+fileName+'-val.txt');
elif _platform == "win32" or _platform == "win64":
    os.system('DEL '+logDir+'/csv_tag_file_name.csv');
    os.system('DEL '+logDir+'/'+fileName+'-val.txt');
    os.system('rm -rf '+logDir+'/csv_tag_file_name.csv');
    os.system('rm -rf '+logDir+'/'+fileName+'-val.txt');
