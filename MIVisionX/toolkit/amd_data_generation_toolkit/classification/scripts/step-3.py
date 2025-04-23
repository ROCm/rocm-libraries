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

#import os
import getopt
import sys
#from PIL import Image
#from itertools import izip
#import re

opts, args = getopt.getopt(sys.argv[1:], 'l:t:')
 
labels = ''
imageTaglist = ''
outputString = ''

# get user variables
for opt, arg in opts:
    if opt == '-l':
        labels = arg
    elif opt == '-t':
        imageTaglist = arg

# error check and script help
if labels == '' or imageTaglist == '':
    print('Invalid command line arguments. Usage python step-3.py -l [label.txt with 1000 labels without numbers] ' \
          '-t [image list .txt with file name & tags] are required')
    exit()

# traverse through the image labels to find label IDs
with open(imageTaglist) as Tags: 
	for TAG in Tags:
		TAG = TAG.strip().split(",")
		labelNotFound = 0
		duplicate = 0
		for indv_tag in TAG:
			break_val = 0
			indv_tag = indv_tag.strip()
			indv_tag = indv_tag.lower()
			with open(labels) as labelList:
				count = 0				
				for l in labelList:
					indv_labels = l.strip().split(",")
					for word in indv_labels:						
						if break_val == 1:
							break
						word = word.strip()
						word = word.lower()
						if indv_tag == 'crane':
							TAG[2] = TAG[2].strip()
							if TAG[2] == 'n03126707':
								outputString = TAG[0] + ' 517'
								print(outputString)
								break_val = 1
								labelNotFound = 0
								duplicate = 1
							elif TAG[2] == 'n02012849':
								outputString = TAG[0] + ' 134'
								print(outputString)
								break_val = 1
								labelNotFound = 0
								duplicate = 1

						elif indv_tag == word:
							outputString = TAG[0] + ' ' + str(count)
							print(outputString)
							break_val = 1
							labelNotFound = 0
							duplicate = 1
					count += 1
				
			if break_val == 0:
				labelNotFound = 1
			
		if labelNotFound == 1:
			outputString = TAG[0] + ' -1'
			if duplicate == 0:
				print(outputString)
					

#orig_stdout = sys.stdout
#logDir = fileName+'-scriptOutput'
#sys.stdout = open(logDir+'/step3.py.log','wt')
#print('step3.py inputs\n'\
#    '\tlabel text: '+labels+'\n\timage list text: '+imageTaglist)
#print('Image Labels from tag ist Generation Complete.')
