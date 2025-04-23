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

opts, args = getopt.getopt(sys.argv[1:], 'd:f:')
 
directory = ''
fileName = ''

# get user variables
for opt, arg in opts:
    if opt == '-d':
        directory = arg
    elif opt == '-f':
        fileName = arg

# error check and script help
if fileName == '' or directory == '':
    print('Invalid command line arguments. Usage python step-2.py -d [input image directory] ' \
          '-f [Tag file name] are required')
    exit()

# output exif data
for image in sorted(os.listdir(directory)):
    print('Writing Tags Image ' + image)
    os.system('exiftool -m -filename -subject -s -s -s -t '+ directory + image + ' >> ' + fileName);

# fix output format
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    print('Script2.py Linux Detected')
    os.system('awk \'sub("\t", ",")\' ' + fileName +' >> ' + 'csv_'+fileName)
elif _platform == "win32" or _platform == "win64":
    print('Script2.py Windows Detected')
    os.system('gawk \'sub("\t", ",")\' ' + fileName +' >> ' + 'csv_'+fileName)

# print script variables and report
print('step2.py inputs\n'\
    '\tinput directory: '+directory+'\n\ttag fileName: '+fileName)
print('Image Tag List csv_'+fileName+' Generation Complete and Successful.')
