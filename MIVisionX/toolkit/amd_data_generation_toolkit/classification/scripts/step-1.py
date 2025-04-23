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
import re
from PIL import Image

def make_square_image(img, width, height, padVal):
    x, y = img.size
    fill_color=(padVal, padVal, padVal, padVal)
    min_size=224
    size = max(min_size, x, y)
    new_img = Image.new('RGBA', (size, size), fill_color)
    new_img.paste(img, ((size - x) / 2, (size - y) / 2))
    new_img = new_img.resize((width, height), Image.BILINEAR)
    new_img = new_img.convert("RGB")
    return new_img

opts, args = getopt.getopt(sys.argv[1:], 'd:o:f:w:h:p:c:')
 
directory = ''
outputDir = ''
fileName = ''
width = -1
height = -1
padVal = -1
userCount = -1

# get user variables
for opt, arg in opts:
    if opt == '-d':
        directory = arg
    elif opt == '-o':
        outputDir = arg
    elif opt == '-f':
        fileName = arg
    elif opt == '-w':
        width = int(arg)
    elif opt == '-h':
        height = int(arg)
    elif opt == '-p':
        padVal = int(arg)
    elif opt == '-c':
        userCount = int(arg)

# error check and script help
if fileName == '' or directory == '' or outputDir == '':
    print('Invalid command line arguments. Usage python step-1.py -d [input image directory] ' \
          '-o [output image directory] -f [new image file name] are required '\
	      '-w [resize width] -h [resize height] -p [padding value] -c [image start count] are optional')
    exit()

# user image start count
count = 0
if userCount != -1:
    count = userCount

# image size
size = width, height

# script output folder
logDir = fileName+'-scriptOutput'
if not os.path.exists(logDir):
    os.makedirs(logDir)

# original std out location 
orig_stdout = sys.stdout

# looping through user images in ascending order
for image in sorted(os.listdir(directory),key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]):
 
    img = Image.open(os.path.join(directory, image))
    # setup filename dictionary for old and new file names
    sys.stdout = open(logDir+'/'+fileName+'-FileNameTranslation.csv','a')

    # resize and pad image
    if width != -1 and height != -1:
        if padVal != -1:
            img = make_square_image(img,width,height,padVal)
        else:
            img = img.resize((width, height), Image.BILINEAR)

    # rename and save images in output folder
    if count < 10:
        img.save( outputDir + fileName + '_000' + str(count) + '.JPEG')
        print(image+', '+fileName+'_000'+str(count)+'.JPEG')
        os.system('exiftool -m -TagsFromFile '+ directory + image + ' ' + outputDir + fileName + '_000' + str(count) + '.JPEG >> output.log');
    elif count > 9 and count < 100:
        img.save( outputDir + fileName + '_00' + str(count) + '.JPEG')
        print(image+', '+fileName+'_00'+str(count)+'.JPEG')
        os.system('exiftool -m -TagsFromFile '+ directory + image + ' ' + outputDir + fileName + '_00' + str(count) + '.JPEG >> output.log');
    elif count > 99 and count < 1000:
        img.save( outputDir + fileName + '_0' + str(count) + '.JPEG')
        print(image+', '+fileName+'_0'+str(count)+'.JPEG')
        os.system('exiftool -m -TagsFromFile '+ directory + image + ' ' + outputDir + fileName + '_0' + str(count) + '.JPEG >> output.log');
    elif count > 999 and count < 10000:
        img.save( outputDir + fileName + '_' + str(count) + '.JPEG')
        print(image+', '+fileName+'_'+str(count)+'.JPEG')
        os.system('exiftool -m -TagsFromFile '+ directory + image + ' ' + outputDir + fileName + '_' + str(count) + '.JPEG >> output.log');

    sys.stdout = orig_stdout
    print('image processed - '+image+' count:'+str(count+1))
    count += 1

# remove system files and duplicates
from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    print('Script1.py Linux Detected')
    os.system('rm -rf '+ outputDir + '*.JPEG_original');
elif _platform == "win32" or _platform == "win64":
    print('Script1.py Windows Detected')
    os.system('DEL '+ outputDir + '*.JPEG_original');
    os.system('rm -rf '+ outputDir + '*.JPEG_original');

# print script variables and report
print('step1.py inputs\n'\
	'\tinput directory: '+directory+'\n\toutput directory: '+outputDir+'\n\timage fileName: '+fileName+'\n\timage width: '+str(width)+'\n'\
	'\timage height: '+str(height)+'\n\timage padding value: '+str(padVal)+'\n\timage count start: '+str(count))
print('Image resize and name change complete and successful.')
