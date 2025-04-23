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

__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2018 - 2024, AMD Dataset Analysis Tool"
__credits__     = ["Mike Schmit"]
__license__     = "MIT"
__version__     = "0.9.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "mivisionx.support@amd.com"
__status__      = "Alpha"

import os
import getopt
import sys
#import random
import collections
import csv
#import numpy
import datetime

opts, args = getopt.getopt(sys.argv[1:], 'i:o:f:l:g:d:t:')

inputCSVFile = '';
inputImageDirectory = '';
outputDirectory = '';
fileName = '';
labelFile = '';
groundTruthFile = '';
boundingBoxToolFile = ''

for opt, arg in opts:
    if opt == '-i':
        inputCSVFile = arg;
    elif opt == '-d':
        inputImageDirectory = arg;
    elif opt == '-o':
        outputDirectory = arg;        
    elif opt == '-f':
        fileName = arg;
    elif opt == '-l':
        labelFile = arg;
    elif opt == '-g':
        groundTruthFile = arg;
    elif opt == '-t':
        boundingBoxToolFile = arg;

# report error
if inputCSVFile == '' or outputDirectory == '' or labelFile == '' or groundTruthFile == '' or inputImageDirectory == '' or boundingBoxToolFile == '':
    print('Invalid command line arguments.\n'
        '\t\t\t\t-i [input Result CSV File - required](File Format:ImgFileName, L,R,L,R,L,R,L,R)[L:Label R:Result]\n'\
        '\t\t\t\t-d [input Image Directory - required]\n'\
        '\t\t\t\t-o [output Directory - required]\n'\
        '\t\t\t\t-l [input Label File      - required]\n'\
        '\t\t\t\t-f [output file name - required]\n'\
        '\t\t\t\t-g [ground truth File - required]\n'\
        '\t\t\t\t-t [bounding box tool output file - required]\n')

    exit();


if not os.path.exists(inputImageDirectory):
    print("ERROR Invalid Input Image Directory")
    exit();

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory);

# read results.csv
numElements = 0;
with open(inputCSVFile) as resultFile:
    resultCSV = csv.reader(resultFile)
    next(resultCSV) # skip header
    resultDataBase = [r for r in resultCSV]
    numElements = len(resultDataBase)

# read labels.csv
labelElements = 0;
with open(labelFile) as labels:
    labelCSV = csv.reader(labels)
    next(labelCSV) # skip header
    labelLines = [r for r in labelCSV]
    labelElements = len(labelLines)

# read groundtruth.csv
groundtruthElements = 0;
with open(groundTruthFile) as gt:
    groundtruthCSV = csv.reader(gt)
    next(groundtruthCSV) # skip header
    gtLines = [r for r in groundtruthCSV]
    gtElements = len(gtLines)

# read boundingBoxToolFile.csv
toolElements = 0;
with open(boundingBoxToolFile) as bb:
    boundingBoxCSV = csv.reader(bb)
    next(boundingBoxCSV) # skip header
    boxLines = [r for r in boundingBoxCSV]
    boxElements = len(boxLines)

 # create toolkit with icons and images
toolKit_Dir = outputDirectory +'/'+ fileName + '-toolKit'
toolKit_dir = os.path.expanduser(toolKit_Dir)
if not os.path.exists(toolKit_dir):
    os.makedirs(toolKit_dir);

# copy images and icons
from distutils.dir_util import copy_tree
fromDirectory = inputImageDirectory;
toDirectory = toolKit_dir+'/images';
copy_tree(fromDirectory, toDirectory)

from distutils.dir_util import copy_tree
dir_path = os.path.dirname(os.path.realpath(__file__))
fromDirectory = dir_path+'/icons';
toDirectory = toolKit_dir+'/icons';
copy_tree(fromDirectory, toDirectory)
#copy utils
new_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fromDirectory = new_path+'/utils';
toDirectory = toolKit_dir+'/utils';
copy_tree(fromDirectory, toDirectory)

dataFolder = 'images';

#create results directory
resultsDirectory = toolKit_dir+'/results';
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory);

dirs = sorted(os.listdir(inputImageDirectory))


# generate results summary
modelName = 'Object Detection'
totalImages = numElements;
totalMatch = totalMismatch = totalUnknown = totalBoundingBox = totalGroundTruth = totalWithoutGroundTruth = totalNotFound = 0;
accuracyMatch = 0;

for row in resultDataBase:
    for result in row:
        if result == 'Match':
            totalMatch += 1
        elif result == 'IOU Mismatch' or result == 'Label Mismatch':
            totalMismatch += 1
        elif result == 'Unknown':
            totalUnknown += 1
        elif result == 'Not Found':
            totalNotFound += 1

totalGroundTruth = totalMatch + totalMismatch
totalBoundingBox = totalMatch + totalMismatch + totalUnknown
totalWithoutGroundTruth = totalBoundingBox - totalGroundTruth
accuracyMatch = float(totalMatch)/totalBoundingBox*100
accuracyMismatch = float(totalMismatch)/totalBoundingBox*100

#Results Summary 
print("resultsSummary.txt generation ..")
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/resultsSummary.txt','w')
print("\n\n ***************** OBJECT DETECTION SUMMARY ***************** \n");
import numpy as np

print('Total Number of Images -- '+ str(totalImages));
print('Total Number of Bounding Boxes -- '+ str(totalBoundingBox));
print('Number of Bounding Boxes With GroundTruth -- '+str(totalGroundTruth));
print('Number of Bounding Boxes Without GroundTruth -- '+str(totalWithoutGroundTruth));
print("");

print("\n*****GroundTruth*****");
print('Number of Bounding Boxes Matched -- '+str(totalMatch));
print('Number of Bounding Boxes Mismatched -- '+str(totalMismatch));
print('Number of Bounding Boxes not found by tool -- ' +str(totalNotFound))

print("");
print("\n*****Without GroundTruth*****");
print('Number of Bounding Boxes Unknown -- '+str(totalUnknown));

print("\n*****Statistics*****");
print('Match Accuracy -- '+str(np.around(accuracyMatch,decimals=2))+' %');
print('Mismatch Accuracy -- '+str(np.around(accuracyMismatch,decimals=2))+' %');
sys.stdout = orig_stdout
print("resultsSummary.txt generated")

#generate label summary
labelMatch = []
for i in range(20):
    labelMatch.append([])
    for j in range(8):
        labelMatch[i].append(0)

for i in range(labelElements):
    labelMatch[i][0] = labelLines[i][0]
    labelMatch[i][1] = labelLines[i][1]
    #number of images with label for atleast one bounding box
    numImages = 0
    for row in gtLines:
        for label in row:
            if label == labelLines[i][1]:
                numImages += 1
                break
        labelMatch[i][2] = numImages

    #number of matched bounding boxes
    numBoxesMatch = 0
    numBoxesNotFound = 0
    numBoxesMismatch = 0
    numBoxesUnkown = 0
    for x in range(numElements):
        for y in range(len(resultDataBase[x])):
            if resultDataBase[x][y] == labelMatch[i][1] and resultDataBase[x][y+1] == 'Match':
                numBoxesMatch += 1 
            elif resultDataBase[x][y] == labelMatch[i][1] and resultDataBase[x][y+1] == 'Not Found':
                numBoxesNotFound += 1
            elif resultDataBase[x][y] == labelMatch[i][1] and (resultDataBase[x][y+1] == 'IOU Mismatch' or resultDataBase[x][y+1] == 'Label Mismatch'):
                numBoxesMismatch += 1
            elif resultDataBase[x][y] == labelMatch[i][1] and resultDataBase[x][y+1] == 'Unknown':
                numBoxesUnkown += 1
        labelMatch[i][4] = numBoxesMatch
        labelMatch[i][5] = numBoxesMismatch
        labelMatch[i][6] = numBoxesUnkown
        labelMatch[i][7] = numBoxesNotFound
       
    #number of bounding boxes of each label
    numBoxes = numBoxesMatch + numBoxesMismatch + numBoxesNotFound + numBoxesUnkown
    labelMatch[i][3] = numBoxes

# Label Summary
print("labelSummary.csv generation ..")
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/labelSummary.csv','w')
print("Label ID, Label Description, Images in DataBase, Bounding Boxes in DataBase, Bounding Boxes Matched, Bounding Boxes Mismatched, Bounding Boxes Unknown, Bounding Boxes Not Found ");
for i in range(20):
    print("%s,%s,%d,%d,%d,%d,%d,%d"%(labelMatch[i][0], labelMatch[i][1], labelMatch[i][2], labelMatch[i][3], labelMatch[i][4], labelMatch[i][5], labelMatch[i][6], labelMatch[i][7]));
sys.stdout = orig_stdout
print("labelSummary.csv generated")

# generate html file
print("index.html generation ..")
#orig_stdout = sys.stdout
sys.stdout = open(toolKit_dir+'/index.html','w')

print ("<!DOCTYPE HTML PUBLIC \" -//W3C//DTD HTML 4.0 Transitional//EN\">");
print ("\n<html>");
print ("<head>");
print ("\n\t<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\"/>");
print ("\t<title>AMD Dataset Analysis Tool</title>");
print ("\t<link rel=\"icon\" href=\"icons/vega_icon_150.png\"/>");

# page style
print ("\n\t<style type=\"text/css\">");
print ("\t");
print ("\tbody,div,table,thead,tbody,tfoot,tr,th,td,p { font-family:\"Liberation Sans\"; font-size:x-small }");
print ("\ta.comment-indicator:hover + comment { background:#ffd; position:absolute; display:block; border:1px solid black; padding:0.5em;  }");
print ("\ta.comment-indicator { background:red; display:inline-block; border:1px solid black; width:0.5em; height:0.5em;  }");
print ("\tcomment { display:none;  } tr:nth-of-type(odd) { background-color:#f2f2f2;}");
print ("\t");
print ("\t#myImg { border-radius: 5px; cursor: pointer; transition: 0.3s; }");
print ("\t#myImg:hover { opacity: 0.7; }");
print ("\t.modal{ display: none; position: fixed; z-index: 8; padding-top: 100px; left: 0; top: 0;width: 100%;");
print ("\t       height: 100%; overflow: auto; background-color: rgb(0,0,0); background-color: rgba(0,0,0,0.9); }");
print ("\t.modal-content { margin: auto; display: block; width: 80%; max-width: 500px; }");
print ("\t#caption { margin: auto; display: block; width: 80%; max-width: 700px; text-align: center; color: white;font-size: 12px; padding: 10px 0; height: 150px;}");
print ("\tcanvas { padding-left: 0; padding-right: 0;    margin-left: auto;    margin-right: auto;    display: block;  width: 416px; height: 416px}");
print ("\t.modal-content, #caption {  -webkit-animation-name: zoom;  -webkit-animation-duration: 0.6s;");
print ("\t                           animation-name: zoom; animation-duration: 0.6s; }");
print ("\t@-webkit-keyframes zoom {  from { -webkit-transform:scale(0) }  to { -webkit-transform:scale(1) }}");
print ("\t@keyframes zoom {    from {transform:scale(0)}     to {transform:scale(1) }}");
print ("\t.close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; transition: 0.3s; }");
print ("\t.close:hover,.close:focus { color: #bbb; text-decoration: none; cursor: pointer; }");
print ("\t@media only screen and (max-width: 400px){ .modal-content {     width: 100%; } }");
print ("\t");
print ("\tbody { font-family: \"Lato\", sans-serif;}");
print ("\t.sidenav { height: 100%; width: 0; position: fixed; z-index: 7; top: 0; left: 0; background-color: #111;");
print ("\t\t overflow-x: hidden;    transition: 0.5s; padding-top: 60px;}");
print ("\t.sidenav a { padding: 8px 8px 8px 32px; text-decoration: none; font-size: 25px; color: #818181; display: block; transition: 0.3s;}");
print ("\t.sidenav a:hover { color: #f1f1f1;}");
print ("\t.sidenav .closebtn {  position: absolute; top: 0; right: 25px; font-size: 36px; margin-left: 50px;}");
print ("\t#main {  transition: margin-left .5s;  padding: 16px; }");
print ("\t@media screen and (max-height: 450px) { .sidenav {padding-top: 15px;} .sidenav a {font-size: 18px;} }");
print ("\t");
print ("\tbody {margin:0;}");
print ("\t.navbar {  overflow: hidden;  background-color: #333;  position: fixed; z-index: 6;  top: 0;  width: 100%;}");
print ("\t.navbar a {  float: left;  display: block;  color: #f2f2f2;  text-align: center;  padding: 14px 16px;  text-decoration: none;  font-size: 17px; }");
print ("\t.navbar a:hover {  background: #ddd;  color: black;}");
print ("\t.main {  padding: 16px;  margin-top: 30px; }");
print ("\t");
print ("\tselect {-webkit-appearance: none; -moz-appearance: none; text-indent: 0px; text-overflow: ''; color:maroon; }");
print ("\t");
print ("\t.tooltip { position: relative; display: inline-block;}");
print ("\t.tooltip .tooltiptext { visibility: hidden; width: 150px; background-color: black; color: gold;");
print ("\t\ttext-align: center;  border-radius: 6px;  padding: 5px; position: absolute; z-index: 3;}");
print ("\t.tooltip:hover .tooltiptext { visibility: visible;}");
print ("\t");
print ("\t.footer { position: fixed; left: 0;    bottom: 0;  width: 100%;    background-color: #333;  color: white;  text-align: center;}");
print ("\t");
print ("\t</style>");
print ("\n</head>");
print ("\n\n<body>");
print ("\t");
print ("\t<div id=\"myModal\" class=\"modal\" align=\"center\"> <span class=\"close\">&times;</span>  <img class=\"modal-content\" id=\"img01\"> \
    <canvas id=\"myCanvas\" width=\"416\" height=\"416\" > Your browser does not support the HTML5 canvas tag.</canvas> \
    <div id=\"caption\"></div> </div>");
print ("\t");

# table content order
print ("\t<div id=\"mySidenav\" class=\"sidenav\">");
print ("\t<a href=\"javascript:void(0)\" class=\"closebtn\" onclick=\"closeNav()\">&times;</a>");
print ("\t<A HREF=\"#table0\"><font size=\"5\">Summary</font></A><br>");
print ("\t<A HREF=\"#table1\"><font size=\"5\">Graphs</font></A><br>");
#print ("\t<A HREF=\"#table2\"><font size=\"5\">Hierarchy</font></A><br>");
print ("\t<A HREF=\"#table3\"><font size=\"5\">Labels</font></A><br>");
print ("\t<A HREF=\"#table4\"><font size=\"5\">Image Results</font></A><br>");
print ("\t<A HREF=\"#table5\"><font size=\"5\">Compare</font></A><br>");
print ("\t<A HREF=\"#table6\"><font size=\"5\">Model Score</font></A><br>");
print ("\t<A HREF=\"#table7\"><font size=\"5\">Help</font></A><br>");
print ("\t</div>");
print ("\t");

# scripts
print ("\t<script>");
print ("\t\tfunction openNav() {");
print ("\t\t\tdocument.getElementById(\"mySidenav\").style.width = \"250px\";");
print ("\t\t\tdocument.getElementById(\"main\").style.marginLeft = \"250px\";}");
print ("\t\tfunction closeNav() {");
print ("\t\t\tdocument.getElementById(\"mySidenav\").style.width = \"0\";");
print ("\t\t\tdocument.getElementById(\"main\").style.marginLeft= \"0\";}");
print ("\t\tfunction myreload() { location.reload();}");
print ("\t");
print ("\t\tfunction sortTable(coloum,descending) {");
print ("\t\tvar table, rows, switching, i, x, y, shouldSwitch;");
print ("\t\ttable = document.getElementById(id=\"resultsTable\"); switching = true;");
print ("\t\twhile (switching) {  switching = false; rows = table.getElementsByTagName(\"TR\");");
print ("\t\t\tfor (i = 1; i < (rows.length - 1); i+= 1) { shouldSwitch = false;");
print ("\t\t\t\tx = rows[i].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\ty = rows[i + 1].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\tif(descending){if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}");
print ("\t\t\t\telse{if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}}");
print ("\t\t\t\tif (shouldSwitch) {  rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);");
print ("\t\t\t\t\tswitching = true;}}}");
print ("\t");
print ("\t");
print ("\t\tfunction sortLabelsTable(coloum,descending) {");
print ("\t\tvar table, rows, switching, i, x, y, shouldSwitch;");
print ("\t\ttable = document.getElementById(id=\"labelsTable\"); switching = true;");
print ("\t\twhile (switching) {  switching = false; rows = table.getElementsByTagName(\"TR\");");
print ("\t\t\tfor (i = 1; i < (rows.length - 1); i+= 1) { shouldSwitch = false;");
print ("\t\t\t\tx = rows[i].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\ty = rows[i + 1].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\tif(descending){if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}");
print ("\t\t\t\telse{if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}}");
print ("\t\t\t\tif (shouldSwitch) {  rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);");
print ("\t\t\t\t\tswitching = true;}}}");
print ("\t");
print ("\t</script>");
print ("\t<script src=\"utils/sorttable.js\"></script>");
print ("\t");
print ("\t<script>");
print ("\t\tfunction filterResultTable(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) > -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t</script>");
print ("\t");
print ("\t");
print ("\t<script>");
print ("\t\tfunction filterLabelTable(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"labelsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) > -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t</script>");
print ("\t");
print ("");
print ("\t<script>");
print ("\t\tfunction clearLabelFilter() {");
print ("\t\tdocument.getElementById('Label ID').value = ''");
print ("\t\tdocument.getElementById('Label Description').value = ''");
print ("\t\tdocument.getElementById('Images in DataBase').value = ''");
print ("\t\tdocument.getElementById('Bounding Boxes in DataBase').value = ''");
print ("\t\tdocument.getElementById('Matched').value = ''");
print ("\t\tdocument.getElementById('Not Found').value = ''");
print ("\t\tdocument.getElementById('Mismatched').value = ''");
print ("\t\tfilterLabelTable(0,'Label ID') }");
print ("\t</script>");
print ("");
print ("");
print ("\t<script>");
print ("\t\tfunction clearResultFilter() {");
print ("\t\tdocument.getElementById('GroundTruthText').value = ''");
print ("\t\tdocument.getElementById('BoundingBoxToolText').value = ''");
print ("\t\tdocument.getElementById('Confidence').value = ''");
print ("\t\tfilterResultTable(2,'GroundTruthText') }");
print ("\t</script>");
print ("");
print ("\t<script>");
print ("\t\tfunction findGroundTruthLabel(label) {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('GroundTruthText').value = label;");
print ("\t\tfilterResultTable(2,'GroundTruthText');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findMisclassifiedGroundTruthLabel(label) {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Text1').value = label;");
print ("\t\tfilterResultTable(10,'Text1');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction highlightRow(obj){");
print ("\t\tvar tr=obj; while (tr.tagName.toUpperCase()!='TR'&&tr.parentNode){  tr=tr.parentNode;}");
print ("\t\tif (!tr.col){tr.col=tr.bgColor; } if (obj.checked){  tr.bgColor='#d5f5e3';}");
print ("\t\telse {  tr.bgColor=tr.col;}}");
print ("");
print ("\t\tfunction goToImageResults() { window.location.href = '#table4';}");
print ("");
print ("\t\tfunction findImagesWithNoGroundTruthLabel() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('GroundTruthText').value = '';");
print ("\t\tfilterResultTable(2,'GroundTruthText');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findImageMisMatch() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '0';");
print ("\t\tfilterResultTable(9,'Matched');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findTopKMatch() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '0';");
print ("\t\tnotResultFilter();");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction filterResultTableInverse(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) <= -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t\tfunction findImagesWithGroundTruthLabel(){");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '-1';");
print ("\t\tfilterResultTableInverse(9, 'Matched')");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction notResultFilter( ) {");
print ("\t\tvar input, filter, table, tr, td, i, rowNum, count;");
print ("\t\tcount=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinput = document.getElementById('GroundTruthText');  rowNum = 2;count+= 1;}");
print ("\t\tif(document.getElementById('BoundingBoxToolText').value != ''){");
print ("\t\tinput = document.getElementById('BoundingBoxToolText'); rowNum = 3;count+= 1;}");
print ("\t\tif(document.getElementById('Confidence').value != ''){");
print ("\t\tinput = document.getElementById('Top1'); rowNum = 4;count+= 1; }");
print ("\t\tif(count == 0){alert(\"Not Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count > 1){");
print ("\t\talert(\"Not Filter ERROR: Only one variable filtering supported. Use Clear Filter and enter one filter variable\");}");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) <= -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t\tfunction andResultFilter( ) {");
print ("\t\tvar inputOne, inputTwo, filterOne, filterTwo, table, tr, tdOne, tdTwo, i, rowNumOne, rowNumTwo,count;");
print ("\t\tcount=0;");
print ("\t\trowNumOne=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthText');   rowNumOne = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('BoundingBoxToolText').value != ''){");
print ("\t\tinputOne = document.getElementById('BoundingBoxToolText'); rowNumOne = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Confidence').value != ''){");
print ("\t\tinputOne = document.getElementById('Confidence'); rowNumOne = 4;count+= 1; }");
print ("\t\tif(document.getElementById('GroundTruthText').value != '' && rowNumOne  != 2){");
print ("\t\tinputTwo = document.getElementById('GroundTruthText');   rowNumTwo = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('BoundingBoxToolText').value != '' && rowNumOne  != 3){");
print ("\t\tinputTwo = document.getElementById('BoundingBoxToolText'); rowNumTwo = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Confidence').value != '' && rowNumOne  != 4){");
print ("\t\tinputTwo = document.getElementById('Confidence'); rowNumTwo = 4;count+= 1; }");
print ("\t\tif(count == 0){alert(\"AND Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count == 1){alert(\"AND Filter ERROR: Enter two variables\");}");
print ("\t\telse if(count > 2){");
print ("\t\talert(\"AND Filter ERROR: Only two variable filtering supported. Use Clear Filter and enter two filter variable\");}");
print ("\t\tfilterOne = inputOne.value.toUpperCase();");
print ("\t\tfilterTwo = inputTwo.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttdOne = tr[i].getElementsByTagName(\"td\")[rowNumOne];");
print ("\t\ttdTwo = tr[i].getElementsByTagName(\"td\")[rowNumTwo];");
print ("\t\tif (tdOne && tdTwo) { ");
print ("\t\tif (tdOne.innerHTML.toUpperCase().indexOf(filterOne) > -1 && tdTwo.innerHTML.toUpperCase().indexOf(filterTwo) > -1) ");
print ("\t\t{tr[i].style.display = \"\";}");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t\tfunction orResultFilter( ) {");
print ("\t\tvar inputOne, inputTwo, filterOne, filterTwo, table, tr, tdOne, tdTwo, i, rowNumOne, rowNumTwo, count;");
print ("\t\tcount=0;");
print ("\t\trowNumOne=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthText');   rowNumOne = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('BoundingBoxToolText').value != ''){");
print ("\t\tinputOne = document.getElementById('BoundingBoxToolText'); rowNumOne = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Confidence').value != ''){");
print ("\t\tinputOne = document.getElementById('Confidence'); rowNumOne = 4;count+= 1; }");
print ("\t\tif(document.getElementById('GroundTruthText').value != '' && rowNumOne  != 2){");
print ("\t\tinputTwo = document.getElementById('GroundTruthText');   rowNumTwo = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('BoundingBoxToolText').value != '' && rowNumOne  != 3){");
print ("\t\tinputTwo = document.getElementById('BoundingBoxToolText'); rowNumTwo = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Confidence').value != '' && rowNumOne  != 4){");
print ("\t\tinputTwo = document.getElementById('Confidence'); rowNumTwo = 4;count+= 1; }");
print ("\t\tif(count == 0){alert(\"OR Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count == 1){alert(\"OR Filter ERROR: Enter two variables\");}");
print ("\t\telse if(count > 2){");
print ("\t\talert(\"OR Filter ERROR: Only two variable filtering supported. Use Clear Filter and enter two filter variable\");}");
print ("\t\tfilterOne = inputOne.value.toUpperCase();");
print ("\t\tfilterTwo = inputTwo.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttdOne = tr[i].getElementsByTagName(\"td\")[rowNumOne];");
print ("\t\ttdTwo = tr[i].getElementsByTagName(\"td\")[rowNumTwo];");
print ("\t\tif (tdOne && tdTwo) { ");
print ("\t\tif (tdOne.innerHTML.toUpperCase().indexOf(filterOne) > -1 || tdTwo.innerHTML.toUpperCase().indexOf(filterTwo) > -1) ");
print ("\t\t{tr[i].style.display = \"\";}");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t</script>");
print ("");

print("\t<script>");
print("\t\tfunction clearResultBoxChoices() {");
print("\t\tdocument.getElementById('NumberOfBoxes').value = ''");
print("\t\tdocument.getElementById('TypeOfBox1').checked = false");
print("\t\tdocument.getElementById('TypeOfBox2').checked = false");
print("\t\tdocument.getElementById('TypeOfBox3').checked = true");
print("\t\t}");
print("\t</script>");
print ("");


#draw bounding box for tool
print ("\t<script>");
print ("\t\tfunction drawBoundingBox(x,y,w,h,label,conf) {");
print("\t\tstoreNumber();");
print("\t\tfunction storeNumber(){");
print("\t\tvar numB = document.getElementById(\"NumberOfBoxes\").value;");

print("\t\tif (numB == \"\") {numB = x.length;}");
print ("\t\tvar canvas, context;");
print ("\t\tfor(i = 0; i < numB; i++) {");
print ("\t\tcanvas = document.getElementById('myCanvas');");
print ("\t\tcontext = canvas.getContext('2d');");
print ("\t\tcontext.beginPath();");
print ("\t\tcontext.rect(Math.round(x[i]),Math.round(y[i]),Math.round(w[i]),Math.round(h[i]));");
print ("\t\tcontext.lineWidth = 3;");
print ("\t\tcontext.strokeStyle = 'green';");
print ("\t\tcontext.stroke();");
print ("\t\tvar font = '16px Times New Roman';");
print ("\t\tvar txt = label[i] + \";\" + (parseFloat(conf[i])*100).toFixed() + \"%\";");  
print ("\t\tcontext.font = font;");
print ("\t\tcontext.textBaseline = 'top';");
print ("\t\tcontext.fillStyle = 'green';");      
print ("\t\tvar width1 = context.measureText(txt).width;");
print ("\t\tcontext.fillRect(x[i], y[i], width1, parseInt(font, 10));");
print ("\t\tcontext.fillStyle = '#000'; ");
print ("\t\tcontext.fillText(txt,Math.round(x[i]),Math.round(y[i]));}}}"); 
print ("");
print ("\t</script>");
print ("");

#draw bounding box for GT
print ("\t<script>");
print ("\t\tfunction drawBoundingBoxGT(x,y,w,h,label) {");
print("\t\tstoreNumber();");
print("\t\tfunction storeNumber(){");
print("\t\tvar numB = document.getElementById(\"NumberOfBoxes\").value;");
print("\t\tif (numB == \"\") {numB = x.length;}");
print ("\t\tvar canvas, context;");
print ("\t\tfor(i = 0; i < numB; i++) {");
print ("\t\tcanvas = document.getElementById('myCanvas');");
print ("\t\tcontext = canvas.getContext('2d');");
print ("\t\tcontext.beginPath();");
print ("\t\tcontext.rect(Math.round(x[i]),Math.round(y[i]),Math.round(w[i]),Math.round(h[i]));");
print ("\t\tcontext.lineWidth = 3;");
print ("\t\tcontext.strokeStyle = 'blue';");
print ("\t\tcontext.stroke();");
print ("\t\tvar font = '16px Times New Roman';");
print ("\t\tvar txt = label[i];");  
print ("\t\tcontext.font = font;");
print ("\t\tcontext.textBaseline = 'top';");
print ("\t\tcontext.fillStyle = 'blue';");     
print ("\t\tvar width = context.measureText(txt).width;");
print ("\t\tcontext.fillRect(x[i], y[i], width, parseInt(font, 10));");
print ("\t\tcontext.fillStyle = '#000'; ");
print ("\t\tcontext.fillText(txt,Math.round(x[i]),Math.round(y[i]));}}}"); 
print ("");
print ("\t</script>");
print ("");

#Top view header
print ("\t<div class=\"navbar\">");
print ("\t<a href=\"#\">");
print ("\t<div id=\"main\">");
print ("\t<span style=\"font-size:30px;cursor:pointer\" onclick=\"openNav()\">&#9776; Views</span>");
print ("\t</div></a>");
print ("\t<a href=\"https://www.amd.com/en\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_amd_logo.png\" alt=\"AMD\" /></a>");
print ("\t<a href=\"https://gpuopen.com/\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_radeon_logo.png\" alt=\"GPUopen\" /></a>");
print ("\t<a href=\"https://github.com/ROCm/amdovx-modules#amd-openvx-modules-amdovx-modules\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_github_logo.png\" alt=\"AMD GitHub\" /></a>");
print ("\t<img \" src=\"icons/ADAT_500x100.png\" alt=\"AMD Inference ToolKit\" hspace=\"100\" height=\"90\"/> ");
print ("\t</div>");
print ("\t");

# graph script
print("\t<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>");
print("\t<script type=\"text/javascript\">");
print("\t");

#overall summary
print("\tgoogle.charts.load('current', {'packages':['bar']});");
print("\tgoogle.charts.setOnLoadCallback(drawChart);");
print("\tfunction drawChart(){");
print("\tvar data = google.visualization.arrayToDataTable([");
print("\t['  '     ,  'Match'  , 'Mismatch', 'No Ground Truth' ],");
print("\t['Summary',   %d     , %d        , %d         ]"%(totalMatch,totalMismatch,totalWithoutGroundTruth));
print("\t]);");
print("\tvar options = { title: 'Overall Result Summary', vAxis: { title: 'Images' }, width: 800, height: 400 };");
print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats'));");
print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
print("\t");

#TopK pass fail summary
topKValue = 1;
print("\tgoogle.charts.load('current', {'packages':['corechart']});");
print("\tgoogle.charts.setOnLoadCallback(drawTopKResultChart);");
print("\tfunction drawTopKResultChart() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('string', 'Top K');");
print("\tdata.addColumn('number', 'Matchs');");
print("\tdata.addRows([");
print("\t[ 'Matched Top%d Choice', %d  ],"%(topKValue,accuracyMatch));
print("\t[ 'MisMatched', %d  ]]);"%(accuracyMismatch));
print("\tvar options = { title:'Image Match/Mismatch Summary', width:750, height:400 };");
print("\tvar chart = new google.visualization.PieChart(document.getElementById('topK_result_chart_div'));");
print("\tchart.draw(data, options);}");
print("\t");


#only matched confidence
confidence_match_boxes = []

for i in range(numElements):
    n = 3
    m = 2
    while n < len(resultDataBase[i]):
        if(resultDataBase[i][m] == 'Match'):
            confidence_match_boxes.append(float(resultDataBase[i][n]))
        n = n + 3
        m = m + 3 

confidence_match_boxes = [round(elem,2) for elem in confidence_match_boxes]

confidence_mismatch_boxes = []

for i in range(numElements):
    n = 3
    m = 2
    while n < len(resultDataBase[i]):
        if resultDataBase[i][m] == 'IOU Mismatch' or resultDataBase[i][m] == 'Label Mismatch':
            confidence_mismatch_boxes.append(float(resultDataBase[i][n]))
        n = n + 3
        m = m + 3 

confidence_mismatch_boxes = [round(elem,2) for elem in confidence_mismatch_boxes]
"""
sys.stdout = orig_stdout
print confidence_match_boxes
print("len = ", len(confidence_match_boxes))
"""
freq_match = collections.Counter(confidence_match_boxes)
fVal = 0.99
i = 99
while i >= 0:
    if round(fVal,2) not in freq_match:
        freq_match[round(fVal,2)] = 0;
    fVal = round(fVal,2) - round(0.01,2)
    i = i-1
freq_match = collections.OrderedDict(sorted(freq_match.items()))

keys_match = list(freq_match.keys())
values_match = list(freq_match.values())

freq_mismatch = collections.Counter(confidence_mismatch_boxes)
fVal = 0.99
i = 99
while i >= 0:
    if round(fVal,2) not in freq_mismatch:
        freq_mismatch[round(fVal,2)] = 0;
    fVal = round(fVal,2) - round(0.01,2)
    i = i-1
freq_mismatch = collections.OrderedDict(sorted(freq_mismatch.items()))

keys_mismatch = list(freq_mismatch.keys())
values_mismatch = list(freq_mismatch.values())

#Cummulative Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawConfBoxGraph);");
print("\tfunction drawConfBoxGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'Boxes');");
print("\tdata.addColumn('number', 'Match');");
print("\tdata.addColumn('number', 'Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
i = 99;
sumPass = 0;
sumFail = 0;
while i >= 0:    
    sumPass = float(sumPass + values_match[i]);
    sumFail = float(sumFail + values_mismatch[i]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal, (sumPass/totalGroundTruth), (sumFail/totalGroundTruth)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal, (sumPass/totalGroundTruth), (sumFail/totalGroundTruth)));
    fVal = fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Number of Boxes'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

print("\t</script>");

#Overall Summary

print("\t<!-- Overall Summary -->");
print("\t<A NAME=\"table0\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>%s Overall Summary</em></font></h1></A>" %(modelName));
print("\t<table align=\"center\">");
print("\t<col width=\"350\">");
print("\t<col width=\"50\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Total Number of <b>Images</b></font></td>");
print("\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"black\" size=\"4\" onclick=\"clearResultFilter();goToImageResults();\"><b>%d</b></font><span class=\"tooltiptext\">Click on the Text to go to Image Results</span></div></b></td>"%(len(dirs)));
print("\t</tr>");
print("\t</table>\n<br><br><br>");
print("\t<table align=\"center\">");
print("\t<col width=\"350\">");
print("\t<col width=\"50\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Bounding Boxes <b>With Ground Truth</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalGroundTruth));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Bounding Boxes <b>Without Ground Truth</b></font></td>");
print("\t<td align=\"center\"><b><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalWithoutGroundTruth));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Total Bounding Boxes</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalBoundingBox));
print("\t</tr>");
print("\t</table>\n<br><br><br>");
print("\t<table align=\"center\">\n \t<col width=\"300\">\n \t<col width=\"100\">\n \t<col width=\"350\">\n \t<col width=\"100\">\n<tr>");
print("\t<td><font color=\"black\" size=\"4\">Total <b>Match</b></font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalMatch));
print("\t<td><font color=\"black\" size=\"4\">Total <b>Mismatch</b></font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalMismatch));
print("\t</tr>\n<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Match Percentage</b></font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%(accuracyMatch));
print("\t<td><font color=\"black\" size=\"4\"><b>Mismatch Percentage</b></font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%(accuracyMismatch));
print("\t</tr>\n<tr>");
print("\t</tr>\n</table>\n<br><br><br>");

#summary date and time
print("\t<h1 align=\"center\"><font color=\"DodgerBlue\" size=\"4\"><br><em>Summary Generated On: </font><font color=\"black\" size=\"4\"> %s</font></em></h1>"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')));

#Graph
print("\t<!-- Graph Summary -->");
print("<A NAME=\"table1\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Graphs</em></font></h1></A>");
print("\t<center><div id=\"Model_Stats\" style=\"border: 1px solid #ccc\"></div></center>");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t");
print("\t<tr>");
print("\t<center><div id=\"topK_result_chart_div\" style=\"border: 1px solid #ccc\"></div></center>");
print("\t <td><center><div id=\"pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t</tr>");

#label
print("\t<!-- Label Summary -->");
print("<A NAME=\"table3\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Label Summary (stats per image class)</em></font></h1></A>");
print("\t\t<table id=\"filterLabelTable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 70%\">");
print("\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Label ID\" onkeyup=\"filterLabelTable(0,id)\" placeholder=\"Label ID\" title=\"Label ID\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Label Description\" onkeyup=\"filterLabelTable(1,id)\" placeholder=\"Label Description\" title=\"Label Description\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Images in DataBase\" onkeyup=\"filterLabelTable(2,id)\" placeholder=\"Images in DataBase\" title=\"Images in DataBase\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Bounding Boxes in DataBase\" onkeyup=\"filterLabelTable(3,id)\" placeholder=\"Bounding Boxes in DataBase\" title=\"Bounding Boxes in DataBase\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched\" onkeyup=\"filterLabelTable(4,id)\" placeholder=\"Matched\" title=\"Matched\"></td>");
print("\t\t</tr>\n\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Not Found\" onkeyup=\"filterLabelTable(5,id)\" placeholder=\"Not Found\" title=\"Not Found\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Mismatched\" onkeyup=\"filterLabelTable(6,id)\" placeholder=\"Mismatched\" title=\"Mismatched\"></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:yellow;\" onclick=\"clearLabelFilter()\">Clear Filter</button></td>");
print("\t\t</tr>");
print("\t\t</table>");
print("\t\t<br>");
print("\t\t");
print("\t<table class=\"sortable\" id=\"labelsTable\" align=\"center\">");
print("\t<col width=\"100\">");
print("\t<col width=\"250\">");
print("\t<col width=\"200\">");
print("\t<col width=\"200\">");
print("\t<col width=\"200\">");
print("\t<col width=\"200\">");
print("\t<tr>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Label ID</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Label Description</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Images in DataBase</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Bounding Boxes in DataBase</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Bounding Boxes Matched</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Bounding Boxes Not Found</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Bounding Boxes Mismatched</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>Check</b></font></td>");
print("\t\t</tr>");

totalLabelsFound = totalLabelsNeverFound = totalLabelsGroundTruth = totalLabelsWithoutGroundTruth = 0
i = 0;       
while i < 20:
    totalLabelsFound += labelMatch[i][4] + labelMatch[i][5] + labelMatch[i][6]
    totalLabelsNeverFound += labelMatch[i][7]
    totalLabelsGroundTruth += labelMatch[i][4] + labelMatch[i][5] + labelMatch[i][7]
    totalLabelsWithoutGroundTruth += labelMatch[i][6]
    labelTxt = labelLines[i][1];
    print("\t<tr>");
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\" onclick=\"findGroundTruthLabel('%s')\"><b>%d</b></font></td>"%(labelTxt,i));
    print("\t\t<td align=\"left\" onclick=\"findGroundTruthLabel('%s')\"><b>%s</b></td>"%(labelTxt,labelTxt));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][2])); 
    if (labelMatch[i][3] == 0):
        print("\t\t<td align=\"center\"><font color=\"red\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][3]));
    else:
        print("\t\t<td align=\"center\"><font color=\"green\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][3]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][4]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][5]));
    if (labelMatch[i][6] == 0):
        print("\t\t<td align=\"center\"><font color=\"green\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][6]));
    else:
        print("\t\t<td align=\"center\"><font color=\"red\" size=\"2\"><b>%d</b></font></td>"%(labelMatch[i][6]));
    print("\t\t<td align=\"center\"><input id=\"id_%d\" name=\"id[%d]\" type=\"checkbox\" value=\"%d\" onClick=\"highlightRow(this);\"></td>"%(i,i,i));
    print("\t\t</tr>");
    i = i + 1;
print("</table>");

print("<h1 align=\"center\"><font color=\"DodgerBlue\" size=\"4\"><br><em>Label Summary</em></font></h1>");
print("\t<table align=\"center\">");
print("\t<col width=\"350\">");
print("\t<col width=\"150\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels in Ground Truth <b>found</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%((totalLabelsFound-totalLabelsWithoutGroundTruth)));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels in Ground Truth <b>not found</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsNeverFound));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Total</b> Labels in Ground Truth</font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsGroundTruth));
print("\t</tr>");
print("</table>");
print("\t<br><br><table align=\"center\">");
print("\t<col width=\"400\">");
print("\t<col width=\"150\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels <b>not in Ground Truth</b> found</font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsWithoutGroundTruth));
print("\t</tr>");
print("</table>");

#generate image resullts


#Image result
print("\t<!-- Image Summary -->");
print("<A NAME=\"table4\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Image Results</em></font></h1></A>");
print("\t\t<table id=\"filterTable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 60%\">");
print("\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"GroundTruthText\" onkeyup=\"filterResultTable(2,id)\" placeholder=\"Ground Truth Text\" title=\"Ground Truth Text\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"BoundingBoxToolText\" onkeyup=\"filterResultTable(3,id)\" placeholder=\"Bounding Box Tool Text\" title=\"Bounding Box Tool Text\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Confidence\" onkeyup=\"filterResultTable(4,id)\" placeholder=\"Match Confidence\" title=\"Confidence\"></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:yellow;\" onclick=\"clearResultFilter()\">Clear Filter</button></td>");
print("\t\t</tr>");
print("\t\t<tr>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"notResultFilter()\">Not Filter</button></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"andResultFilter()\">AND Filter</button></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"orResultFilter()\">OR Filter</button></td>");
print("\t\t</tr>");
print("\t\t</table>");
print("\t\t<br>");
print("\t\t");
print("\t\t<table id=\"DrawBoxTable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 60%\">");
print("\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"NumberOfBoxes\" oninput= \"storeNumber()\"placeholder=\"Number of Boxes\" title=\"Number of Boxes\"></td>");
print("\t\t<td>");
print("\t\t<form name=\"boxTypeForm\" onclick=\"return onclickEventBoxType()\">");
print("\t\tSelect type of boxes to show:");
print("\t\t<br>");
print("\t\t<input type=\"radio\" id=\"TypeOfBox1\" name = \"TypeOfBox\" value = \"GT\">Only GT<br>");
print("\t\t<input type=\"radio\" id=\"TypeOfBox2\" name = \"TypeOfBox\" value = \"BB\">Only Bounding Boxes<br>");
print("\t\t<input type=\"radio\" id=\"TypeOfBox3\" name = \"TypeOfBox\" value = \"Both\" checked>Both<br>");
print("\t\t</form>");
print("\t\t</td>");
print("\t\t<tr>");
print("\t\t<td align=\"center\"><button style=\"background-color:yellow;\" onclick=\"clearResultBoxChoices()\">Clear Box Filter</button></td>");
print("\t\t</tr>");
print("\t\t</table>");
print("\t\t<br>");
print("<table id=\"resultsTable\" class=\"sortable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 98%\">");
print("\t<tr>");
print("\t\t<td height=\"17\" align=\"center\"><font color=\"Maroon\" size=\"2\" ><b>Image</b></font></td>");
print("\t\t<td height=\"17\" align=\"center\"><font color=\"Maroon\" size=\"2\"><b>FileName</b></font></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Ground Truth Text</font><span class=\"tooltiptext\">Input Image Label. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Bounding Box Tool Text</font><span class=\"tooltiptext\">Output Image Label. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Confidence</font><span class=\"tooltiptext\">Confidence of the Match. Click on the Text to Sort</span></div></b></td>");
print("\t\t</tr>");

gtText = []
boxText = []
conf = []

for i in range(gtElements):
    gtText.append([])
    boxText.append([])
    conf.append([])
    for j in range(16):
        gtText[i].append("")
        boxText[i].append("")
        conf[i].append("")

#should hold 15 bounding boxes per image as of now!!!!!!!!!!!!!

for i in range(gtElements):
    n = 1
    for j in range(1,16):
        if n < len(gtLines[i]):
            gtText[i][0] = gtLines[i][0]
            gtText[i][j] = gtLines[i][n]
        n = n+6

#should hold 15 bounding boxes per image as of now!!!!!!!!!!!!!

for i in range(boxElements):
    n = 1
    for j in range(1,16):
        if n < len(boxLines[i]):
            boxText[i][0] = boxLines[i][0]
            boxText[i][j] = boxLines[i][n]
        n = n+6


for i in range(boxElements):
    n = 6
    for j in range(1,16):
        if n < len(boxLines[i]):
            conf[i][0] = boxLines[i][0]
            conf[i][j] = boxLines[i][n]
        n = n+6
checked_box = [False] * len(dirs)  
checked_gt = [False] * len(dirs) 
for i in range(len(dirs)):
    print("\t\t<tr>");
    print("\t\t<td height=\"17\" align=\"center\"><img id=\"myImg%d\" src=\"%s/%s\" alt=\"green: Bounding Box Tool Output <br> blue: Ground Truth Output <br>Showing as many boxes as chosen(defualt: all boxes)\" width=\"30\" height=\"30\"></td>"
        %(i,dataFolder,dirs[i]));
    print("\t\t<td height=\"17\" align=\"center\"><a href=\"%s/%s\" target=\"_blank\">%s</a></td>"
        %(dataFolder,dirs[i], dirs[i]));
    print("\t\t<td align = \"left\">");
    for j in range(len(gtText)):
        if gtText[j][0] == dirs[i]:
            gtlabelTxt = ",".join(string for string in gtText[j][1:] if len(string) > 0)
            print("\t\t%s"%(gtlabelTxt.replace(',', '<br>')));
    print("\t\t</td>");
    print("\t\t<td align=\"left\" >");
    for k in range(len(boxText)):
        if boxText[k][0] == dirs[i]:
            boxlabelTxt = ",".join(string for string in boxText[k][1:] if len(string) > 0)
            print("\t\t%s"%(boxlabelTxt.replace(',', '<br>')));
    print("\t\t</td>");    
    print("\t\t<td align=\"left\" >");
    for k in range(len(conf)):
        if conf[k][0] == dirs[i]:
            confValue = ",".join(string for string in conf[k][1:] if len(string) > 0)
            print("\t\t%s"%(confValue.replace(',', '<br>')));
    print("\t\t</td>");    
    for k in range(len(boxLines)):
        if boxLines[k][0] == dirs[i]:
            x = []
            y = []
            w = []
            h = []
            label = []
            c = []
            nl = 1
            nx = 2
            ny = 3
            nw = 4
            nh = 5 
            nc = 6
            while nh < len(boxLines[k]):
                label.append(boxLines[k][nl])
                w.append(float(boxLines[k][nw]) * 416)
                h.append(float(boxLines[k][nh]) * 416)
                w_val = float(boxLines[k][nw])*208
                h_val = float(boxLines[k][nh])*208
                x.append(float(boxLines[k][nx]) * 416 - w_val)
                y.append(float(boxLines[k][ny]) * 416 - h_val)
                c.append(boxLines[k][nc])
                nl += 6 
                nx += 6
                ny += 6
                nw += 6
                nh += 6
                nc += 6
            checked_box[i] = True   
            print("\t\t</tr>");
            print("\t\t");
            print("\t\t<script>");
            print("\t\tvar modal = document.getElementById('myModal');");
            print("\t\tvar img%d = document.getElementById(\"myImg%d\");"%(i,i));       
            print("\t\tvar modalImg = document.getElementById(\"img01\");");
            print("\t\tvar captionText = document.getElementById(\"caption\");");
            print("\t\tvar image = img%d.addEventListener(\"click\", onImgClick, false);"%(i));
            print("\t\tfunction onImgClick(myImg%d) {"%(i));            
            print("\t\tif(document.boxTypeForm.TypeOfBox[1].checked == true) {");
            print("\t\tvar canvasid = document.getElementById('myCanvas');");
            print("\t\tvar ctx = canvasid.getContext(\"2d\");");
            print("\t\tctx.drawImage(img%d,0,0);"%(i));
            print("\t\tmodal.style.display = \"block\";");
            print("\t\tcaptionText.innerHTML = this.alt;");
            print("\t\tmyModal.src = \"ctx\";");
            if x and y and w  and h and label:
                print("\t\tmyModal.onload = drawBoundingBox({},{},{},{},{},{});".format(x,y,w,h,label,c));
                print("\t\t}");

            print("\t\tif(document.boxTypeForm.TypeOfBox[2].checked == true) {");
            print("\t\tvar canvasid = document.getElementById('myCanvas');");
            print("\t\tvar ctx = canvasid.getContext(\"2d\");");
            print("\t\tctx.drawImage(img%d,0,0);"%(i));
            print("\t\tmodal.style.display = \"block\";");
            print("\t\tcaptionText.innerHTML = this.alt;");
            print("\t\tmyModal.src = \"ctx\";");
            if x and y and w  and h and label:
                print("\t\tmyModal.onload = drawBoundingBox({},{},{},{},{},{});".format(x,y,w,h,label,c));
    for k in range(len(gtLines)):
        if gtLines[k][0] == dirs[i]:
            x = []
            y = []
            w = []
            h = []
            label = []
            nl = 1
            nx = 2
            ny = 3
            nw = 4
            nh = 5 
            while nh < len(gtLines[k]):
                label.append(gtLines[k][nl])
                w.append(float(gtLines[k][nw]) * 416)
                h.append(float(gtLines[k][nh]) * 416)
                w_val = float(gtLines[k][nw])*208
                h_val = float(gtLines[k][nh])*208
                x.append(float(gtLines[k][nx]) * 416 - w_val)
                y.append(float(gtLines[k][ny]) * 416 - h_val)
                nl += 6 
                nx += 6
                ny += 6
                nw += 6
                nh += 6
            checked_gt[i] = True
            if checked_box[i] ==  True:
                if x and y and w  and h and label:
                    print("\t\tmyModal.onload = drawBoundingBoxGT({},{},{},{},{});".format(x,y,w,h,label));
                    print("\t\t}");
            elif checked_box[i] == False:
                print("\t\t</tr>");
                print("\t\t");
                print("\t\t<script>");                
                print("\t\tvar modal = document.getElementById('myModal');");
                print("\t\tvar img%d = document.getElementById(\"myImg%d\");"%(i,i));
                print("\t\tvar modalImg = document.getElementById(\"img01\");");
                print("\t\tvar captionText = document.getElementById(\"caption\");");
                print("\t\tvar image = img%d.addEventListener(\"click\", onImgClick, false);"%(i));
                print("\t\tfunction onImgClick(myImg%d) {"%(i)); 
                print("\t\tif(document.boxTypeForm.TypeOfBox[1].checked == true) {")
                print("\t\tvar canvasid = document.getElementById('myCanvas');");
                print("\t\tvar ctx = canvasid.getContext(\"2d\");");
                print("\t\tctx.drawImage(img%d,0,0);"%(i));
                print("\t\tmodal.style.display = \"block\";");
                print("\t\tcaptionText.innerHTML = this.alt;");
                print("\t\t}");           
                print("\t\tif(document.boxTypeForm.TypeOfBox[2].checked == true) {")
                print("\t\tvar canvasid = document.getElementById('myCanvas');");
                print("\t\tvar ctx = canvasid.getContext(\"2d\");");
                print("\t\tctx.drawImage(img%d,0,0);"%(i));
                print("\t\tmodal.style.display = \"block\";");
                print("\t\tcaptionText.innerHTML = this.alt;");
                print("\t\tmyModal.src = \"ctx\";");
                if x and y and w  and h and label:
                    print("\t\tmyModal.onload = drawBoundingBoxGT({},{},{},{},{});".format(x,y,w,h,label));
                print("\t\t}");           
            print("\t\t");         
            print("\t\tif(document.boxTypeForm.TypeOfBox[0].checked == true) {")
            print("\t\tvar canvasid = document.getElementById('myCanvas');");
            print("\t\tvar ctx = canvasid.getContext(\"2d\");");
            print("\t\tctx.drawImage(img%d,0,0);"%(i));
            print("\t\tmodal.style.display = \"block\";");
            print("\t\tcaptionText.innerHTML = this.alt;");
            print("\t\tmyModal.src = \"ctx\";");
            if x and y and w  and h and label:
                print("\t\tmyModal.onload = drawBoundingBoxGT({},{},{},{},{});".format(x,y,w,h,label));
            print("\t\t}");

    if checked_gt[i] == False and checked_box[i] == True:        
        print("\t\t}");
        print("\t\tif(document.boxTypeForm.TypeOfBox[0].checked == true) {")
        print("\t\tvar canvasid = document.getElementById('myCanvas');");
        print("\t\tvar ctx = canvasid.getContext(\"2d\");");
        print("\t\tctx.drawImage(img%d,0,0);"%(i));
        print("\t\tmodal.style.display = \"block\";");
        print("\t\tcaptionText.innerHTML = this.alt;");
        print("\t\tmyModal.src = \"ctx\";");
        print("\t\t}");

    if checked_box[i] == True or checked_gt[i] == True:
        print("\t\tvar span = document.getElementsByClassName(\"modal\")[0];");
        print("\t\tspan.onclick = function() { modal.style.display = \"none\"; ");
        print("\t\t}}");
        print("\t\t</script>");     
        print("\t\t");
    if checked_box[i] == False and checked_gt[i] == False:
        print("\t\t");
        print("\t\t<script>");
        print("\t\tvar modal = document.getElementById('myModal');");
        print("\t\tvar img%d = document.getElementById(\"myImg%d\");"%(i,i));       
        print("\t\tvar modalImg = document.getElementById(\"img01\");");
        print("\t\tvar captionText = document.getElementById(\"caption\");");
        print("\t\tvar image = img%d.addEventListener(\"click\", onImgClick, false);"%(i));
        print("\t\tfunction onImgClick(myImg%d) {"%(i));
        print("\t\tvar canvasid = document.getElementById('myCanvas');");
        print("\t\tvar ctx = canvasid.getContext(\"2d\");");
        print("\t\tctx.drawImage(img%d,0,0);"%(i));
        print("\t\tmodal.style.display = \"block\";");
        print("\t\tcaptionText.innerHTML = this.alt;");
        print("\t\tmyModal.src = \"ctx\";");
        print("\t\tvar span = document.getElementsByClassName(\"modal\")[0];");
        print("\t\tspan.onclick = function() { modal.style.display = \"none\"; ");
        print("\t\t}}");
        print("\t\t</script>");     
        print("\t\t");
print("</table>");

# Compare result summary

SummaryFileName = '';
FolderName = os.path.expanduser("~/.adatCompare")
if not os.path.exists(FolderName):
    os.makedirs(FolderName);

ModelFolderName = FolderName; ModelFolderName +="/"; ModelFolderName += modelName;
if not os.path.exists(ModelFolderName):
    os.makedirs(ModelFolderName);

SummaryFileName = ModelFolderName; SummaryFileName += "/modelRunHistoryList.csv";


# write summary details into csv
if os.path.exists(SummaryFileName):
    sys.stdout = open(SummaryFileName,'a')
    print("%s, %s, %d, %d, %d"%(modelName,dataFolder,numElements,totalMatch,totalMismatch));
else:
    sys.stdout = open(SummaryFileName,'w')
    print("Model Name, Image DataBase, Number Of Images, Match, MisMatch");
    print("%s, %s, %d, %d, %d"%(modelName,dataFolder,numElements,totalMatch,totalMismatch));
sys.stdout.close()

# write into HTML
savedResultElements = 0;
with open(SummaryFileName) as savedResultFile:
    savedResultFileCSV = csv.reader(savedResultFile)
    next(savedResultFileCSV, None) # skip header
    savedResultDataBase = [r for r in savedResultFileCSV]
    savedResultElements = len(savedResultDataBase)

sys.stdout = orig_stdout
print("results saved in backup drive")


sys.stdout = open(toolKit_dir+'/index.html','a')
print("\t<!-- Compare ResultSummary -->");
print("<A NAME=\"table5\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Compare Results Summary</em></font></h1></A>");
print("\t<!-- Compare Graph Script -->");
print("\t<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>");
print("\t<script type=\"text/javascript\">");
print("\t");

lineNumber = 0;
while lineNumber < savedResultElements:
    print("\tgoogle.charts.load('current', {'packages':['bar']});");
    print("\tgoogle.charts.setOnLoadCallback(drawChart_%d);"%(lineNumber));
    print("\tfunction drawChart_%d(){"%(lineNumber));
    print("\tvar data = google.visualization.arrayToDataTable([");
    print("\t['  '     ,  'Match'  , 'Mismatch'],");
    print("\t['Summary',   %d     , %d        ]"%(int(savedResultDataBase[lineNumber][3]),int(savedResultDataBase[lineNumber][4])));                        
    print("\t]);");
    print("\tvar options = { title: '%s Overall Result Summary', vAxis: { title: 'Images' }, width: 400, height: 400 };"%(savedResultDataBase[lineNumber][0]));
    print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats_%d'));"%(lineNumber));
    print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
    print("\t");

    lineNumber += 1;

print("\t");

# draw combined graph
print("\tgoogle.charts.load('current', {'packages':['bar']});");
print("\tgoogle.charts.setOnLoadCallback(drawChart_master);");
print("\tfunction drawChart_master(){");
print("\tvar data = google.visualization.arrayToDataTable([");
print("\t['Model'   ,'Match',   'Mismatch'],");
for i in range(savedResultElements):
    if(i != (lineNumber-1)):
        print("\t['%s'   ,%d ,   %d],"%(savedResultDataBase[i][0],int(savedResultDataBase[i][3]),int(savedResultDataBase[i][4])));
    else:
        print("\t['%s'   ,%d ,   %d]"%(savedResultDataBase[i][0],int(savedResultDataBase[i][3]),int(savedResultDataBase[i][4])));

print("\t]);");
print("\tvar options = { title: 'Overall Result Summary', vAxis: { title: 'Images' }, width: 800, height: 600, bar: { groupWidth: '30%' }, isStacked: true , series: { 0:{color:'green'},1:{color:'Indianred'} }};");
print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats_master'));");
print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
print("\t");
print("\t</script>");

# draw graph
print("\t");
print("\t");
print("\t<center><div id=\"Model_Stats_master\" style=\"border: 1px solid #ccc\"></div></center>");
print("\t");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t<tr>");
i = 1;
while i <= savedResultElements:
    print("\t");
    print("\t<td><center><div id=\"Model_Stats_%d\" style=\"border: 1px solid #ccc\"></div></center></td>"%(i));
    if( i % 3 == 0):
        print("\t</tr>");
        print("\t<tr>");
    i += 1;

print("\t</tr>");
print("\t</table>");

#Model Score
print("\t<!-- Model Score -->");
print("<A NAME=\"table6\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Model Score</em></font></h1></A>");
print("\t");
#Standard Method
print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Standard Scoring</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(totalMatch));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyMatch));
print("\t\t</tr>");
print("</table>");
print("\t");

#calculating values for scoring
top1PassScore = 0
top1FailScore = 0
i = 99
confID = 0.99
while i >= 0:
    top1PassScore += confID * values_match[i];
    top1FailScore += confID * values_mismatch[i];
    i = i - 1


#Method 1 - Confidence Aware
top1Score = float(top1PassScore);
ModelScoreTop1 = (top1Score/totalGroundTruth)*100;
print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Method 1 Scoring - Confidence Aware</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(totalMatch));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t</tr>");
print("</table>");
print("\t");

#Method 2 - Error Aware
top1Score =  top1PassScore - top1FailScore
ModelScoreTop1 = (top1Score/totalGroundTruth)*100
print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Method 2 Scoring - Error Aware</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(totalMatch));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t</tr>");
print("</table>");
print("\t");
print("\t");

top1ModelScore = [];
for i in range(100):
    top1ModelScore.append([])
    for j in range(3):
        top1ModelScore[i].append(0)

for i in range(100):
    top1ModelScore[i][0] = 0;
    top1ModelScore[i][1] = 0; 
    top1ModelScore[i][2] = 0; 

standardPassTop1 = 0;

Top1PassScore = 0; 
Top1FailScore = 0; 

confID = 0.99;
i = 99;
while i >= 0:
    Top1PassScore += confID * values_match[i];
    Top1FailScore += confID * values_mismatch[i];
    
    # method 1
    top1ModelScore[i][1] = (float(Top1PassScore)/totalGroundTruth)*100;
    
    # method 2
    top1ModelScore[i][0] = (float(Top1PassScore - Top1FailScore)/totalGroundTruth)*100;
  
    # standard method
    standardPassTop1 += float(values_match[i]);
    top1ModelScore[i][2] = (standardPassTop1/totalGroundTruth)*100;
    
    confID = confID - 0.01;
    i = i - 1;

print("\t<script type=\"text/javascript\">");
print("\t");
# Top 1 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top1ScoreGraph);");
print("\tfunction Top1ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f,  %.2f,   %.2f,  %.2f]"%(fVal,top1ModelScore[i][2],top1ModelScore[i][1],top1ModelScore[i][0]));
  else:
    print("\t[%.2f,  %.2f,   %.2f,   %.2f],"%(fVal,top1ModelScore[i][2],top1ModelScore[i][1],top1ModelScore[i][0]));

  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 1', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top1_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

# Standard Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(StandardTop5Graph);");
print("\tfunction StandardTop5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addRows([");
print("\t[1, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.2f]"%(fVal,top1ModelScore[i][2]));
  else:
    print("\t[%.2f, %.2f],"%(fVal,top1ModelScore[i][2]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Standard Scoring Method', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('standard_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Method 1 Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Method1Top5Graph);");
print("\tfunction Method1Top5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addRows([");
print("\t[1, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f]"%(fVal,top1ModelScore[i][1]));
  else:
    print("\t[%.2f, %.4f],"%(fVal,top1ModelScore[i][1]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Method 1 Scoring', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('method_1_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
#Method 2 Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Method2Top5Graph);");
print("\tfunction Method2Top5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addRows([");
print("\t[1, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f]"%(fVal,top1ModelScore[i][0]));
  else:
    print("\t[%.2f, %.4f],"%(fVal,top1ModelScore[i][0]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Method 2 Scoring', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('method_2_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

print("\t");
print("\t</script>");
print("\t");
print("\t");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t<tr>");
print("\t <td><center><div id=\"Top1_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"standard_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"method_1_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"method_2_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t <td><center></center></td>");
print("\t</tr>");
print("\t</table>");
print("\t");


# HELP
print ("\t<!-- HELP -->");
print ("<A NAME=\"table7\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>HELP</em></font></h1></A>");
print ("\t");
print ("\t<table align=\"center\" style=\"width: 50%\">");
print ("\t<tr><td>");
print ("\t<h1 align=\"center\">AMD Neural Net ToolKit</h1>");
print ("\t</td></tr><tr><td>");
print ("\t<p>AMD Neural Net ToolKit is a comprehensive set of help tools for neural net creation, development, training and");
print ("\tdeployment. The ToolKit provides you with help tools to design, develop, quantize, prune, retrain, and infer your neural");
print ("\tnetwork work in any framework. The ToolKit is designed to help you deploy your work to any AMD or 3rd party hardware, from ");
print ("\tembedded to servers.</p>");
print ("\t<p>AMD Neural Net ToolKit provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle,");
print ("\tfrom creating a model to deploying them for your target platforms.</p>");
print ("\t<h2 >List of Features Available in this release</h2>");
print ("\t<ul>");
print ("\t<li>Overall Summary</li>");
print ("\t<li>Graphs</li>");
#print ("\t<li>Hierarchy</li>");
print ("\t<li>Labels</li>");
print ("\t<li>Image Results</li>");
print ("\t<li>Compare</li>");
print ("\t<li>Help</li>");
print ("\t</ul>");
print ("\t<h3 >Overall Summary</h3>");
print ("\t<p>This section summarizes the results for the current session, with information on the dataset and the model.");
print ("\tThe section classifies the dataset into images with or without ground truth and only considers the images with ground truth ");
print ("\tfor analysis to avoid skewing the results.</p>");
print ("\t<p>The summary calculates all the metrics to evaluate the current run session, helps evaluate the quality of the data set,");
print ("\taccuracy of the current version of the model and links all the high level result to individual images to help the user to ");
print ("\tquickly analyze and detect if there are any problems.</p>");
print ("\t<p>The summary also timestamps the results to avoid confusion with different iterations.</p>");
print ("\t<h3>Graphs</h3>");
print ("\t<p>The graph section allows the user to visualize the dataset and model accurately. The graphs can help detect any");
print ("\tanomalies with the data or the model from a higher level. The graphs can be saved or shared with others.</p>");
"""
print ("\t<h3 >Hierarchy</h3>");
print ("\t<p>This section has AMD proprietary hierarchical result analysis. Please contact us to get more information.</p>");
"""
print ("\t<h3 >Labels</h3>");
print ("\t<p>Label section is the summary of all the classes the model has been trained to detect. The Label Summary presents the");
print ("\thighlights of all the classes of images available in the database. The summary reports if the classes are found or not ");
print ("\tfound.</p>");
print ("\t<p>Click on any of the label description and zoom into all the images from that class in the database.</p>");
print ("\t<h3 >Image Results</h3>");
print ("\t<p>The Image results has all the low level information about each of the individual images in the database. It reports on ");
print ("\tthe results obtained for the image in the session and allows quick view of the image.</p>");
print ("\t<img \" src=\"icons/sampleOutput.png\" alt=\"Sample Output\" /></a>");
print ("\t<br>The blue boxes represent the ground truth. The green boxes represent the boudning boxes identified by a framework such as YOLOv2, and shows the confidence with which the output was predicted.");
print ("\t<h3 >Compare</h3>");
print ("\t<p>This section compares the results of a database or the model between different sessions. If the database was tested with");
print ("\tdifferent models, this section reports and compares results among them.</p>");
print ("\t</td></tr>");
print ("\t</table>");
print ("\t<br><br><br>");
#TBD: symbol
print ("\t\t<div class=\"footer\"> <p>2018 - 2024 Advanced Micro Devices, Inc</p></div>");
print ("\t");
print ("\n</body>");
print ("\n</html>");

sys.stdout = orig_stdout
print("index.html generated")
exit (0)
