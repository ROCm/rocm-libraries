# Object Detection Verification Tool
## Introduction:
This tool allows you to do object detection for images. It is a command line utility to match the bounding box tool output to the ground truth for the given set of images. Both the files contain details for each bounding box of an image. The details included are:
1.	Label for each bounding box
2.	Center coordinates of a bounding box (x,y)
3.	Width and height of the bounding box (w,h)

## Details:
### 1.	Ground Truth Directory:
•	The ground truth directory has one file (.ann: which is a text file) for each image. 
•	Each image contains details on all the bounding boxes for that image.
•	The details about each bounding box i.e: center coordinates and, width and height are in normalized format (between 0 -1).

### 2.	Bounding Box Tool Output File:
•	The bounding box tool output file is a single .csv file that contains all the bounding boxes for all the images.
•	The details for each bounding box can be either in normalized format or pixel format. This should be specified as an input in the command line utility.
•	Example of a tool used is YOLO v2.

### 3.	Comparison:
•	Both the files were parsed and stored in a similar data structure. These data structures could then be compared directly.
•	The compared results were written to a file and stored in a data structure.

### 4.	Results:
•	The results could be one of the following:
1.	Match: The bounding box was present in both the files and their parameters were within acceptable range.

2.	IOU Mismatch: The bounding box was present in both the files and their labels matched, but their parameters were beyond acceptable range.

3.	Label Mismatch: The bounding box was present in both the files and their parameters were within acceptable range, but their labels did not match.

4.	Unknown: The ground truth for the bounding box was not available.

5.	Not Found: The bounding box was not detected by the tool but is present in Ground Truth.
## Usage:
The tool was written in C++
Step 1: Compile using C++11:
```
g++ std=c++11 <cpp file> -o <output file>

Example: g++ std=c++11 main.cpp -o main
```
Step 2: Run the program:
```
./main  -i <bounding box tool output file> -g <ground truth directory> -o <results file> -m <mode> -iou <IOU value> - w <width of image> -h <height of image>
```
-i: <output of the bounding box tool CSV file> - required (eg: Using a yolo output)

-g: <ground truth directory> - required
-o: <Output results file> - required
-m: <mode indicating type of input> (can be ‘normalized’ or ‘pixel’ – ie: format of x,y,w,h) – required
-iou: <IOU value> (Intersection over Union Value – float value between 0 and 1) – required
-w: <width of image> (specify width of original image if mode is ‘pixel’) – required depending on mode
-h: <height of image> (specify height of original image if mode is ‘pixel’) – required depending on mode 
 ```
Example: 
./main -i yolo_out_pixel.csv -g ground_truth/ -o outputFile.txt -m pixel -iou 0.5 -w 416 -h 416
(Or)
./main -i yolo_out_normalized.csv -g ground_truth/ -o outputFile.txt -m normalized -iou 0.5 
```
## Glossary:
•	YOLO: Real-time object detection system.

