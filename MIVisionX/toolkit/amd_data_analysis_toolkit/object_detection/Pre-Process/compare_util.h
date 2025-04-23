/* Project: YOLO Result Verification Tool Version 2
 * Date: 08/23/2018
 * Author: Lakshmi 
 * Version: Correct structure for GT and YOLO. Includes confidence!
*/

#ifndef COMPARE_UTIL_V3_H
#define COMPARE_UTIL_V3_H

#include<string>
#include<vector>
using namespace std; 

//struct to hold data of a bounding box 
struct box
{
	string label;
	double x_normalized;
	double y_normalized;
	double w_normalized;
	double h_normalized;
	double confidence;
	bool checked;
};

//struct to hold data for an image
struct image
{
	string image_name;
	vector<box> boxes;																	/*vector to hold data for multiple bounding boxes of the same image of ground truth*/										
};

//struct to hold the compared results for a bounding box
struct comparedResults_boxes
{
	string label;
	string match_value;
	double confidence;
};

//struct to hold compared results for an image
struct comparedResults_image
{
	string image_name;
	vector<comparedResults_boxes> boxes;
};


//function declarations
void parseGTFile(const string& filename);												
void parseFileNormalized(const string& filename);		
void parseFilePixel(const string& filename, int width, int height);	
void computeNormalizedValues(string image_name, string class_name, string x_1, string y_1, string x_2, string y_2, string confidence, int width_image, int height_image);				
void compareValues(float IOU);																
void display();			
void writeToFile(const string &filename);												

#endif