/* Project: YOLO Result Verification Tool Version 2
 * Date: 08/30/2018
 * Author: Lakshmi 
 * Version: v4 + outputFolder + forming a single file for gt & yolo
*/

#include "compare_util.h"
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
using namespace std;

vector<image> gt_images;								/*vector to hold data of multiple images of ground truth*/
vector<image> yolo_images;								/*vector to hold data of multiple images of yolo output*/
vector<comparedResults_image> results;					/*vector to hold the results of all the compared images*/

//function to process Ground Truth file
void parseGTFile(const string& filename)
{

	ifstream groundTruthFile;
	groundTruthFile.open(filename);
	if(groundTruthFile.fail())
		fprintf(stderr, "Cannot open ground truth file\n\n");

	box gtBox_object;								/*object for a bounding box of ground truth file*/	
	image gtImage_object;							/*object for one image of ground truth*/

	string image_name;
	getline(groundTruthFile, image_name, ',');	
	size_t found = image_name.find(".");
	image_name = image_name.substr(0,found);
	
	//continue reading line 1
	string numBoundingBoxes;
	getline(groundTruthFile, numBoundingBoxes);

	//parameters read from the file
	string id;
	string label;
	string importance;
	string x;
	string y;
	string w;
	string h;

	
	while(!groundTruthFile.eof())
	{
		//reading each parameter in the line
		getline(groundTruthFile, id, ',');
		getline(groundTruthFile, label, ',');
		getline(groundTruthFile, importance, ',');
		getline(groundTruthFile, x, ',');
		getline(groundTruthFile, y, ',');
		getline(groundTruthFile, w, ',');
		getline(groundTruthFile, h);

		//parameters stored in the structure for ground truth box
		gtBox_object.label = label;
		gtBox_object.x_normalized = stod(x);
		gtBox_object.y_normalized = stod(y);
		gtBox_object.w_normalized = stod(w);
		gtBox_object.h_normalized = stod(h);
		gtBox_object.confidence = 1.00;
		gtBox_object.checked = false;

		gtImage_object.boxes.push_back(gtBox_object);
	}	

	gtImage_object.image_name = image_name;


	gt_images.push_back(gtImage_object);

	
	groundTruthFile.close();

}


void parseFileNormalized(const string& filename)
{
	
	ifstream inputFile;
	inputFile.open(filename);
	if(inputFile.fail())
		fprintf(stderr, "Cannot open bounding box tool results file\n\n");

	string image_name;
	string x_normalized;
	string y_normalized;
	string w_normalized;
	string h_normalized;
	string confidence;
	string classValue;	
	string class_name;

	//ignoring the first line of file
	string dummyLine;
	getline(inputFile, dummyLine);								

	//reading each parameter in the line the first time
	getline(inputFile, image_name, ',');
	getline(inputFile, x_normalized, ',');
	getline(inputFile, y_normalized, ',');
	getline(inputFile, w_normalized, ',');
	getline(inputFile, h_normalized, ',');
	getline(inputFile, confidence, ',');
	getline(inputFile, classValue, ',');
	getline(inputFile, class_name);

	while(inputFile)
	{
		//class_name.pop_back();

		box this_label;				/*object for bounding box of current YOLO output*/
		this_label.label = class_name;
		this_label.x_normalized = stod(x_normalized);
		this_label.y_normalized = stod(y_normalized);
		this_label.w_normalized = stod(w_normalized);
		this_label.h_normalized = stod(h_normalized);
		this_label.confidence = stod(confidence);
		this_label.checked = false;
		
		size_t found = image_name.find(".");
		image_name = image_name.substr(0,found);

		bool found_image = false;
		for(unsigned int i = 0; i < yolo_images.size(); i++)
		{
			if(yolo_images[i].image_name == image_name)
			{
				yolo_images[i].boxes.push_back(this_label);
				found_image = true;
				break;
			}
		}
		if(!found_image)
		{
			image newImage;			/*object for image of a new image from YOLO output*/
			newImage.image_name = image_name;
			newImage.boxes.push_back(this_label);
			yolo_images.push_back(newImage);
		}
			
		
		//reading each parameter in the line for consecutive lines
		getline(inputFile, image_name, ',');
		getline(inputFile, x_normalized, ',');
		getline(inputFile, y_normalized, ',');
		getline(inputFile, w_normalized, ',');
		getline(inputFile, h_normalized, ',');
		getline(inputFile, confidence, ',');
		getline(inputFile, classValue, ',');
		getline(inputFile, class_name);
		
	}

	inputFile.close();
}


//fucntion to process YOLO Output file
void parseFilePixel(const string& filename, int width_image, int height_image)
{
	
	ifstream inputFile;
	inputFile.open(filename);
	if(inputFile.fail())
		fprintf(stderr, "Cannot open bounding box tool results file\n\n");
	
	string imageName;
	string leftTop_x;
	string leftTop_y;
	string rightBottom_x;
	string rightBottom_y;
	string confidence;
	string classValue;	
	string class_name;

	//ignoring the first line of file
	string dummyLine;
	getline(inputFile, dummyLine);								

	//reading each parameter in the line the first time
	getline(inputFile, imageName, ',');
	getline(inputFile, leftTop_x, ',');
	getline(inputFile, leftTop_y, ',');
	getline(inputFile, rightBottom_x, ',');
	getline(inputFile, rightBottom_y, ',');
	getline(inputFile, confidence, ',');
	getline(inputFile, classValue, ',');
	getline(inputFile, class_name);

	while(inputFile)
	{
		//call fucntion to compute normalized values
		computeNormalizedValues(imageName, class_name, leftTop_x, leftTop_y, rightBottom_x, rightBottom_y, confidence, width_image, height_image);
		
		//reading each parameter in the line for consecutive lines
		getline(inputFile, imageName, ',');
		getline(inputFile, leftTop_x, ',');
		getline(inputFile, leftTop_y, ',');
		getline(inputFile, rightBottom_x, ',');
		getline(inputFile, rightBottom_y, ',');
		getline(inputFile, confidence, ',');
		getline(inputFile, classValue, ',');
		getline(inputFile, class_name);
		
	}

	inputFile.close();
}

//funtion to normalize the values obtained by parsing YOLO Output
void computeNormalizedValues(string image_name, string class_name, string x_1, string y_1, string x_2, string y_2, string conf, int width_image, int height_image)
{
	size_t found = image_name.find(".");
	image_name = image_name.substr(0,found);


	double leftTop_x = stod(x_1);
	double leftTop_y = stod(y_1);
	double rightBottom_x = stod(x_2);
	double rightBottom_y = stod(y_2);


	double width_box = rightBottom_x - leftTop_x;
	double height_box = rightBottom_y - leftTop_y;

	double x_center = (leftTop_x + rightBottom_x)/2;
	double y_center = (leftTop_y + rightBottom_y)/2;

	double x_normalized = x_center/width_image;
	double y_normalized = y_center/height_image;

	double w_normalized = width_box/width_image;
	double h_normalized = height_box/height_image;

	double confidence = stod(conf);

	box this_label;				/*object for bounding box of current YOLO output*/
	this_label.label = class_name;
	this_label.x_normalized = x_normalized;
	this_label.y_normalized = y_normalized;
	this_label.w_normalized = w_normalized;
	this_label.h_normalized = h_normalized;
	this_label.confidence = confidence;
	this_label.checked = false;
	
	bool found_image = false;
	for(unsigned int i = 0; i < yolo_images.size(); i++)
	{
		if(yolo_images[i].image_name == image_name)
		{
			yolo_images[i].boxes.push_back(this_label);
			found_image = true;
			break;
		}
	}
	if(!found_image)
	{
		image newImage;			/*object for image of a new image from YOLO output*/
		newImage.image_name = image_name;
		newImage.boxes.push_back(this_label);
		yolo_images.push_back(newImage);
	}

}



void display()
{/*
	for(unsigned int i = 0; i < gt_images.size(); i++)
	{
		cout << "Image Name: " << gt_images[i].image_name << endl;
		for( unsigned int j = 0; j < gt_images[i].boxes.size(); j++)
		{ 	
			cout<< gt_images[i].boxes[j].label << " " << gt_images[i].boxes[j].x_normalized << " " << gt_images[i].boxes[j].y_normalized << " " 
			<< gt_images[i].boxes[j].w_normalized << " " << gt_images[i].boxes[j].h_normalized << " " << gt_images[i].boxes[j].confidence << " " << gt_images[i].boxes[j].checked << endl;
		}
	} */
	cout << "\n\n YOLO FILE: \n\n"; 
	for(unsigned int i = 0; i < yolo_images.size(); i++)
	{
		cout << "Image Name: " << yolo_images[i].image_name << endl;
		for(unsigned int j = 0; j < yolo_images[i].boxes.size(); j++)
		{ 	
			cout<< yolo_images[i].boxes[j].label << " " << yolo_images[i].boxes[j].x_normalized << " " << yolo_images[i].boxes[j].y_normalized << " " 
			<< yolo_images[i].boxes[j].w_normalized << " " << yolo_images[i].boxes[j].h_normalized << " " << yolo_images[i].boxes[j].confidence << " " << yolo_images[i].boxes[j].checked<< endl;
		}
	} /*
cout << "\n\n RESULTS: \n\n"; 

	for(unsigned int i = 0; i < results.size(); i++)
	{
		cout << endl << "Image Name: " << results[i].image_name << endl;
		for(unsigned int j = 0; j < results[i].boxes.size(); j++)
		{ 	
			cout<< results[i].boxes[j].label << " " << results[i].boxes[j].match_value << " " << results[i].boxes[j].confidence << endl;
		}
	}*/

}

/*Function to compare values of yolo output with ground truth:
 *1. Create a file and append values to it depending on match/mismatch
 *2. Store the results in a vector containing struct
 *3. Match = when label, center coordinates and height & width match
 *4. IOU Mismatch = when label matches but center coordinates and height & width do not match
 *5. Label Mismatch = when center corrdinates and height & width match but label does not match
 *6. Unknown = when a bounding box present in YOLO isn't present in Ground Truth
 */

void compareValues(float IOU)
{
	
	for(unsigned int j = 0; j < yolo_images.size(); j++)
    {  	
    	//flag to keep track to write to structure
    	bool flag_image = false;	

    	comparedResults_image result_image;		/*image object for results*/

    	for(unsigned int i = 0; i < gt_images.size(); i++)
      	{
      		
			comparedResults_boxes result_box;	/*bounding box object for results*/
    		if(yolo_images[j].image_name == gt_images[i].image_name)
    		{   
				for(unsigned int yolo_ind = 0; yolo_ind < yolo_images[j].boxes.size(); yolo_ind++)
				{
					flag_image = true;
					result_image.image_name = yolo_images[j].image_name;

					bool flag_label = false, flag_IOU = false;
		    		for(unsigned int gt_ind = 0;  gt_ind < gt_images[i].boxes.size(); gt_ind++)
					{	
						if(gt_images[i].boxes[gt_ind].label == yolo_images[j].boxes[yolo_ind].label)
    					{
    						flag_label = true;
    						gt_images[i].boxes[gt_ind].checked = true;
    					}

    					if( fabs(gt_images[i].boxes[gt_ind].x_normalized - yolo_images[j].boxes[yolo_ind].x_normalized) <= IOU
	    						&& fabs(gt_images[i].boxes[gt_ind].y_normalized - yolo_images[j].boxes[yolo_ind].y_normalized) <= IOU
	    						&& fabs(gt_images[i].boxes[gt_ind].w_normalized - yolo_images[j].boxes[yolo_ind].w_normalized) <= IOU
	    						&& fabs(gt_images[i].boxes[gt_ind].h_normalized - yolo_images[j].boxes[yolo_ind].h_normalized) <= IOU)
    					{					
	    					flag_IOU = true;
	    					gt_images[i].boxes[gt_ind].checked = true;
	    				}
	    			}

	    			if(flag_label && flag_IOU)
	    			{
						result_box.label = yolo_images[j].boxes[yolo_ind].label;
						result_box.confidence = yolo_images[j].boxes[yolo_ind].confidence;
						result_box.match_value = "Match";
					
		    		}
		    		else if(flag_label && !flag_IOU)
	    			{
						result_box.label = yolo_images[j].boxes[yolo_ind].label;
						result_box.confidence = yolo_images[j].boxes[yolo_ind].confidence;
						result_box.match_value = "IOU Mismatch";
					}
		    		else if(!flag_label && flag_IOU)
	    			{
						result_box.label = yolo_images[j].boxes[yolo_ind].label;
						result_box.confidence = yolo_images[j].boxes[yolo_ind].confidence;
						result_box.match_value = "Label Mismatch";
					}
		    		else if(!flag_label && !flag_IOU)
	    			{
						result_box.label = yolo_images[j].boxes[yolo_ind].label;
						result_box.confidence = yolo_images[j].boxes[yolo_ind].confidence;
						result_box.match_value = "Unknown";
					}


	    	
		    		result_image.boxes.push_back(result_box);
		    		
    			}
    			
    			for(unsigned int gt_ind = 0;  gt_ind < gt_images[i].boxes.size(); gt_ind++)
				{
					if(!gt_images[i].boxes[gt_ind].checked)
					{
						result_box.label = gt_images[i].boxes[gt_ind].label;
						result_box.confidence = gt_images[i].boxes[gt_ind].confidence;
						result_box.match_value = "Not Found";
						result_image.boxes.push_back(result_box);
					}	
				}
    			
    		}

    	}

    	if(flag_image)
    		results.push_back(result_image);

    }
}


void writeToFile(const string &filename)
{

	char cwd[PATH_MAX];
	string path; 
    if(getcwd(cwd, sizeof(cwd)) != NULL)
    {
    	path = cwd;
    }
    string folder = path + "/" + "outputFolder";

    int status = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    //file to store the results
	string outputFile = folder + "/" + filename;

	FILE *fp_outputFile = fopen(outputFile.c_str(), "w");
	fprintf(fp_outputFile, "image_number,label,result,confidence,label,result,confidence,label,result,confidence,label,result,confidence\n");

	if(fp_outputFile == NULL)
		fprintf(stderr, "Cannot open results file\n\n");

						
	for(unsigned int i = 0; i < results.size(); i++)
	{
		results[i].image_name += ".JPEG";
		fprintf(fp_outputFile, "%s,", results[i].image_name.c_str());
		for(unsigned int j = 0; j < results[i].boxes.size(); j++)
		{ 	
			fprintf(fp_outputFile, "%s,%s,%lf,",  results[i].boxes[j].label.c_str(),results[i].boxes[j].match_value.c_str(),results[i].boxes[j].confidence);
		}
		fprintf(fp_outputFile, "\n");
	} 

	fclose(fp_outputFile);

	string gtFile = folder + "/" + "groundTruthFile.csv";
	FILE *fp_groundtruthFile = fopen(gtFile.c_str(), "w");
	if(fp_groundtruthFile == NULL)
		fprintf(stderr, "Cannot open groundTruthFile file to write\n\n");

	fprintf(fp_groundtruthFile, "image_number,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence\n");


	for(unsigned int i = 0; i < gt_images.size(); i++)
	{
		gt_images[i].image_name += ".JPEG";
		fprintf(fp_groundtruthFile, "%s,", gt_images[i].image_name.c_str());
		for(unsigned int j = 0; j < gt_images[i].boxes.size(); j++)
		{ 	
			fprintf(fp_groundtruthFile, "%s,%lf,%lf,%lf,%lf,%lf,", gt_images[i].boxes[j].label.c_str(),gt_images[i].boxes[j].x_normalized,gt_images[i].boxes[j].y_normalized,gt_images[i].boxes[j].w_normalized,
				gt_images[i].boxes[j].h_normalized,gt_images[i].boxes[j].confidence);
		}
		fprintf(fp_groundtruthFile, "\n");
	} 

	fclose(fp_groundtruthFile);

	string toolFile = folder + "/" + "boundingBoxToolFile.csv";
	FILE *fp_toolFile = fopen(toolFile.c_str(), "w");
	if(fp_toolFile == NULL)
		fprintf(stderr, "Cannot open bounding box output file to write\n\n");

	fprintf(fp_toolFile, "image_number,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence,label,x_normalized,y_normalized,w_normalized,h_normalized,confidence\n");


	for(unsigned int i = 0; i < yolo_images.size(); i++)
	{
		yolo_images[i].image_name += ".JPEG";
		fprintf(fp_toolFile, "%s,", yolo_images[i].image_name.c_str());
		for(unsigned int j = 0; j < yolo_images[i].boxes.size(); j++)
		{ 	
			fprintf(fp_toolFile, "%s,%lf,%lf,%lf,%lf,%lf,", yolo_images[i].boxes[j].label.c_str(),yolo_images[i].boxes[j].x_normalized,yolo_images[i].boxes[j].y_normalized,yolo_images[i].boxes[j].w_normalized,
				yolo_images[i].boxes[j].h_normalized,yolo_images[i].boxes[j].confidence);
		}
		fprintf(fp_toolFile, "\n");
	} 

	fclose(fp_toolFile);

    
}