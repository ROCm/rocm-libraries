/* Project: YOLO Result Verification Tool
 * Date: 08/30/2018
 * Author: Lakshmi
*/

//header files
#include "compare_util.h"
#include "compare_util.cpp"
#include <iostream>
#include <string>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <sys/stat.h>	/*structure of the data returned by the functions stat()*/
#include <dirent.h>		/*format of directory entries*/
using namespace std;

void print_usage()
{
    cout << "Invalid command line arguments.\n" 
         << "-i [Output of Bounding Box tool CSV File - required](File Format:image_name, x, y, w, h, label)(x,y: center coordinates of bounding box; w,h: width and heigth of image) \n"
         << "-g [Groutnd Truth Image Directory - required] \n"
         << "-o [Output Reuslt CSV File        - required] \n"
         << "-m [Mode indicating type of input - required] (m: 'normalized' or 'pixel' - format of x,y,w,h) \n"
         << "-iou [Intersection over Union Value - requiured] (float value between 0 and 1)\n"
         << "-w [width of image] (for pixel format) \n"
         << "-h [heigth of image] (for pixel format) \n";
         exit(EXIT_FAILURE);
}

//main function
int main(int argc, char **argv)
{
	
	//checks if all files were provided by the user
	string inputFile;
    string groundTruthDir;
    string outputFile;
    string mode;
    int width_image = -1;
    int height_image = -1;
    float IOU;

    if(argc < 11){
        print_usage();
    }
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-i") == 0){
            i++;
            inputFile = argv[i];
        }
        else if(strcmp(argv[i], "-g") == 0){
            i++;
            groundTruthDir = argv[i];
        }
        else if(strcmp(argv[i], "-o") == 0){
            i++;
            outputFile = argv[i];
        }
        else if(strcmp(argv[i], "-m") == 0){
            i++;
            mode = argv[i];
        }
        else if(strcmp(argv[i], "-iou") == 0){
            i++;
            IOU = atof(argv[i]);
        }
        else if(strcmp(argv[i], "-w") == 0){
            i++;
            width_image = atoi(argv[i]);
        }
        else if(strcmp(argv[i], "-h") == 0){
            i++;
            height_image = atoi(argv[i]);
        }
    }

    struct dirent *pDirent; /*holds information of a directory independent of the file system*/
    DIR *pDir;              /*represents a directory stream*/
    struct stat filestat;   /*holds information about a file based on it's path*/
	vector<string> gtFileNames;
    
    // open the input file for parsing the ground truth 
	if ((pDir = opendir(groundTruthDir.c_str())) == NULL)
	{
        fprintf(stderr, "Can't open directory %s\n", groundTruthDir.c_str());
        exit(1);
    }

    //reads all files in the directory one by one
    while((pDirent = readdir(pDir)) != NULL)
    {	
        string groundTruthFileName = groundTruthDir + string(pDirent->d_name);

        // If the file is a directory (or is in some way invalid) - skip it 
        if (stat(groundTruthFileName.c_str(), &filestat))
            continue;
        if (S_ISDIR(filestat.st_mode))         
            continue;
        
        gtFileNames.push_back(groundTruthFileName);
   	
    }

    closedir(pDir);

    sort(gtFileNames.begin(), gtFileNames.end());

    for(int i = 0; i < gtFileNames.size(); i++)
    {
        //call GT parser function
        parseGTFile(gtFileNames[i]);       
    }
	//call YOLO parser functions
    if(mode == "normalized")
    {
        parseFileNormalized(inputFile);    
    }
    else if(mode == "pixel")
    {
        if(width_image == -1 || height_image == -1)
            print_usage();
        parseFilePixel(inputFile, width_image, height_image);
    }
    
  	//call compare function
  	compareValues(IOU);

  	writeToFile(outputFile);
	
    //display();

	return 0;
}
