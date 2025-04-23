#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>

#include "common.h"
#include "vx_ext_opencv.h"

// Name Output Display Windows
#define MIVisionX_LEGEND_D    "MIVisionX Image Detection"
#define MIVisionX_DISPLAY_D   "MIVisionX Image Detection - Display"

static const int colors[20][3] = {
			{160,82,45},        // aeroplane
			{128,0,0},          // bicycle
			{47,79,79},         // bird
			{155,140,145},      // boat
			{140,155,255},      // bottle
			{255,105,180},      // bus
			{255,0,0},          // car
			{75,0,130},         // cat
			{255,140,0},        // chair
			{250,128,114},      // cow
			{153,50,204},       // diningtable
			{130,230,150},      // dog
			{0,220,255},        // horse
			{0,191,255},        // motorbike
			{0,0,255},          // person
			{0,255,255},        // potted plant
			{107,142,35},       // sheep
			{0,128,0},          // sofa
			{124,252,0},        // train
			{199,21,133}        // tvmonitor
};

struct Region
{
	bool initialized;
	int totalLength;
	int totalObjects;
	std::vector<float> output;
	std::vector<ibox> boxes;
	std::vector<indexsort> s;
	int mConfidence;
	int mColorNum;
	int mWidth;
	int mHeight;

	Region();

	void Initialize(int c, int h, int w, int size);


	void GetDetections(cv::Mat &image, float* data, int c, int h, int w,
		int classes, int imgw, int imgh,
		float thresh, float nms,
		int blockwd,
		std::vector<DetectedObject> &objects, std::string labelText[]);
	
	void show(cv::Mat &image, int confidence, std::vector<DetectedObject> &results);
	
	void LegendImage(std::string labelText[]);


};
