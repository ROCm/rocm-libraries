
#include "detection.h"
#include "common.h"
#include "cvui.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>

const int N = 5;
const float biases[N*2] = {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};
//const float biases[N*2] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

Region::Region()
{
	initialized = false;
	totalLength = 0;
}


void Region::Initialize(int c, int h, int w, int size)
{
	totalLength = c * h * w;
	totalObjects = N * h * w;
	output.resize(totalLength);
	boxes.resize(totalObjects);
	s.resize(totalObjects);

	for(int i = 0; i < totalObjects; ++i)
	{
		s[i].index = i;
		s[i].channel = size;
		s[i].prob = &output[5];
	}

	mConfidence ;
	mColorNum = c;
	mWidth = w;
	mHeight = h;

	cv::namedWindow(MIVisionX_DISPLAY_D, cv::WINDOW_GUI_EXPANDED);

	initialized = true;
}

void Region::show(cv::Mat &mImage, int mConfidence, std::vector<DetectedObject> &mResults) {
	cv::Mat img_cp = mImage.clone();
	resize(img_cp, img_cp, cv::Size(mWidth, mHeight));
	int detectedNum = (int)mResults.size();
	for (int i = 0; i < detectedNum; i++) {
		float confidence = mResults[i].confidence;
		confidence = confidence * 100;

		if (confidence > mConfidence) {

			float left = mResults[i].left;
			float top = mResults[i].top;
			float right = (mResults[i].right - mResults[i].left) + left;
			float bottom = (mResults[i].bottom - mResults[i].top)  + top;
			int index = mResults[i].objType % mColorNum;
			cv::Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
			std::string txt = mResults[i].name;
			rectangle(mImage, cv::Point((int)left, (int)top), cv::Point((int)right, (int)bottom), clr, 2);
			cv::Size size = cv::getTextSize(txt, CV_FONT_HERSHEY_SIMPLEX, 0.8, 1, 0);
			int width = size.width;
			int height = size.height;
			rectangle(mImage, cv::Point((int)left, ((int)bottom - 5) - (height + 5)), cv::Point(((int)left + width), ((int)bottom - 5)), clr, -1);
			putText(mImage, txt, cv::Point(((int)left + 5), ((int)bottom - 10)), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, 8);
		}
	}
	resize(img_cp, img_cp, cv::Size(mImage.cols, mImage.rows));
	imshow(MIVisionX_DISPLAY_D, mImage);

	return;

}

void Region::LegendImage(std::string labelText[]) {

	cv::Size legendGeometry = cv::Size(325, (20 * 40) + 40);
	cv::Mat legend = cv::Mat::zeros(legendGeometry, CV_8UC3);
	cv::Rect roi = cv::Rect(0, 0, 325, (20 * 40) + 40);
	legend(roi).setTo(cv::Scalar(255, 255, 255));

	for (int l = 0; l < 20; l++) {
		cv::Scalar clr(colors[l][0], colors[l][1], colors[l][2]);
		std::string className = labelText[l];
		putText(legend, className, cv::Point(20, (l * 40) + 30), CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, 8);
		rectangle(legend, cv::Point(225, (l * 40)), cv::Point(300, (l * 40) + 40), clr, -1);
	}
	imshow(MIVisionX_LEGEND_D, legend);

	return;
}

void Region::GetDetections(cv::Mat &frame, float* data, int c, int h, int w,
						   int classes, int imgw, int imgh,
						   float thresh, float nms,
						   int blockwd,
						   std::vector<DetectedObject> &objects, std::string labelText[])
{
	objects.clear();

	int size = 4 + classes + 1;
	if(!initialized)
	{
		Initialize(c, h, w, size);
	}

	if(!initialized)
	{
		printf("Fail to initialize internal buffer!\n");
		return ;
	}

	int i,j,k;

	transpose(data, &output[0], size*N, w*h);
	int confidence = 0.2;
	// Initialize box, scale and probability
	for(i = 0; i < h*w*N; ++i)
	{
		int index = i * size;
		//Box
		int n = i % N;
		int row = (i/N) / w;
		int col = (i/N) % w;

		boxes[i].x = (col + logistic_activate(output[index + 0])) / blockwd; //w;
		boxes[i].y = (row + logistic_activate(output[index + 1])) / blockwd; //h;
		boxes[i].w = exp(output[index + 2]) * biases[2*n]   / blockwd; //w;
		boxes[i].h = exp(output[index + 3]) * biases[2*n+1] / blockwd; //h;

		//Scale
		output[index + 4] = logistic_activate(output[index + 4]);

		//Class Probability
		softmax(&output[index + 5], classes, 1, &output[index + 5]);
		for(j = 0; j < classes; ++j)
		{
			output[index+5+j] *= output[index+4];
			if(output[index+5+j] < thresh) output[index+5+j] = 0;
		}
	}

	//nms
	for(k = 0; k < classes; ++k)
	{
		for(i = 0; i < totalObjects; ++i)
		{
			s[i].iclass = k;
		}
		qsort(&s[0], totalObjects, sizeof(indexsort), indexsort_comparator);
		for(i = 0; i < totalObjects; ++i){
			if(output[s[i].index * size + k + 5] == 0) continue;
			ibox a = boxes[s[i].index];
			for(j = i+1; j < totalObjects; ++j){
				ibox b = boxes[s[j].index];
				if (box_iou(a, b) > nms){
					output[s[j].index * size + 5 + k] = 0;
				}
			}
		}
	}

	// generate objects
	for(i = 0, j = 5; i < totalObjects; ++i, j += size)
	{
		int iclass = max_index(&output[j], classes);

		float prob = output[j+iclass];

		if(prob > thresh)
		{
			ibox b = boxes[i];

			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			int left  = (b.x-b.w/2.)*imgw;
			int right = (b.x+b.w/2.)*imgw;
			int top   = (b.y-b.h/2.)*imgh;
			int bot   = (b.y+b.h/2.)*imgh;

			if(left < 0) left = 0;
			if(right > imgw-1) right = imgw-1;
			if(top < 0) top = 0;
			if(bot > imgh-1) bot = imgh-1;


			DetectedObject obj;
			obj.left = left;
			obj.top = top;
			obj.right = right;
			obj.bottom = bot;
			obj.x = b.x;
			obj.y = b.y;
			obj.w = b.w;
			obj.h = b.h;
			obj.confidence = prob;
			obj.objType = iclass;
			obj.name = labelText[iclass];
            //std::cout << "BoundingBox(ltrb): "<< i << "( " << left << " " << top << " "<< right << " "<< bot << ") " << "confidence: " << prob << " lablel: " << iclass << std::endl;			
			objects.push_back(obj);
		}
	}

	Region::LegendImage(labelText);
	Region::show(frame, confidence, objects);

	return ;
}