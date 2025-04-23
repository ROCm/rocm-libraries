#include "visualize.h"

using namespace cv;
using namespace std;

Visualize::Visualize(float confidence) : mConfidence(confidence){
};

Visualize::~Visualize() {};

void Visualize::show(const Mat& img, std::vector<BBox>& Results, int batchSize) 
{
	// create Mat from data
	int detectedNum = (int)Results.size();
	int imgheight =  img.rows;
	int imgWidth = img.cols;
	float xoffs = 0.f, yoffs = 0.f;
	if (batchSize == 4) {
		imgheight >>= 1;
		imgWidth >>= 1;
		xoffs = (float)imgWidth;
		yoffs = (float)imgheight;
		for (int i = 0; i < detectedNum; i++) {
		    BBox *pb = &Results[i];
			if (pb->confidence > mConfidence) {
				float w2 = pb->w/2.f;  float h2 = pb->h/2.f;
				float left = (pb->x - w2)* imgWidth;
				float right = (pb->x + w2)* imgWidth;
				float top = (pb->y - h2)* imgheight;
				float bottom = (pb->y + h2)* imgheight;
				if(left < 0.0) left = 0.0;
				if(right > img.cols-1) right = (float)(imgWidth-1);
				if(top < 0.0) top = 0;
				if(bottom > imgheight-1) bottom = (float)(imgheight-1);
				if (pb->imgnum == 1) {
					left += xoffs; right += xoffs;
				} else if (pb->imgnum == 2) {
					top += yoffs; bottom += yoffs;
				} else if (pb->imgnum == 3) {
					left += xoffs; right += xoffs;
					top += yoffs; bottom += yoffs;
				}
				int index = pb->label; //Results[i].objType % mColorNum;
				Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
				string txt = yoloClasses[index];
				rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
				Size size = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
				int width = size.width;
				int height = size.height;
				rectangle(img, Point((int)left, ((int)top - (height + 4))), Point(((int)left + width), (int)top), clr, -1);
				putText(img, txt, Point((int)left, (int)top), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255, 255, 255), 1, 8);
	        }
		}
	} else if (batchSize > 4) {
		imgheight >>= (batchSize == 8) ? 1: 2;
		imgWidth >>= 2;
		xoffs = (float)imgWidth;
		yoffs = (float)imgheight;
		for (int i = 0; i < detectedNum; i++) {
		    BBox *pb = &Results[i];
			if (pb->confidence > mConfidence) {
				float w2 = pb->w/2.f;  float h2 = pb->h/2.f;
				float left = (pb->x - w2)* imgWidth;
				float right = (pb->x + w2)* imgWidth;
				float top = (pb->y - h2)* imgheight;
				float bottom = (pb->y + h2)* imgheight;
				if(left < 0.0) left = 0.0;
				if(right > img.cols-1) right = (float)(imgWidth-1);
				if(top < 0.0) top = 0;
				if(bottom > imgheight-1) bottom = (float)(imgheight-1);
				xoffs = (float)imgWidth*(pb->imgnum&0x3); 
				yoffs = (float)imgheight*((pb->imgnum&0xc)>>2);
				left += xoffs; right += xoffs;
				top += yoffs; bottom += yoffs;
				int index = pb->label; //Results[i].objType % mColorNum;
				Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
				string txt = yoloClasses[index];
				rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
				Size size = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
				int width = size.width;
				int height = size.height;
				rectangle(img, Point((int)left, ((int)top - (height + 4))), Point(((int)left + width), (int)top), clr, -1);
				putText(img, txt, Point((int)left, (int)top), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255, 255, 255), 1, 8);
	    	}
		}
	} else {
		for (int i = 0; i < detectedNum; i++) {		
		    BBox *pb = &Results[i];
			if (pb->confidence > mConfidence) {
				float w2 = pb->w/2.f;  float h2 = pb->h/2.f;
				float left = (pb->x - w2)* imgWidth;
				float right = (pb->x + w2)* imgWidth;
				float top = (pb->y - h2)* imgheight;
				float bottom = (pb->y + h2)* imgheight;
				if(left < 0.0) left = 0.0;
				if(right > img.cols-1) right = (float)(imgWidth-1);
				if(top < 0.0) top = 0;
				if(bottom > imgheight-1) bottom = (float)(imgheight-1);
				int index = pb->label; //Results[i].objType % mColorNum;
				Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
				string txt = yoloClasses[index];
				rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
				Size size = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
				//Size size = getTextSize(txt, FONT_HERSHEY_SIMPLEX, 0.6, 1, 0);
				int width = size.width;
				int height = size.height;
				//rectangle(img, Point((int)left, ((int)bottom - 5) - (height + 5)), Point(((int)left + width), ((int)bottom - 5)), clr, -1);
				//putText(img, txt, Point(((int)left + 5), ((int)bottom - 10)), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, 8);
				rectangle(img, Point((int)left, ((int)top - (height + 4))), Point(((int)left + width), (int)top), clr, -1);
				putText(img, txt, Point((int)left, (int)top), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255, 255, 255), 1, 8);
	        }
		}
	}
	//resize(img_cp, img_cp, Size(img.cols, img.rows));
	imshow("Detected Image", img);
}

void Visualize::LegendImage() {
	string window_name = "AMD Object Detection - Legend";

	Size legendGeometry = Size(325, (20 * 40) + 40);
	Mat legend = Mat::zeros(legendGeometry, CV_8UC3);
	Rect roi = Rect(0, 0, 325, (20 * 40) + 40);
	legend(roi).setTo(Scalar(255, 255, 255));

	for (int l = 0; l < 20; l++) {
		Scalar clr(colors[l][0], colors[l][1], colors[l][2]);
		string className = yoloClasses[l];
		putText(legend, className, Point(20, (l * 40) + 30), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1, 8);
		rectangle(legend, Point(225, (l * 40)), Point(300, (l * 40) + 40), clr, -1);
	}
	imshow(window_name, legend);

	return;
}

