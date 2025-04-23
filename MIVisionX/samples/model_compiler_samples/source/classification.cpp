#include "classification.h"
// Header file details in include folder
#define CVUI_IMPLEMENTATION
#include "cvui.h"

Classifier::Classifier()
{
    initialized = false;
}


void Classifier::initialize()
{
    threshold_slider_max = 100;
    threshold_slider = 50;
    thresholdValue = 0.5;
    cv::namedWindow(MIVisionX_LEGEND);
    cvui::init(MIVisionX_LEGEND);
    cv::namedWindow(MIVisionX_DISPLAY, cv::WINDOW_GUI_EXPANDED);
    initialized = true;
}

void Classifier::threshold_on_trackbar( int, void* ){
    thresholdValue = (double) threshold_slider/threshold_slider_max ;
    return;
}

// create legend image for the app
void Classifier::createLegendImage(std::string modelName, float modelTime_g)
{   
    bool runModel = true;
    // create display legend image
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 0.75;
    int thickness = 1.3;
    cv::Size legendGeometry = cv::Size(625, (7 * 40) + 40);
    cv::Mat legend = cv::Mat::zeros(legendGeometry,CV_8UC3);
    cv::Rect roi = cv::Rect(0,0,625,(7 * 40) + 40);
    legend(roi).setTo(cv::Scalar(128,128,128));
    int l = 0, model = 0;
    int red, green, blue;
    std::string className;
    std::string bufferName;
    char buffer [50];

    // add headers
    bufferName = MIVisionX_LEGEND;
    putText(legend, bufferName, cv::Point(20, (l * 40) + 30), fontFace, 1.2, cv::Scalar(66,13,9), thickness,5);
    l++;
    l++;
    bufferName = "Model";
    putText(legend, bufferName, cv::Point(100, (l * 40) + 30), fontFace, 1, cv::Scalar::all(0), thickness,5);
    bufferName = "ms/frame";
    putText(legend, bufferName, cv::Point(300, (l * 40) + 30), fontFace, 1, cv::Scalar::all(0), thickness,5);
    bufferName = "Color";
    putText(legend, bufferName, cv::Point(525, (l * 40) + 30), fontFace, 1, cv::Scalar::all(0), thickness,5);
    l++;
    
    // add legend item
    thickness = 1;    
    red = 255; green = 0; blue = 0;
    className = modelName;
    sprintf (buffer, " %.2f ", modelTime_g );
    cvui::checkbox(legend, 30, (l * 40) + 15,"", &runModel);
    putText(legend, className, cv::Point(80, (l * 40) + 30), fontFace, fontScale, cv::Scalar::all(0), thickness,3);
    putText(legend, buffer, cv::Point(320, (l * 40) + 30), fontFace, fontScale, cv::Scalar::all(0), thickness,3);
    rectangle(legend, cv::Point(550, (l * 40)) , cv::Point(575, (l * 40) + 40), cv::Scalar(blue,green,red),-1);
    l++;
    l++;

    // Model Confidence Threshold
    bufferName = "Confidence";
    putText(legend, bufferName, cv::Point(250, (l * 40) + 30), fontFace, 1, cv::Scalar::all(0), thickness,5);
    l++;
    cvui::trackbar(legend, 100, (l * 40)+10, 450, &threshold_slider, 0, 100);
    cvui::update();

    cv::imshow(MIVisionX_LEGEND, legend);
    thresholdValue = (double) threshold_slider/threshold_slider_max ;

}

void Classifier::visualize(cv::Mat &frame, int channels, float *outputBuffer, std::string NN_ModelName, std::string labelText[], float modelTime_g)
{
    if(!initialized)
    {
        initialize();
    }

    if(!initialized)
    {
        printf("Fail to initialize internal buffer!\n");
        return ;
    }

    //threshold_on_trackbar(thresholdValue, threshold_slider);

    int outputImgWidth = 1080, outputImgHeight = 720;
    cv::Size output_geometry = cv::Size(outputImgWidth, outputImgHeight);
    cv::Mat inputDisplay, outputDisplay;  
    float threshold = (float)thresholdValue;
    const int N = channels;
    int resnetID;

    cv::Mat inputFrame_299x299, inputFrame_data_resized;
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1.5;

    //process probabilty
    resnetID = std::distance(outputBuffer, std::max_element(outputBuffer, outputBuffer + N));

    // Write Output on Image
    cv::resize(frame, outputDisplay, cv::Size(outputImgWidth,outputImgHeight));
    int l = 1;
    std::string modelName = NN_ModelName + " -- ";
    std::string modelText = "Unclassified";

    if(outputBuffer[resnetID] >= threshold){ modelText = labelText[resnetID]; }

    modelName = modelName + modelText;
    int red, green, blue;
    red = 255; green = 0; blue = 0;
    putText(outputDisplay, modelName, cv::Point(20, (l * 40) + 30), fontFace, fontScale, cv::Scalar(blue,green,red), thickness,8);

    // display img time
    cv::imshow(MIVisionX_DISPLAY, outputDisplay);
    createLegendImage(NN_ModelName, modelTime_g);
}