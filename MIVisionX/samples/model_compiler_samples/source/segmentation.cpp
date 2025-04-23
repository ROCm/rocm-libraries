#include "segmentation.h"
#include "cvui.h"

// source: adapted from cityscapes-dataset.org

unsigned char overlayColors[20][3] = {
    {200,200,200},      // unclassified
    {128, 64,128},      // road
    {244, 35,232},      // sidewalk
    { 250, 150, 70},    // building
    {102,102,156},      // wall
    {190,153,153},      // fence
    { 0,  0,   0},      // pole
    {250,170, 30},      // traffic light
    {220,220,  0},      // traffic sign
    {0, 255, 0},        // vegetation
    {152,251,152},      // terrain
    { 135,206,250},     // sky
    {220, 20, 60},      // person
    {255,  0,  0},      // rider
    {  0,  0,255},      // car
    {  0,  0, 70},      // truck
    {  0, 60,100},      // bus
    {  0, 80,100},      // train
    {  0,  0,230},      // motorcycle
    {119, 11, 32}       // bicycle
};

Segment::Segment()
{
    initialized = false;
}

void Segment::threshold_on_trackbar( int, void* object){
    Segment *mSegment = (Segment *) object;
    mSegment->thresholdValue = (double) mSegment->threshold_slider/mSegment->threshold_slider_max ;
    return;
}

void Segment::alpha_on_trackbar( int, void* object){
    Segment *mSegment = (Segment *) object;
    mSegment->alphaValue = (double) mSegment->alpha_slider/mSegment->alpha_slider_max ;
    return;
}


void Segment::initialize()
{
    alpha_slider_max = 100;
	alpha_slider = 70;
	alphaValue = 0.7;
    threshold_slider_max = 100;
    threshold_slider = 50;
    thresholdValue = 0.5;

    cv::namedWindow(MIVisionX_LEGEND_S);
    cvui::init(MIVisionX_LEGEND_S);
    cv::namedWindow(MIVisionX_DISPLAY_S_I,cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow(MIVisionX_DISPLAY_S_M,cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow(MIVisionX_DISPLAY_S_O,cv::WINDOW_GUI_EXPANDED);
    initialized = true;
}


void Segment::findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg)
{
    for(int c = 0; c < numClasses; c++)
    {
        for(int i = start; i < end; i++)
        {
            if((output_layer[i] >= threshold) && (output_layer[i] > prob[i]))
            {
                prob[i] = output_layer[i];
                classImg[i] = c + 1;
            }
        }
        output_layer += (width * height);
    }

    return;
}

void Segment::createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage)
{
    cv::Vec3b pix;
    int classId = 0;
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < imageWidth; j++)
        {
            classId = classImg[(i * imageWidth) + j];

            pix.val[0] = (overlayColors[classId][2]);
            pix.val[1] = (overlayColors[classId][1]);
            pix.val[2] = (overlayColors[classId][0]);

            maskImage.at<cv::Vec3b>(i, j) = pix;
        }
    }
    return;
}

void Segment::createLegendImage(std::string labelText[])
{
     // create legend image
    int fontFace = CV_FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1.2;
    cv::Size legendGeometry = cv::Size(325, (30 * 25) + 25);
    cv::Mat legend = cv::Mat::zeros(legendGeometry,CV_8UC3);
    cv::Rect roi = cv::Rect(0,0,325,(30 * 25) + 25);
    legend(roi).setTo(cv::Scalar(255,255,255));
    int l;
    for (l = 0; l < 20; l ++){
        int red, green, blue;
        red = (overlayColors[l][2]) ;
        green = (overlayColors[l][1]) ;
        blue = (overlayColors[l][0]) ;
        std::string className = labelText[l];
        putText(legend, className, cv::Point(5, (l * 25) + 17), fontFace, fontScale, cv::Scalar::all(0), thickness,8);
        rectangle(legend, cv::Point(125, (l * 25)) , cv::Point(300, (l * 25) + 25), cv::Scalar(red,green,blue),-1);
    }

    l = 21;
    std::string bufferName = "Confidence";
    putText(legend, bufferName, cv::Point(10, (l * 25) + 25), fontFace, fontScale, cv::Scalar::all(0), thickness,8);
    l++;
    cvui::trackbar(legend, 10, (l * 25)+25, 200, &threshold_slider, 0, 100);
    cvui::update();
    l+=2;
    bufferName = "Alpha Blend";
    putText(legend, bufferName, cv::Point(10, (l * 25) + 25), fontFace, fontScale, cv::Scalar::all(0), thickness,8);
    l++;
    cvui::trackbar(legend, 10, (l * 25)+25, 200, &alpha_slider, 0, 100);
    cvui::update();
    cv::imshow(MIVisionX_LEGEND_S, legend);

    
    thresholdValue = (double) threshold_slider/threshold_slider_max ;
    alphaValue = (double) alpha_slider/alpha_slider_max ;

    return;
}

void Segment::getMaskImage(cv::Mat& inputImage, int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, cv::Size input_geometry, cv::Mat& maskImage, std::string labelText[])
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

    createLegendImage(labelText);
   
    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];
    int numthreads = std::thread::hardware_concurrency();
    size_t start = 0, end = 0, chunk = 0;
    int outputImgWidth = 1080, outputImgHeight = 720;

    double alpha = alphaValue, beta;
    beta = ( 1.0 - alpha );
    
    cv::Mat inputDisplay, outputDisplay,maskDisplay;  

    // Initialize buffers
    memset(prob, 0, (width * height * sizeof(float)));
    memset(classImg, 0, (width * height));

    // Class ID generation
    // parallel processing
    start = 0;
    end = height*width;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread t[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;

        t[i] = std::thread(&Segment::findClassProb, this, s, e, width, height, numClasses, output_layer, thresholdValue, prob, classImg) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ t[i].join() ; }
    
    // Mask generation
    // parallel create mask
    start = 0;
    end = input_geometry.height;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread M[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        M[i] = std::thread(&Segment::createMask, this, s, e, input_geometry.width, classImg, std::ref(maskImage)) ;
    }

    for(int i = 0 ; i < numthreads ; i++){ M[i].join() ; }
    
    cv::resize(inputImage, inputDisplay, cv::Size(outputImgWidth,outputImgHeight));
    cv::resize(maskImage, maskDisplay, cv::Size(outputImgWidth,outputImgHeight));
    cv::addWeighted( inputDisplay, alpha, maskDisplay, beta, 0.0, outputDisplay);
    cv::imshow(MIVisionX_DISPLAY_S_I, inputDisplay);
    cv::imshow(MIVisionX_DISPLAY_S_M, maskDisplay);
    cv::imshow(MIVisionX_DISPLAY_S_O, outputDisplay );    
    return;
}
