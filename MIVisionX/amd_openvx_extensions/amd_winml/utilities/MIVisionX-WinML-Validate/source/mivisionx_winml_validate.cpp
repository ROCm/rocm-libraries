/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "mivisionx_winml_validate.h"

// usage/help function
static void show_usage()
{
	printf("\n*************************************************************************************************************************************\n");
	printf("\n                                              MIVisionX ONNX Model Validation - %s\n", MIVISIONX_WINML_UTILITY_VERSION);
	printf("\n*************************************************************************************************************************************\n");
	printf("\n");
	printf("Usage:\n\n");
	printf("\tMIVisionX-WinML-Validate.exe [options]\t--m <ONNX.model full path>\n");
	printf("\t\t\t\t\t\t--i <model input tensor name>\n");
	printf("\t\t\t\t\t\t--o <model output tensor name>\n");
	printf("\t\t\t\t\t\t--s <output tensor size in (n,c,h,w)>\n");
	printf("\t\t\t\t\t\t--l <label.txt full path>\n");
	printf("\t\t\t\t\t\t--f <image frame full path>\n");
	printf("\t\t\t\t\t\t--d <Learning Model Device Kind <DirectXHighPerformance>> [optional]\n\n");
	printf("\n");
	printf("\nMIVisionX ONNX Model Validation Parameters\n\n");
	printf("\t--m/--model \t\t\t-- onnx model full path [required]\n");
	printf("\t--i/--inputName \t\t-- model input tensor name [required]\n");
	printf("\t--o/--outputName \t\t-- model output tensor name [required]\n");
	printf("\t--s/--outputSize \t\t-- model output tensor size <n,c,h,w> [required]\n");
	printf("\t--l/--label \t\t\t-- label.txt file full path [required]\n");
	printf("\t--f/--imageFrame \t\t-- imageFrame.png file full path [required]\n");
	printf("\t--d/--deviceKind \t\t-- Learning Model Device Kind <0-4> [optional]\n");
	for(int i = 0; i < 5; i++)
		printf("\t                \t\t %d - %s\n",i,deviceNameArray[i]);
	printf("\n");
	printf("\nMIVisionX ONNX Model Validation Options\n\n");
	printf("\t--h/--help\t-- Show full help\n");
	printf("\n");
}

// load ONNX model to WinML
void LoadModelFromPath(hstring modelLocation)
{
	printf("\n\nMIVisionX: Loading modelfile '%ws' on the '%s' device\n", modelLocation.c_str(), deviceNameArray[deviceIndex].c_str());
	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();
	model = LearningModel::LoadFromFilePath(modelLocation);
	t1 = clockCounter();
	printf("MIVisionX: Model file loading took -- %.3f msec\n", (float)(t1 - t0)*1000.0f / (float)freq);
}

// load image file for inference
VideoFrame LoadImageFile(hstring filePath)
{
	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();
	VideoFrame inputImage = nullptr;

	try
	{
		// open the file
		StorageFile file = StorageFile::GetFileFromPathAsync(filePath).get();
		// get a stream on it
		auto stream = file.OpenAsync(FileAccessMode::Read).get();
		// Create the decoder from the stream
		BitmapDecoder decoder = BitmapDecoder::CreateAsync(stream).get();
		// get the bitmap
		SoftwareBitmap softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
		// load a videoframe from it
		inputImage = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
	}
	catch (...)
	{
		printf("ERROR:failed to load the image file, make sure you are using fully qualified paths\r\n");
		exit(EXIT_FAILURE);
	}

	t1 = clockCounter();
	printf("MIVisionX: Image file loading took -- %.3f msec\n", (float)(t1 - t0)*1000.0f / (float)freq);
	// all done
	return inputImage;
}

// bind the ONNX model
void BindModel(hstring inputTensorName, hstring outputTensorName, int64_t *outputDim)
{
	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();

	// now create a session and binding
	session = LearningModelSession{ model, LearningModelDevice(deviceKindArray[deviceIndex]) };
	binding = LearningModelBinding{ session };
	// bind the intput image
	binding.Bind(inputTensorName, ImageFeatureValue::CreateFromVideoFrame(imageFrame));
	// bind the output
	vector<int64_t> shape({ outputDim[0],  outputDim[1],  outputDim[2],  outputDim[3] });
	binding.Bind(outputTensorName, TensorFloat::Create(shape));

	t1 = clockCounter();
	printf("MIVisionX: Model file binding took -- %.3f msec\n", (float)(t1 - t0)*1000.0f / (float)freq);
}

// load label.txt
void LoadLabels()
{
	// Parse labels from labels file.  We know the file's entries are already sorted in order.
	ifstream labelFile{ labelsFilePath, ifstream::in };
	if (labelFile.fail())
	{
		printf("ERROR:failed to load the %s file.  Make sure it exists in the same folder as the app\r\n", labelsFilePath.c_str());
		exit(EXIT_FAILURE);
	}

	std::string s;
	while (std::getline(labelFile, s, ','))
	{
		int labelValue = atoi(s.c_str());
		if (labelValue >= labels.size())
		{
			labels.resize(labelValue + 1);
		}
		std::getline(labelFile, s);
		labels[labelValue] = s;
	}
}

// print results on screen
void PrintResults(IVectorView<float> results)
{
	// load the labels
	LoadLabels();
	// Find the top 3 probabilities
	vector<float> topProbabilities(3);
	vector<int> topProbabilityLabelIndexes(3);
	// SqueezeNet returns a list of 1000 options, with probabilities for each, loop through all
	for (uint32_t i = 0; i < results.Size(); i++)
	{
		// is it one of the top 3?
		for (int j = 0; j < 3; j++)
		{
			if (results.GetAt(i) > topProbabilities[j])
			{
				topProbabilityLabelIndexes[j] = i;
				topProbabilities[j] = results.GetAt(i);
				break;
			}
		}
	}
	// Display the result
	for (int i = 0; i < 3; i++)
	{
		printf("Top-%d: %s with confidence of %f\n", i+1, labels[topProbabilityLabelIndexes[i]].c_str(), topProbabilities[i]);
	}
}

// run inference
void EvaluateModel(hstring modelOutputTensorName)
{
	// now run the model
	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();
	auto results = session.Evaluate(binding, L"RunId");
	t1 = clockCounter();
	printf("MIVisionX: Model first run took -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

	// get the output
	auto resultTensor = results.Outputs().Lookup(modelOutputTensorName).as<TensorFloat>();
	auto resultVector = resultTensor.GetAsVectorView();
	PrintResults(resultVector);
}

// run inference for timing
void EvaluateModelPlain()
{
	auto results = session.Evaluate(binding, L"RunId");
}

// main entry function
int main(int argc, char * argv[])
{
	int status = 0;
	char *modelFileLocation = NULL, *modelInputName = NULL, *modelOutputName = NULL;
	char *modelOutputSize = NULL, *labelFileLocation = NULL, *imageFileLocation = NULL;
	int parameter = 0;
	// device set to DirectXHighPerformance
	deviceIndex = 3;

	for (int arg = 1; arg < argc; arg++)
	{
		if (!strcasecmp(argv[arg], "--help") || !strcasecmp(argv[arg], "--H") || !strcasecmp(argv[arg], "--h"))
		{
			show_usage();
			exit(status);
		}
		else if (!strcasecmp(argv[arg], "--model") || !strcasecmp(argv[arg], "--M") || !strcasecmp(argv[arg], "--m"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing ONNX .model file location on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			modelFileLocation = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--inputName") || !strcasecmp(argv[arg], "--I") || !strcasecmp(argv[arg], "--i"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing model input tensor name on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			modelInputName = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--outputName") || !strcasecmp(argv[arg], "--O") || !strcasecmp(argv[arg], "--o"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing model output tensor name on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			modelOutputName = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--outputSize") || !strcasecmp(argv[arg], "--S") || !strcasecmp(argv[arg], "--s"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing model output tensor size on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			modelOutputSize = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--label") || !strcasecmp(argv[arg], "--L") || !strcasecmp(argv[arg], "--l"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing label.txt file on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			labelFileLocation = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--imageFrame") || !strcasecmp(argv[arg], "--F") || !strcasecmp(argv[arg], "--f"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing image.png file on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			imageFileLocation = (argv[arg]);
			parameter++;
		}
		else if (!strcasecmp(argv[arg], "--deviceKind") || !strcasecmp(argv[arg], "--D") || !strcasecmp(argv[arg], "--d"))
		{
			if ((arg + 1) == argc)
			{
				printf("\n\nERROR: missing device kind index on command-line (see help for details)\n\n\n");
				show_usage();
				status = -1;
				exit(status);
			}
			arg++;
			deviceIndex = atoi(argv[arg]);
			if (deviceIndex > 4) deviceIndex = 0;
		}
	}
	// check if all the parameters needed was passed
	if (parameter != 6)
	{
		printf("\nERROR: missing parameters in command-line. Please check help for details\n");
		show_usage();
		status = -1;
		exit(status);
	}

	// get model location
	std::string Model(modelFileLocation);
	wstring wModel(Model.begin(), Model.end());
	hstring modelLocation = wModel.c_str();

	// get model input tensor name
	std::string inputName(modelInputName);
	wstring wInputName(inputName.begin(), inputName.end());
	hstring modelInputTensorName = wInputName.c_str();

	// get model output tensor name
	std::string outputName(modelOutputName);
	wstring wOutputName(outputName.begin(), outputName.end());
	hstring modelOutputTensorName = wOutputName.c_str();

	// get model output tensor size
	std::string outputSize(modelOutputSize);
	std::vector<int> sizeVector;
	std::stringstream sizeStream(outputSize);
	int i;
	while (sizeStream >> i)
	{
		sizeVector.push_back(i);

		if (sizeStream.peek() == ',')
			sizeStream.ignore();
	}

	if (sizeVector.size() != 2 && sizeVector.size() != 4)
	{
		printf("\nERROR:Output Tensor Size: %d. Please check help for details\n", sizeVector.size());
		show_usage();
		status = -1;
		exit(status);
	}

	int64_t outputDims[4] = { 0 };
	for (i = 0; i < sizeVector.size(); i++)
		outputDims[i] = sizeVector.at(i);

	// get label.txt location
	std::string labelLocation(labelFileLocation);
	labelsFilePath = labelLocation;

	// get image location
	std::string Image(imageFileLocation);
	wstring wImage(Image.begin(), Image.end());
	hstring imageLocation = wImage.c_str();

	// load model
	LoadModelFromPath(modelLocation);
	// load image
	imageFrame = LoadImageFile(imageLocation);
	// bind model
	BindModel(modelInputTensorName, modelOutputTensorName, outputDims);
	// run inference
	EvaluateModel(modelOutputTensorName);

	// get avg inference time in msec
	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();
	for (i = 0; i < 100; i++)
		EvaluateModelPlain();
	t1 = clockCounter();
	printf("MIVisionX: Avg model run run time for 100 iterations -- %.3f msec\n\n", (float)((t1 - t0)/100)*1000.0f / (float)freq);

	return status;
}
