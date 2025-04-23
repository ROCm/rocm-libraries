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

#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <vector>

using namespace std;

vector<string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

int main(int argc, char* argv[])
{
	string tensorFileName;
	string splitFileName;
	string tempSplitFileName;
	string outputTensorName;
	int batchSize;
	FILE *tensorFP;
	char * buffer;
	FILE *splitFP;
	ostringstream oss;
	vector<string> tokenized_name;
	

	if (argc < 3){
		printf("Usage: split-tensor [tensor name] [batch size]\n");
		return -1;
	}
	else{
		buffer = new char[4000];
		tensorFileName = argv[1];
		
		batchSize = atoi(argv[2]);

		//printf("tensorFileName = %s\nbatch size = %d\n", tensorFileName.c_str(), batchSize);
		tokenized_name = split(tensorFileName, '.');
		
		tensorFP = fopen(tensorFileName.c_str(), "rb");

		for(int i = 0; i < batchSize; i++){

				
			oss << tokenized_name[0] << "-" << i << ".fp";
			outputTensorName = oss.str();
			//printf("outputTensorName = %s\n", outputTensorName.c_str());

			
			splitFP = fopen(outputTensorName.c_str(), "wb");
		
			fread(buffer,1,4000,tensorFP);
		
			fwrite(buffer,sizeof(char),4000,splitFP);

			
			fclose(splitFP);
			oss.str("");
			oss.clear();
		}
		fclose(tensorFP);
		delete[] buffer;
	}
	return 0;
}