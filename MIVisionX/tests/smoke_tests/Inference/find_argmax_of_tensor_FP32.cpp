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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

struct tensor_element {
	int index;
	float value;
};

// A utility function to print array of size n 
/*void printArray(tensor_element tensor[], int n) 
{ 
    for (int i=0; i<n; ++i) 
        cout << tensor[i].index << " = " << tensor[i].value << endl; 
}*/

// To heapify a subtree rooted with node i which is 
// an index in tensor[]. n is size of heap 
void heapify(tensor_element tensor[], int n, int i) 
{ 
    int largest = i; // Initialize largest as root 
    int l = 2*i + 1; // left = 2*i + 1 
    int r = 2*i + 2; // right = 2*i + 2 
  
    // If left child is larger than root 
    if (l < n && tensor[l].value > tensor[largest].value) 
        largest = l; 
  
    // If right child is larger than largest so far 
    if (r < n && tensor[r].value > tensor[largest].value) 
        largest = r; 
  
    // If largest is not root 
    if (largest != i) 
    { 
        swap(tensor[i], tensor[largest]); 
  
        // Recursively heapify the affected sub-tree 
        heapify(tensor, n, largest); 
    } 
} 
  
// main function to do heap sort 
void heapSort(tensor_element tensor[], int n) 
{ 
    // Build heap (rearrange array) 
    for (int i = n / 2 - 1; i >= 0; i--) 
        heapify(tensor, n, i); 
  
    // One by one extract an element from heap 
    for (int i=n-1; i>=0; i--) 
    { 
        // Move current root to end 
        swap(tensor[0], tensor[i]); 
  
        // call max heapify on the reduced heap 
        heapify(tensor, i, 0); 
    } 
} 

void printTensor(tensor_element tensor[]) 
{ 
    //for (int i=0; i<1000; i++) 
    //    printf("%d,%f\n", tensor[i].index, tensor[i].value);

    printf("%d,%f\n", tensor[999].index, tensor[999].value);
} 

int main(int argc, char * argv[])
{
	float current_value;
	tensor_element current_element;
	tensor_element inputTensor[1000];
    
    if(argc < 1){
        printf("ERROR: arguments are tensor.fp\n");
        return -1;
    }

	std::ifstream inputTensorFile;
    
    inputTensorFile.open(argv[1], std::fstream::binary | std::fstream::in);

    //inputTensor read and initialize
    int current_index = 0;
	while(inputTensorFile.read(reinterpret_cast<char*>(&current_value), sizeof(current_value)))
    {
    	inputTensor[current_index].index = current_index;
    	inputTensor[current_index].value = current_value;
        current_index++;
    }   

    heapSort(inputTensor, 1000);
    printTensor(inputTensor);

    return 0;
}