/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <fstream>
#include <iomanip>
#include <iostream>
using namespace std;

void bin_2_hex(string infilename, ofstream* outfile, string dataname)
{
    ifstream infile;
    infile.open(infilename, ios_base::binary);
    if(!infile)
    {
        cout << "Failed to open:" << infilename.c_str() << endl;
        exit(-1);
    }

    char          readByte;
    unsigned char readByte2;
    int           cnt = 0;

    *outfile << "{ \"" << dataname.substr(0, dataname.length() - 3) << "\", " << endl;
    *outfile << "{ " << endl;

    while(!infile.eof())
    {
        if(cnt % 16 == 0)
        {
            *outfile << " " << endl;
            *outfile << "   ";
        }
        infile.get(readByte);
        readByte2 = (unsigned char)readByte;
        cnt++;
        *outfile << "0x" << setfill('0') << setw(2);
        *outfile << hex << (unsigned int)readByte2;
        *outfile << ",";
    }

    *outfile << "\n}}, \n";
    infile.close();
}

std::string get_kernel_name(char* filename)
{
    std::string fname(filename);
    size_t      index = fname.find_last_of("\//");
    return fname.substr(index + 1);
}

int main(int argc, char** argv)
{
    int filenum = argc;

    string outfilepath = argv[1]; //"spmm_kernels"; // output header file
    string outfilename = argv[2]; //"spmm_kernels"; // output header file
    string cppfilename = outfilepath + "/" + outfilename + ".cpp";
    string hppfilename = outfilepath + "/" + outfilename + ".hpp";
    remove(cppfilename.c_str());
    remove(hppfilename.c_str());

    ofstream outfile;
    outfile.open(cppfilename, ios_base::binary | ios_base::app);
    if(!outfile)
    {
        cout << "Failed to open:" << cppfilename.c_str() << endl;
        exit(-1);
    }

    //outfile << "#ifndef SPMM_KERNELS_H\n#define SPMM_KERNELS_H" << endl;
    //outfile << "#include \"" << outfilename << ".hpp\"" << endl;
    outfile << "#include <map>" << endl;
    outfile << "#include <string>" << endl;
    outfile << "#include <vector>" << endl;
    outfile << "std::map<std::string, std::vector<unsigned char>> kernel_map = {" << endl;
    for(int i = 3; i < filenum; i++)
    {
        bin_2_hex(argv[i], &outfile, get_kernel_name(argv[i]));
    }
    outfile << "};" << endl;

    outfile << "extern \"C\" int get_map_size() { return kernel_map.size(); }" << endl;
    outfile << "extern \"C\" unsigned char* get_kernel_byte(const char* name) { return "
               "kernel_map[name].data(); }"
            << endl;
    //outfile << "#endif" << endl;
    outfile.close();

    outfile.open(hppfilename, ios_base::binary | ios_base::app);
    if(!outfile)
    {
        cout << "Failed to open:" << hppfilename.c_str() << endl;
        exit(-1);
    }

#if 0
    outfile << "#pragma once" << endl;
    outfile << "#include <map>" << endl;
    outfile << "#include <string>" << endl;
    outfile << "#include <vector>" << endl;
    outfile << "__attribute__ ((visibility(\"default\"))) int get_map_size();";
    outfile.close();
#endif
    return 0;
}
