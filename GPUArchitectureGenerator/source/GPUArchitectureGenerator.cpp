#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string.h>
#include <string>

#include "GPUArchitectureGenerator/GPUArchitectureGenerator.hpp"

const std::string HELP_MESSAGE
    = "Usage: GPUArchitectureGenerator OUTPUT_FILE [ASSEMBLER_LOCATION]\n"
      "OUTPUT_FLE:            Location to store generated file.\n"
      "ASSEMBLER_LOCATION:    PAth to assembler executable.\n"
      "\n"
      "Options:\n"
      "  -Y               output YAML (msgpack by default)\n"
      "  -h, --help       display this help and exit";

const std::string YAML_ARG = "-Y";

int main(int argc, char** argv)
{
    std::string assembler = GPUArchitectureGenerator::DEFAULT_ASSEMBLER;
    bool        useYAML   = false;

    if(argc < 2 || argc > 3 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        fprintf(stderr, "%s\n", HELP_MESSAGE.c_str());
        if(argc < 2 || argc > 3)
        {
            return 1;
        }
        return 0;
    }
    if(argc > 2 && strcmp(argv[2], YAML_ARG.c_str()) != 0)
    {
        assembler = argv[2];
    }
    else if(argc > 3 && strcmp(argv[3], YAML_ARG.c_str()) != 0)
    {
        assembler = argv[3];
    }

    if((argc > 2 && strcmp(argv[2], YAML_ARG.c_str()) == 0)
       || (argc > 3 && strcmp(argv[3], YAML_ARG.c_str()) == 0))
    {
        useYAML = true;
    }

    if(!GPUArchitectureGenerator::CheckAssembler(assembler))
    {
        fprintf(stderr, "Error using assembler:\n%s\n", assembler.c_str());
        return 1;
    }

    GPUArchitectureGenerator::FillArchitectures(assembler);

    try
    {
        GPUArchitectureGenerator::GenerateFile(argv[1], useYAML);
    }
    catch(std::exception& e)
    {
        fprintf(stderr,
                "The following exception was encountered while generating the source file:\n%s\n",
                e.what());
        return 1;
    }
    catch(...)
    {
        fprintf(stderr, "An unkown error was encountered while generating the source file.");
        return 1;
    }
    return 0;
}
