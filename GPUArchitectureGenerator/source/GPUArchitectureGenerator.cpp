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
      "OUTPUT_FILE:           Location to store generated file.\n"
      "ASSEMBLER_LOCATION:    Path to assembler executable.\n"
      "\n"
      "Options:\n"
      "  -Y               output YAML (msgpack by default)\n"
      "  --xml_dir        source directory for ISA xml files\n"
      "  -h, --help       display this help and exit";

const std::string YAML_ARG    = "-Y";
const std::string XML_DIR_ARG = "--xml_dir";

struct ProgramArgs
{
    std::string outputFile;
    std::string assembler = GPUArchitectureGenerator::DEFAULT_ASSEMBLER;
    bool        useYAML   = false;
    std::string xmlDir    = "";

    static ProgramArgs       ParseArgs(int argc, const char* argv[]);
    [[noreturn]] static void Help(bool error = false);
};

ProgramArgs ProgramArgs::ParseArgs(int argc, const char* argv[])
{
    if(argc <= 1)
        Help(true);

    std::vector<std::string> args(argv + 1, argv + argc);
    ProgramArgs              rv;

    for(auto iter = args.begin(); iter != args.end(); iter++)
    {
        if(*iter == "-h" || *iter == "--help")
            Help(false);
        if(*iter == YAML_ARG)
        {
            rv.useYAML = true;
            iter       = args.erase(iter);
            iter--;
        }
        if(*iter == XML_DIR_ARG)
        {
            iter      = args.erase(iter);
            rv.xmlDir = *iter;
            iter      = args.erase(iter);
            iter--;
        }
    }

    if(args.empty() || args.size() > 2)
        Help(true);

    rv.outputFile = args[0];
    if(args.size() > 1)
        rv.assembler = args[1];

    return rv;
}

[[noreturn]] void ProgramArgs::Help(bool error)
{
    fprintf(stderr, "%s\n", HELP_MESSAGE.c_str());

    if(error)
        exit(1);
    else
        exit(0);
}

int main(int argc, const char* argv[])
{
    auto args = ProgramArgs::ParseArgs(argc, argv);

    if(!GPUArchitectureGenerator::CheckAssembler(args.assembler))
    {
        fprintf(stderr, "Error using assembler:\n%s\n", args.assembler.c_str());
        return 1;
    }

    GPUArchitectureGenerator::FillArchitectures(args.assembler, args.xmlDir);

    try
    {
        GPUArchitectureGenerator::GenerateFile(args.outputFile, args.useYAML);
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
