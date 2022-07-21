#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <GPUArchitectureGenerator/GPUArchitectureGenerator.hpp>

TEST(GPUArchitectureGeneratorTest, BasicYAML)
{
    GPUArchitectureGenerator::FillArchitectures();

    GPUArchitectureGenerator::GenerateFile("output.yaml", true);

    std::ifstream     generated_source_file("output.yaml");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();

    auto readback = rocRoller::GPUArchitecture::readYaml("output.yaml");
    EXPECT_EQ(readback.size(), 14);
    for(auto& x : readback)
    {
        std::cout << x.first << ": " << x.second.target() << '\n';
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.yaml");
}

TEST(GPUArchitectureGeneratorTest, BasicMsgpack)
{
    GPUArchitectureGenerator::FillArchitectures();

    GPUArchitectureGenerator::GenerateFile("output.msgpack");

    std::ifstream     generated_source_file("output.msgpack");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();

    auto readback = rocRoller::GPUArchitecture::readMsgpack("output.msgpack");
    EXPECT_EQ(readback.size(), 14);
    for(auto& x : readback)
    {
        std::cout << x.first << ": " << x.second.target() << '\n';
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.msgpack");
}
