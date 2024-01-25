#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <GPUArchitectureGenerator/GPUArchitectureGenerator.hpp>

class GPUArchitectureGeneratorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        GPUArchitectureGenerator::FillArchitectures();
    }
};

TEST_F(GPUArchitectureGeneratorTest, BasicYAML)
{
    GPUArchitectureGenerator::GenerateFile("output.yaml", true);

    std::ifstream     generated_source_file("output.yaml");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();
    EXPECT_NE(generated_source, "");

    auto readback = rocRoller::GPUArchitecture::readYaml("output.yaml");
    EXPECT_EQ(readback.size(), 18);
    for(auto& x : readback)
    {
        std::cout << x.first << ": " << x.second.target() << '\n';
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.yaml");
}

TEST_F(GPUArchitectureGeneratorTest, BasicMsgpack)
{
    GPUArchitectureGenerator::GenerateFile("output.msgpack");

    std::ifstream     generated_source_file("output.msgpack");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();
    EXPECT_NE(generated_source, "");

    auto readback = rocRoller::GPUArchitecture::readMsgpack("output.msgpack");
    EXPECT_EQ(readback.size(), 18);
    for(auto& x : readback)
    {
        std::cout << x.first << ": " << x.second.target() << '\n';
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.msgpack");
}
