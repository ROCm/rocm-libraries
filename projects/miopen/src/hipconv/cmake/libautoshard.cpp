#include "libautoshard.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

int make_autoshard(std::string const& target_dir,
                   char const* arch_name,
                   char const* kernel_namespace,
                   char const* kernel_class,
                   char const* config_table_header,
                   char const* kernel_header,
                   int num_configs,
                   int num_shards)
{
    if(!std::filesystem::is_directory(target_dir))
        return 2;

    const std::string the_namespace =
        (std::ostringstream{} << "hipconv::" << arch_name << "::" << kernel_namespace).str();
    const std::string the_symbol =
        (std::ostringstream{} << kernel_namespace << "_" << arch_name << "_kernels").str();
    const std::string prefix =
        (std::ostringstream{} << target_dir << "/" << kernel_namespace << "_").str();

    for(int shard = 0; shard < num_shards; ++shard)
    {
        const auto file_name = (std::ostringstream{} << prefix << "shard" << shard << ".cpp").str();
        auto file            = std::ofstream(file_name);
        if(!file.good())
            return 3;
        file << "#include \"" << config_table_header << "\"\n";
        file << "#include \"" << kernel_header << "\"\n";
        file << "#include <hip/hip_runtime.h>\n";
        file << "namespace " << the_namespace << "{";
        for(int cfg = shard; cfg < num_configs; cfg += num_shards)
        {
            file << "template void " << the_namespace << "::launch_impl<configs[" << cfg
                 << "]>(const LaunchParams&, const Conv2dParams&, const void*, const void*, void*, "
                    "void*, hipStream_t);";
        }
        file << "}\n";
    }

    {
        const auto file_name = (std::ostringstream{} << prefix << "kernel_table.cpp").str();
        auto file            = std::ofstream(file_name);
        if(!file.good())
            return 3;
        file << "#include \"" << config_table_header << "\"\n";
        file << "#include \"" << kernel_header << "\"\n";
        file << "#include \"conv_kernel.h\"\n";
        file << "#include <hip/hip_runtime.h>\n";
        file << "namespace " << the_namespace << "{";
        for(int cfg = 0; cfg < num_configs; ++cfg)
        {
            file << "extern template void " << the_namespace << "::launch_impl<configs[" << cfg
                 << "]>(const LaunchParams&, const Conv2dParams&, const void*, const void*, void*, "
                    "void*, hipStream_t);";
        }
        file << "inline auto kernels=std::array<" << kernel_class << "," << num_configs << ">{";
        for(int cfg = 0; cfg < num_configs; ++cfg)
        {
            file << kernel_class << "{configs[" << cfg << "],&launch_impl<configs[" << cfg
                 << "]>},";
        }
        file << "};inline auto kernel_ptrs=std::array<hipconv::ConvKernel*," << num_configs << ">{";
        for(int cfg = 0; cfg < num_configs; ++cfg)
        {
            file << "&kernels[" << cfg << "],";
        }
        file << "};}\n"
                "#ifndef __HIP_DEVICE_COMPILE__\n"
                "extern const auto "
             << the_symbol << "=hipconv::ConvKernelSpan{" << the_namespace
             << "::kernel_ptrs};\n"
                "#endif\n";
    }

    return 0;
}
