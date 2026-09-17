#pragma once

#include <string>

int make_autoshard(std::string const& target_dir,
                   char const* arch_name,
                   char const* kernel_namespace,
                   char const* kernel_class,
                   char const* config_table_header,
                   char const* kernel_header,
                   int num_configs,
                   int num_shards);
