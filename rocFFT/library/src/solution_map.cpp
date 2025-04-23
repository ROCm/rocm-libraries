/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "solution_map.h"
#include "../../shared/environment.h"
#include "data_descriptor.h"
#include "library_path.h"
#include "logging.h"
#include "node_factory.h"

#include <fstream>

namespace fs = std::filesystem;

#define REGEX "[^:;,\"\\{\\}\\[\\s]+"

static const char* def_solution_map_path = "rocfft_solution_map.dat";

const int   solution_map::VERSION                       = 3;
const char* solution_map::KERNEL_TOKEN_BUILTIN_KERNEL   = "kernel_token_builtin_kernel";
const char* solution_map::LEAFNODE_TOKEN_BUILTIN_KERNEL = "leafnode_token_builtin_kernel";

static std::map<SolutionNodeType, std::string> SolutionNodeTypetoStrMap()
{
    std::map<SolutionNodeType, std::string> SNTtoStr = {{SOL_DUMMY, "SOL_DUMMY"},
                                                        {SOL_INTERNAL_NODE, "SOL_INTERNAL_NODE"},
                                                        {SOL_LEAF_NODE, "SOL_LEAF_NODE"},
                                                        {SOL_KERNEL_ONLY, "SOL_KERNEL_ONLY"},
                                                        {SOL_BUILTIN_KERNEL, "SOL_BUILTIN_KERNEL"}};
    return SNTtoStr;
}

static std::map<std::string, SolutionNodeType> StrToSolutionNodeTypeMap()
{
    std::map<std::string, SolutionNodeType> StrToSNT;
    for(auto i : SolutionNodeTypetoStrMap())
        StrToSNT.emplace(i.second, i.first);
    return StrToSNT;
}

std::string PrintSolutionNodeType(const SolutionNodeType snt)
{
    static auto SNTtoString = SolutionNodeTypetoStrMap();
    return SNTtoString.at(snt);
}

SolutionNodeType StrToSolutionNodeType(const std::string& str)
{
    static auto str2SNT = StrToSolutionNodeTypeMap();
    return str2SNT.at(str);
}

static fs::path get_solution_map_path(const std::string& read_folder,
                                      const std::string& arch = "any")
{
    // find file in that folder,
    fs::path folder_path(read_folder.c_str());

    // search the file with the arch prefix
    std::string prefix(arch + "_");
    fs::path    file_name(prefix.c_str());
    file_name += def_solution_map_path;

    return folder_path / file_name;
}

template <>
struct ToString<SolutionPtr>
{
    std::string print(const SolutionPtr& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("child_token", value.child_token) + ",";
        str += FieldDescriptor<size_t>().describe("child_option", value.child_option);
        str += "}";
        return str;
    }
};

template <>
struct FromString<SolutionPtr>
{
    void Get(SolutionPtr& ret, std::sregex_token_iterator& current) const
    {
        FieldParser<std::string>().parse("child_token", ret.child_token, current);
        FieldParser<size_t>().parse("child_option", ret.child_option, current);
    }
};

template <>
struct ToString<SolutionNode>
{
    std::string print(const SolutionNode& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("sol_node_type",
                                                       PrintSolutionNodeType(value.sol_node_type));

        if(value.sol_node_type != SOL_BUILTIN_KERNEL)
        {
            str += ",";
            if(value.sol_node_type == SOL_KERNEL_ONLY)
            {
                str += FieldDescriptor<FMKey>().describe("kernel_key", value.kernel_key);
            }
            else
            {
                str += FieldDescriptor<std::string>().describe("using_scheme",
                                                               PrintScheme(value.using_scheme))
                       + ",";
                str += VectorFieldDescriptor<SolutionPtr>().describe("solution_childnodes",
                                                                     value.solution_childnodes);
            }
        }

        str += "}";
        return str;
    }
};

template <>
struct FromString<SolutionNode>
{
    void Get(SolutionNode& ret, std::sregex_token_iterator& current) const
    {
        std::string sol_node_type_str;
        std::string scheme_str;

        FieldParser<std::string>().parse("sol_node_type", sol_node_type_str, current);
        ret.sol_node_type = StrToSolutionNodeType(sol_node_type_str);

        if(ret.sol_node_type != SOL_BUILTIN_KERNEL)
        {
            if(ret.sol_node_type == SOL_KERNEL_ONLY)
            {
                FieldParser<FMKey>().parse("kernel_key", ret.kernel_key, current);
                ret.using_scheme = ret.kernel_key.scheme;
            }
            else
            {
                FieldParser<std::string>().parse("using_scheme", scheme_str, current);

                if(DescriptorFormatVersion::UsingVersion <= 2)
                {
                    // (version <= 2) supports scheme CS_KERNEL_APPLY_CALLBACK
                    // to convert to later version, we change it to CS_NONE first,
                    // and then we will remove these nodes if (scheme == CALLBACK && sol_node_type == LEAF_NODE)
                    if(scheme_str == "CS_KERNEL_APPLY_CALLBACK")
                        scheme_str = "CS_NONE";
                }

                ret.using_scheme = StrToComputeScheme(scheme_str);

                VectorFieldParser<SolutionPtr>().parse(
                    "solution_childnodes", ret.solution_childnodes, current);
            }
        }
    }
};

template <>
struct ToString<ProblemKey>
{
    std::string print(const ProblemKey& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("arch", value.arch) + ",";
        str += FieldDescriptor<std::string>().describe("token", value.probToken);
        str += "}";
        return str;
    }
};

template <>
struct FromString<ProblemKey>
{
    void Get(ProblemKey& ret, std::sregex_token_iterator& current) const
    {
        std::string arch, token;
        FieldParser<std::string>().parse("arch", arch, current);
        FieldParser<std::string>().parse("token", token, current);
        ret = {arch, token};
    }
};

template <>
struct ToString<SolMapEntry>
{
    std::string print(const SolMapEntry& value) const
    {
        static const std::string blanks(14, ' ');

        std::string str = "\n{";
        str += FieldDescriptor<ProblemKey>().describe("Problem", value.first) + ",\n ";
        str += VectorFieldDescriptor<SolutionNode>().describe(
            "Solutions", value.second, true, blanks);
        str += "}";
        return str;
    }
};

template <>
struct FromString<SolMapEntry>
{
    void Get(SolMapEntry& ret, std::sregex_token_iterator& current) const
    {
        FieldParser<ProblemKey>().parse("Problem", ret.first, current);
        VectorFieldParser<SolutionNode>().parse("Solutions", ret.second, current);
    }
};

//////////////////////
// Private Functions
//////////////////////
// private version called by constructor
size_t solution_map::add_solution_private(const ProblemKey& probKey, const SolutionNode& solution)
{
    // no this key, emplace one new vector
    if(primary_sol_map.count(probKey) == 0)
        primary_sol_map.emplace(probKey, SolutionNodeVec());

    auto& sol_vec = primary_sol_map.at(probKey);
    sol_vec.push_back(solution);
    return sol_vec.size() - 1;
}

bool solution_map::SolutionNodesAreEqual(const SolutionNode& lhs,
                                         const SolutionNode& rhs,
                                         const std::string&  arch,
                                         bool                primary_map)
{
    // NB:
    // std::tie couldn't compare the .size() so we compare .size() outside std::tie
    bool members_equal = std::tie(lhs.sol_node_type, lhs.using_scheme, lhs.kernel_key)
                         == std::tie(rhs.sol_node_type, rhs.using_scheme, rhs.kernel_key);
    if((members_equal == false)
       || (lhs.solution_childnodes.size() != rhs.solution_childnodes.size()))
        return false;

    for(size_t i = 0; i < lhs.solution_childnodes.size(); ++i)
    {
        auto& lhs_child_ptr = lhs.solution_childnodes[i];
        auto& rhs_child_ptr = rhs.solution_childnodes[i];

        auto lhs_child_key = ProblemKey(arch, lhs_child_ptr.child_token);
        auto rhs_child_key = ProblemKey(arch, rhs_child_ptr.child_token);

        if(SolutionNodesAreEqual(
               get_solution_node(lhs_child_key, lhs_child_ptr.child_option, primary_map),
               get_solution_node(rhs_child_key, rhs_child_ptr.child_option, primary_map),
               arch,
               primary_map)
           == false)
            return false;
    }

    return true;
}

bool solution_map::remove_solution_tree(SolutionNodeVec& nodeVec, SolutionNode& node, size_t pos)
{
    node.to_be_removed = true;

    // this node is going to be removed, so the following elements change position.
    // We need to update their parent_sol_ptr's option_id
    size_t i = pos + 1;
    for(; i < nodeVec.size(); ++i)
    {
        auto& ref_ptrs = nodeVec[i].parent_sol_ptrs;
        for(auto& ptr : ref_ptrs)
            ptr->child_option -= 1;
    }

    // remove node and its parent solution nodes
    for(auto& parent : node.parent_sol_nodes)
    {
        SolutionNodeVec& vec = *(parent->self_vec);
        auto             it  = std::find(vec.begin(), vec.end(), *parent);
        size_t           idx = it - vec.begin();
        // recursion
        remove_solution_tree(vec, *parent, idx);
    }

    return true;
}

bool solution_map::remove_solution_node(SolutionNodeVec& nodeVec, SolutionNode& node, size_t pos)
{
    node.to_be_removed = true;

    SolutionPtr remove_ptr{node.parent_sol_ptrs[0]->child_token,
                           node.parent_sol_ptrs[0]->child_option};
    // remove node and update its parent's child_vector
    for(auto& parent : node.parent_sol_nodes)
    {
        auto& children    = parent->solution_childnodes;
        auto  it          = std::find(children.begin(), children.end(), remove_ptr);
        it->to_be_removed = true;
    }

    // this node is going to be removed, so the following elements change position.
    // We need to update their parent_sol_ptr 's option_id
    size_t i = pos + 1;
    for(; i < nodeVec.size(); ++i)
    {
        auto& ref_ptrs = nodeVec[i].parent_sol_ptrs;
        for(auto& ptr : ref_ptrs)
            ptr->child_option -= 1;
    }

    return true;
}

void solution_map::generate_link_info()
{
    for(auto& [key, value] : primary_sol_map)
    {
        SolutionNodeVec& solNodeVec = value;
        for(SolutionNode& node : solNodeVec)
        {
            // update the self_vec
            node.self_vec = &solNodeVec;

            for(auto& child_node_ptr : node.solution_childnodes)
            {
                ProblemKey    pKey(key.arch, child_node_ptr.child_token);
                SolutionNode& child = get_solution_node(pKey, child_node_ptr.child_option);

                // update children's parent infos
                child.parent_sol_nodes.push_back(&node);
                child.parent_sol_ptrs.push_back(&child_node_ptr);
            }
        }
    }
}

//////////////////////
// Public Functions
//////////////////////
void solution_map::setup(const std::string& arch_name)
{
    // if we have speicified an explicit file-path, then read from it,
    std::string explict_read_path_str = rocfft_getenv("ROCFFT_READ_EXPLICIT_SOL_MAP_FILE");
    if(!explict_read_path_str.empty())
    {
        fs::path read_from_path(explict_read_path_str.c_str());
        read_solution_map_data(read_from_path);
        return;
    }

    // set ROCFFT_READ_SOL_MAP_FROM_FOLDER to enable reading solution map text files in runtime
    // default is empty
    std::string read_folder_str = rocfft_getenv("ROCFFT_READ_SOL_MAP_FROM_FOLDER");
    if(!read_folder_str.empty())
    {
        // read data from any_arch
        auto sol_map_input = get_solution_map_path(read_folder_str);
        read_solution_map_data(sol_map_input);

        // read data from current arch
        sol_map_input = get_solution_map_path(read_folder_str, arch_name);
        read_solution_map_data(sol_map_input);
    }
}

bool solution_map::has_solution_node(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    // no this key
    if(dst_map.count(probKey) == 0)
        return false;

    // no this option_id
    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions.size() > option_id;
}

SolutionNode&
    solution_map::get_solution_node(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    // be sure we have checked has_solution_node();
    if(!has_solution_node(probKey, option_id, primary_map))
        throw std::runtime_error(
            "get_solution_node failed. the solution_node doesn't exist: ProbKey=(" + probKey.arch
            + "," + probKey.probToken + "), option_id=" + std::to_string(option_id));

    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id];
}

FMKey&
    solution_map::get_solution_kernel(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    // be sure we have checked has_solution_node();
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id].kernel_key;
}

// setup a solution of a problem and insert to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey&               probKey,
                                  TreeNode*                       currentNode,
                                  const std::vector<SolutionPtr>& children,
                                  bool                            isRootProb,
                                  bool                            check_dup,
                                  bool                            primary_map)
{
    SolutionNode solution;
    solution.arch_name     = probKey.arch;
    solution.using_scheme  = currentNode->scheme;
    solution.sol_node_type = (currentNode->nodeType == NT_LEAF) ? SOL_LEAF_NODE : SOL_INTERNAL_NODE;
    solution.solution_childnodes = children;

    return add_solution(probKey, solution, isRootProb, check_dup, primary_map);
}

// setup a solution of a problem and insert to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey& probKey,
                                  const FMKey&      kernel_key,
                                  bool              check_dup,
                                  bool              primary_map)
{
    SolutionNode solution;
    solution.arch_name    = probKey.arch;
    solution.using_scheme = (kernel_key == FMKey::EmptyFMKey()) ? CS_NONE : kernel_key.scheme;
    solution.sol_node_type
        = (kernel_key == FMKey::EmptyFMKey()) ? SOL_BUILTIN_KERNEL : SOL_KERNEL_ONLY;
    solution.kernel_key = kernel_key;

    return add_solution(probKey, solution, false, check_dup, primary_map);
}

// directly insert a solution of a problem to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey&   probKey,
                                  const SolutionNode& solution,
                                  bool                isRootProb,
                                  bool                check_dup,
                                  bool                primary_map)
{
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    // CS_KERNEL_STOCKHAM could be a problem scheme but also a kernel.
    bool is_problem_scheme = ComputeSchemeIsAProblem(solution.using_scheme)
                             && (solution.sol_node_type != SOL_KERNEL_ONLY);

    // no this key, emplace one new vector
    if(dst_map.count(probKey) == 0)
    {
        dst_map.emplace(probKey, SolutionNodeVec());

        // if this is a solution for a problem (not kernel or non-problem)
        // then we insert a dummy one in the front, which is reserved for root-problem
        if(is_problem_scheme)
            dst_map.at(probKey).push_back(SolutionNode::DummySolutionNode());
    }

    auto& sol_vec = dst_map.at(probKey);

    // append solution to the same key
    // Root-solution never checks duplication since it will always be the first element
    // So adding a root-solution = simply overwrite the first one
    if(isRootProb)
    {
        // just a double-check
        assert(is_problem_scheme && (sol_vec.size() > 0));
        sol_vec[0] = solution;
        return 0;
    }
    else if(check_dup)
    {
        // if the solution is not a problem-solution (i.e. kernel solution or non-problem)
        // then there is no a "dummy solution" (or exclusive root-solution) in the vector,
        // so we still start from 0, otherwise, start from option 1.
        const std::string& arch            = probKey.arch;
        size_t             check_option_id = (is_problem_scheme) ? 1 : 0;
        for(; check_option_id < sol_vec.size(); ++check_option_id)
        {
            // find an existing solution that is identical, then don't insert, simply return that option
            if(SolutionNodesAreEqual(solution, sol_vec[check_option_id], arch, primary_map))
                return check_option_id;
        }
    }

    // if we are here, it could be either check_dup but not found any indentical,
    // or force append, don't check_dup (for tuning)
    sol_vec.push_back(solution);

    return sol_vec.size() - 1;
}

bool solution_map::get_typed_nodes_of_tree(const SolutionNode&     root,
                                           SolutionNodeType        type,
                                           std::set<SolutionNode>& ret)
{
    if(root.sol_node_type == type)
    {
        ret.insert(root);
    }
    else
    {
        for(const auto& child : root.solution_childnodes)
        {
            auto& child_node = get_solution_node(ProblemKey(root.arch_name, child.child_token),
                                                 child.child_option);
            get_typed_nodes_of_tree(child_node, type, ret);
        }
    }

    return true;
}

bool solution_map::get_all_kernels(std::vector<SolutionNode>& sol_kernels, bool getUsedOnly)
{
    // if get all, then we simply return all "SOL_KERNEL_ONLY" nodes
    if(!getUsedOnly)
    {
        for(auto& [key, value] : primary_sol_map)
        {
            SolutionNodeVec& sol_vec = value;
            if(sol_vec.front().sol_node_type != SOL_KERNEL_ONLY)
                continue;

            for(auto& kernel : sol_vec)
                sol_kernels.push_back(kernel);
        }
    }
    // if get used only, then we need to start from a real-root-problem and return its kernels
    // the principle of a root-problem is:
    //   1. if the first element of a sol_vec is SOL_INTERNAL_NODE, not DUMMY, then it is a root
    //      (Large 1d, 2d, 3d), if not put in index-0, then they are sub-problems
    //   2. if the first element of a sol_vec is SOL_LEAF_NODE, and ComputeSchemeIsAProblem() is true
    //      (Small-1d, single-2d), leaf-node could also be a root-problem
    else
    {
        std::set<SolutionNode> kernels_set; // to avoid duplicates
        for(auto& [key, value] : primary_sol_map)
        {
            SolutionNodeVec& sol_vec    = value;
            SolutionNode&    first_node = sol_vec.front();

            if((first_node.sol_node_type == SOL_INTERNAL_NODE)
               || (first_node.sol_node_type == SOL_LEAF_NODE
                   && ComputeSchemeIsAProblem(first_node.using_scheme)))
            {
                // get first_node's all kernels
                get_typed_nodes_of_tree(first_node, SOL_KERNEL_ONLY, kernels_set);
            }
        }

        std::copy(kernels_set.begin(), kernels_set.end(), std::back_inserter(sol_kernels));
    }

    return true;
}

// parse the format version of the input file, call by converter
bool solution_map::get_solution_map_version(const fs::path& sol_map_in_path)
{
    static std::regex regEx(REGEX, std::regex_constants::optimize);

    if(LOG_TRACE_ENABLED())
        (*LogSingleton::GetInstance().GetTraceOS())
            << "reading solution map data from: " << sol_map_in_path.c_str() << std::endl;

    if(fs::exists(sol_map_in_path))
    {
        std::ifstream in_file(sol_map_in_path.c_str());
        std::string   line;

        while(std::getline(in_file, line))
        {
            std::size_t found = line.find("Version");
            if(found != std::string::npos)
            {
                std::sregex_token_iterator tokens{line.begin(), line.end(), regEx, 0};
                FieldParser<int>().parse("Version", self_version, tokens);
                break;
            }
        }
        return true;
    }

    return false;
}

// read the map from input stream
bool solution_map::read_solution_map_data(const fs::path& sol_map_in_path, bool primary_map)
{
    static std::regex regEx(REGEX, std::regex_constants::optimize);

    if(LOG_TRACE_ENABLED())
        (*LogSingleton::GetInstance().GetTraceOS())
            << "reading solution map data from: " << sol_map_in_path.c_str() << std::endl;

    // Read text from the file. If file not found, do nothing
    std::string solution_map_text = "";
    if(fs::exists(sol_map_in_path))
    {
        std::ifstream in_file(sol_map_in_path.c_str());
        std::string   line;

        while(std::getline(in_file, line))
        {
            solution_map_text += line;
        }
    }

    if(solution_map_text.back() == ']')
        solution_map_text.resize(solution_map_text.size() - 1);

    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    std::sregex_token_iterator tokens{solution_map_text.begin(), solution_map_text.end(), regEx, 0};
    std::sregex_token_iterator endIt;
    if(tokens == endIt)
    {
        if(LOG_TRACE_ENABLED())
            (*LogSingleton::GetInstance().GetTraceOS())
                << "\tfile not found or file is empty" << std::endl;
        return false;
    }

    auto latestParseProcess = [=, &dst_map, &tokens]() {
        std::vector<SolMapEntry> entry_vec;
        VectorFieldParser<SolMapEntry>().parse("Data", entry_vec, tokens);
        for(auto& entry : entry_vec)
        {
            // automatically set arch_name which is not in the text file
            for(auto& sol : entry.second)
                sol.arch_name = entry.first.arch;
            dst_map.emplace(entry.first, entry.second);
        }
    };

    // should be latest version as long as it's not called by converter
    if(assume_latest_ver)
    {
        try
        {
            FieldParser<int>().parse("Version", self_version, tokens);
            if(self_version != solution_map::VERSION)
                throw std::runtime_error("format version of the input file is not the latest, "
                                         "please execute the solution map converter first.");

            // set the descriptor / parse to the latest version
            DescriptorFormatVersion::UsingVersion = solution_map::VERSION;

            // always do the latest version reading here
            latestParseProcess();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return false;
        }

        return true;
    }

    /* only converter reaches here, and the self_version was already parsed outside.
       we always put the latest reading in (assume_latest_ver) block, and move old reading below*/

    // reading the oldest format: no version number, i.e, is 0
    if(self_version == 0)
    {
        for(; tokens != endIt; ++tokens)
        {
            ProblemKey                probKey;
            std::vector<SolutionNode> solutionVec;
            FieldParser<ProblemKey>().parse("Problem", probKey, tokens);
            VectorFieldParser<SolutionNode>().parse("Solutions", solutionVec, tokens);
            dst_map.emplace(probKey, solutionVec);
        }
    }
    else if(self_version == 1 || self_version == 2)
    {
        try
        {
            FieldParser<int>().parse("Version", self_version, tokens);
            // set the descriptor / parse to the corresponding
            DescriptorFormatVersion::UsingVersion = self_version;
            // if the difference of the version can be handled by parser internally,
            // we still can call the latestParseProcess(). Just remember to set
            // DescriptorFormatVersion::UsingVersion
            latestParseProcess();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return false;
        }
    }
    // handling other version in the future
    // else if() { ... }
    // else if() { ... }

    return true;
}

// write the map to output stream
bool solution_map::write_solution_map_data(const fs::path& sol_map_out_path,
                                           bool            sort,
                                           bool            primary_map)
{
    if(LOG_TUNING_ENABLED())
        (*LogSingleton::GetInstance().GetTuningOS())
            << "writing solution map data to: " << sol_map_out_path.c_str() << std::endl;

    std::ofstream outfile;
    outfile.open(sol_map_out_path.c_str(), (std::ios::out | std::ios::trunc));
    if(!outfile.is_open())
        throw std::runtime_error("Write solution map failed. Cannot open/create output file: "
                                 + sol_map_out_path.string());

    std::stringstream ss;
    ProbSolMap&       writing_map = (primary_map) ? primary_sol_map : temp_working_map;

    std::vector<SolMapEntry> entry_vec;
    for(auto& [key, value] : writing_map)
        entry_vec.push_back(std::make_pair(key, value));

    // sort !
    if(sort)
        std::sort(entry_vec.begin(), entry_vec.end(), ProbSolCmp);

    // guard for safety, though we are always writing with latest version format
    DescriptorFormatVersion::UsingVersion = solution_map::VERSION;

    // write version at the beginning
    ss << "{";
    ss << FieldDescriptor<int>().describe("Version", solution_map::VERSION) << "," << std::endl;
    ss << VectorFieldDescriptor<SolMapEntry>().describe("Data", entry_vec) << std::endl;
    ss << "}";

    outfile << ss.str();
    outfile.close();

    return true;
}

bool solution_map::merge_solutions_from_file(const fs::path&                src_file,
                                             const std::vector<ProblemKey>& root_probs)
{
    bool check_dup       = true;
    bool read_to_primary = false;
    if(read_solution_map_data(src_file, read_to_primary) == false)
        return false;

    // An important note is that we can't use SolutionNode& (alias) for the second arg,
    // since a node may be shared with others (e.g. RTRT, where 2 Rs shared the same one)
    // This Recur-Add-Solution is to add a sub-tree from mapA to mapB, so what we are doing
    // here is not just moving but also updating the "option_id" of a node's child (children
    // are also moved from mapA to mapB). If we pass by reference and update the option_id,
    // Then we also change the data of an "un-moved" node (in mapA). So using call by copy
    // is safer here.
    auto RecursivelyAddSolution = [&](const ProblemKey& key,
                                      SolutionNode      solution,
                                      bool              isRoot,
                                      bool              from_primary,
                                      bool              to_primary,
                                      auto&&            RecursivelyAddSolution) -> size_t {
        std::string archName = key.arch;
        for(auto& child : solution.solution_childnodes)
        {
            ProblemKey childKey(archName, child.child_token);

            // get the child solution object from current solution map (using existing child_option)
            auto& childSol = get_solution_node(childKey, child.child_option, from_primary);

            // since we are add solution to another solution map , so we have to update the child option
            child.child_option = RecursivelyAddSolution(
                childKey, childSol, false, from_primary, to_primary, RecursivelyAddSolution);
        }
        return add_solution(key, solution, isRoot, check_dup, to_primary);
    };

    // for each root-problem, adding the whole solution-tree. from one map to another map
    for(auto& rootProbKey : root_probs)
    {
        bool isRootProb     = true;
        bool getFromPrimary = read_to_primary;
        bool addToPrimary   = !getFromPrimary;

        SolutionNode& sol_node = get_solution_node(rootProbKey, 0, getFromPrimary);

        size_t option_id = RecursivelyAddSolution(rootProbKey,
                                                  sol_node,
                                                  isRootProb,
                                                  getFromPrimary,
                                                  addToPrimary,
                                                  RecursivelyAddSolution);

        // we are adding solution for a root-problem, so the root-solution should always be at index 0
        if(option_id != 0)
            throw std::runtime_error(
                "Merge failed. Inserting solution of a root-problem should return an index 0");
    }

    return true;
}

//////////////////////
// Version Converter
//////////////////////
bool SolutionMapConverter::remove_invalid_half_lds()
{
    static const std::set<ComputeScheme> no_half_lds
        = {CS_KERNEL_STOCKHAM_BLOCK_CR,
           CS_KERNEL_STOCKHAM_BLOCK_RC,
           CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
           CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
           CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY};

    auto& sol_map = solution_map::get_solution_map();

    // generate the parent's info for back reference
    sol_map.generate_link_info();

    // mark those nodes need removing
    for(auto& [key, value] : sol_map.primary_sol_map)
    {
        SolutionNodeVec& solNodeVec = value;
        for(size_t i = 0; i < solNodeVec.size(); ++i)
        {
            SolutionNode& node = solNodeVec[i];
            if(node.sol_node_type == SOL_KERNEL_ONLY)
            {
                ComputeScheme scheme = node.kernel_key.scheme;
                KernelConfig& config = node.kernel_key.kernel_config;
                if(config.half_lds && no_half_lds.count(scheme))
                {
                    sol_map.remove_solution_tree(solNodeVec, node, i);
                }
            }
            else
            {
                break;
            }
        }
    }

    remove_all_marked(sol_map.primary_sol_map);
    return true;
}

bool SolutionMapConverter::remove_callback_nodes()
{
    auto& sol_map = solution_map::get_solution_map();

    // generate the parent's info for back reference
    sol_map.generate_link_info();

    // mark those nodes need removing
    for(auto& [key, value] : sol_map.primary_sol_map)
    {
        SolutionNodeVec& solNodeVec = value;
        for(size_t i = 0; i < solNodeVec.size(); ++i)
        {
            SolutionNode& node = solNodeVec[i];
            if(node.sol_node_type == SOL_LEAF_NODE && node.using_scheme == CS_NONE)
                sol_map.remove_solution_node(solNodeVec, node, i);
        }
    }

    remove_all_marked(sol_map.primary_sol_map);
    return true;
}

void SolutionMapConverter::remove_all_marked(ProbSolMap& target_map)
{
    // erase nodes, and check if any entry need removing
    std::set<ProblemKey> to_be_removed_keys;
    for(auto& [key, value] : target_map)
    {
        SolutionNodeVec& solNodeVec = value;
        for(size_t i = 0; i < solNodeVec.size(); ++i)
        {
            // remove nodes that are marked removed
            SolutionNode& node = solNodeVec[i];
            if(node.to_be_removed)
            {
                solNodeVec.erase(solNodeVec.begin() + i);
                --i; // stay at the same position after removing this element
            }

            auto& child_vec = node.solution_childnodes;
            for(size_t child_id = 0; child_id < child_vec.size(); ++child_id)
            {
                // remove child_ptrs that are marked removed
                SolutionPtr& child = child_vec[child_id];
                if(child.to_be_removed)
                {
                    child_vec.erase(child_vec.begin() + child_id);
                    --child_id; // stay at the same position after removing this element
                }
            }
        }
        if(solNodeVec.empty())
            to_be_removed_keys.insert(key);
    }

    // remove entries with zero-length sol-node-vec
    for(const auto& key : to_be_removed_keys)
        target_map.erase(key);
}

bool SolutionMapConverter::VersionCheckAndConvert(const std::string& in_map_path,
                                                  const std::string& out_map_path)
{
    auto& sol_map = solution_map::get_solution_map();

    try
    {
        // don't assert latest version, so that we can read any version in read_solution_map_data()
        sol_map.assume_latest_ver = false;

        // parse the version individually first
        if(sol_map.get_solution_map_version(in_map_path) == false)
            return false;

        bool has_conversion = sol_map.self_version != solution_map::VERSION;
        if(!has_conversion)
        {
            std::cout << "solution map is already at the latest version.\n";
            return true;
        }

        // the read function should be able to read the file according to the parsed self_version
        if(sol_map.read_solution_map_data(in_map_path) == false)
            return false;

        // ---------
        // some actions that convert the current data to the latest one
        if(sol_map.self_version == 0)
            remove_invalid_half_lds();

        // other actions that need to make it fitting latest version
        // ---------
        if(sol_map.self_version <= 2)
            remove_callback_nodes();

        std::cout << "successfully converted solution map from version(" << sol_map.self_version
                  << ") to latest version(" << solution_map::VERSION << ").\n";
        return sol_map.write_solution_map_data(out_map_path);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }

    return true;
}