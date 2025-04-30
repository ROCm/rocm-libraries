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

#ifndef SOLUTION_MAP_H
#define SOLUTION_MAP_H

#include "tree_node.h"
#include <unordered_map>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

enum SolutionNodeType
{
    SOL_DUMMY          = 0, // a reserved solution slot for a root-problem
    SOL_BUILTIN_KERNEL = 1, // a solution representing a built-in kernel (transpose, r2c, c2r...)
    SOL_KERNEL_ONLY    = 2, // a solution representing a kernel only, nothing to do with problem
    SOL_LEAF_NODE      = 3, // a solution representing the tree-leaf-node
    SOL_INTERNAL_NODE  = 4, // a solution with tree decomposition
};

std::string      PrintSolutionNodeType(const SolutionNodeType snt);
SolutionNodeType StrToSolutionNodeType(const std::string& str);

// arch, problem-size token
struct ProblemKey
{
    std::string arch;
    std::string probToken;

    ProblemKey() = default;

    ProblemKey(const std::string& arch, const std::string& probToken)
        : arch(arch)
        , probToken(probToken)
    {
    }

    bool operator==(const ProblemKey& rhs) const
    {
        return (arch == rhs.arch) && (probToken == rhs.probToken);
    }

    bool operator<(const ProblemKey& rhs) const
    {
        if(arch != rhs.arch)
            return arch < rhs.arch;

        return probToken < rhs.probToken;
    }
};

template <>
struct ToString<ProblemKey>;

template <>
struct FromString<ProblemKey>;

struct ProbKeyHash
{
    size_t operator()(const ProblemKey& k) const noexcept
    {
        size_t h = 0;
        h ^= std::hash<std::string>{}(k.arch);
        h ^= std::hash<std::string>{}(k.probToken);
        return h;
    }
};

struct SolutionPtr
{
    std::string child_token  = "";
    size_t      child_option = 0;

    bool operator==(const SolutionPtr& rhs) const
    {
        return (child_token == rhs.child_token) && (child_option == rhs.child_option);
    }

    bool operator<(const SolutionPtr& rhs) const
    {
        if(child_token == rhs.child_token)
            return child_option < rhs.child_option;

        return child_token < rhs.child_token;
    }

    // flag indicating this to be removed, we assign marks first and remove them all at once
    bool to_be_removed = false;
};

// Implementing the ToString / FromString (data_descriptor.h)
// for writing-to/reading-from texted-format solution map
template <>
struct ToString<SolutionPtr>;

template <>
struct FromString<SolutionPtr>;

struct SolutionNode;

using SolutionNodeVec = std::vector<SolutionNode>;

using ProbSolMap = std::unordered_map<ProblemKey, SolutionNodeVec, ProbKeyHash>;

struct SolutionNode
{
    std::string      arch_name; // a good way to reverse look up
    SolutionNodeType sol_node_type = SOL_INTERNAL_NODE;
    ComputeScheme    using_scheme  = CS_NONE;
    FMKey            kernel_key    = FMKey::EmptyFMKey();
    // like the childnodes on tree-node, a childnode could be internal/leaf/kernel-node
    std::vector<SolutionPtr> solution_childnodes;

    SolutionNode()                    = default;
    SolutionNode(const SolutionNode&) = default;

    static SolutionNode DummySolutionNode()
    {
        static SolutionNode dummy;
        dummy.sol_node_type = SOL_DUMMY;
        return dummy;
    }

    SolutionNode& operator=(const SolutionNode&) = default;

    bool operator==(const SolutionNode& rhs) const
    {
        return std::tie(arch_name, sol_node_type, using_scheme, kernel_key, solution_childnodes)
               == std::tie(rhs.arch_name,
                           rhs.sol_node_type,
                           rhs.using_scheme,
                           rhs.kernel_key,
                           rhs.solution_childnodes);
    }

    bool operator<(const SolutionNode& rhs) const
    {
        return std::tie(arch_name, sol_node_type, using_scheme, kernel_key, solution_childnodes)
               < std::tie(rhs.arch_name,
                          rhs.sol_node_type,
                          rhs.using_scheme,
                          rhs.kernel_key,
                          rhs.solution_childnodes);
    }

    // NB:
    //  The following are only assigned when calling "remove solution"
    //  in normal case, we don't assign anything to them
    //  if a node is deleted, then its parent_sol_node should be deleted
    std::vector<SolutionNode*> parent_sol_nodes;
    //  if a node is changing its position, then its parent_sol_ptr should update it option_id
    std::vector<SolutionPtr*> parent_sol_ptrs;
    //  use to find the SolutionNodeVec containing itself
    SolutionNodeVec* self_vec;
    // flag indicating this to be removed, we assign marks first and remove them all at once
    bool to_be_removed = false;
};

template <>
struct ToString<SolutionNode>;

template <>
struct FromString<SolutionNode>;

using SolMapEntry = std::pair<ProblemKey, SolutionNodeVec>;

template <>
struct ToString<SolMapEntry>;

template <>
struct FromString<SolMapEntry>;

inline bool ProbSolCmp(const SolMapEntry& lhs, const SolMapEntry& rhs)
{
    // 1st, compare the node type and schemes (from value)
    const auto& last_node_lhs = lhs.second.back();
    const auto& last_node_rhs = rhs.second.back();

    if(last_node_lhs.sol_node_type == last_node_rhs.sol_node_type)
    {
        ComputeScheme scheme_lhs = last_node_lhs.using_scheme;
        ComputeScheme scheme_rhs = last_node_rhs.using_scheme;

        // 2nd, compare the prob_token (from key)
        if(scheme_lhs == scheme_rhs)
            return lhs.first < rhs.first;

        return scheme_lhs < scheme_rhs;
    }
    return last_node_lhs.sol_node_type < last_node_rhs.sol_node_type;
}

class solution_map
{
    friend class SolutionMapConverter;

    bool assume_latest_ver = true;

    int        self_version = 0;
    ProbSolMap primary_sol_map;
    ProbSolMap temp_working_map;

    ROCFFT_EXPORT solution_map();

private:
    // a private function version of add_solution which can be called only by ctor.
    // That is, the implementation of solution_map() generated by solution-shipping.py
    size_t add_solution_private(const ProblemKey& probKey, const SolutionNode& solution);

    // check if two solution nodes have identical semantic
    bool SolutionNodesAreEqual(const SolutionNode& lhs,
                               const SolutionNode& rhs,
                               const std::string&  arch,
                               bool                primary_map);

    // remove an entire solution tree linked to the node
    bool remove_solution_tree(SolutionNodeVec& nodeVec, SolutionNode& node, size_t pos);

    // remove a single solution node, and update its parent's child_vector
    bool remove_solution_node(SolutionNodeVec& nodeVec, SolutionNode& node, size_t pos);

    void generate_link_info();

public:
    // the latest version number of solution-map's format
    static const int VERSION;

    // a default kernel-token for any built-in kernel
    static const char* KERNEL_TOKEN_BUILTIN_KERNEL;

    // a default leafnode-token for leafnodes linking to built-in kernels
    static const char* LEAFNODE_TOKEN_BUILTIN_KERNEL;

    solution_map(const solution_map&) = delete;

    solution_map& operator=(const solution_map&) = delete;

    static solution_map& get_solution_map()
    {
        static solution_map sol_map;
        return sol_map;
    }

    ~solution_map() = default;

    void setup(const std::string& arch_name);

    bool
        has_solution_node(const ProblemKey& probKey, size_t option_id = 0, bool primary_map = true);

    SolutionNode&
        get_solution_node(const ProblemKey& probKey, size_t option_id = 0, bool primary_map = true);

    FMKey& get_solution_kernel(const ProblemKey& probKey,
                               size_t            option_id   = 0,
                               bool              primary_map = true);

    // setup a solution of a problem and insert to the map, should be called by a benchmarker
    size_t add_solution(const ProblemKey&               probKey,
                        TreeNode*                       currentNode,
                        const std::vector<SolutionPtr>& children,
                        bool                            isRootProb,
                        bool                            check_dup,
                        bool                            primary_map = true);

    // add a solution of a problem to the map, should be called by a benchmarker
    size_t add_solution(const ProblemKey& probKey,
                        const FMKey&      kernel_key,
                        bool              check_dup,
                        bool              primary_map = true);

    // directly insert a solution of a problem to the map, should be called by a benchmarker
    // NB:
    // (The following is for ComputeSchemeIsAProblem):
    //   For the root-prob, we want the root-solution to be always at option 0, and make
    //   it "Exclusively" used by that root-problem. So we ALWAYS put the root-solution
    //   at the beginning of the solution vector and don't need to check_dup.
    //
    //   Furthermore, since this means the option-0 is reserved for root-prob only, so when doing
    //   check_dup, we start comparing from the second element.
    size_t add_solution(const ProblemKey&   probKey,
                        const SolutionNode& solution,
                        bool                isRootProb,
                        bool                check_dup,
                        bool                primary_map = true);

    // get all child nodes of specified type of a tree-node, put in a set
    bool get_typed_nodes_of_tree(const SolutionNode&     root,
                                 SolutionNodeType        type,
                                 std::set<SolutionNode>& ret);

    // get solution-nodes of SOL_KERNEL_ONLY, getUsedOnly flag filters the kernels
    // that are not used (replaced by newly-tuned), default false, return all kernels
    bool get_all_kernels(std::vector<SolutionNode>& sol_kernels, bool getUsedOnly = false);

    // parse the format version of the input file
    bool get_solution_map_version(const fs::path& sol_map_in_path);

    // read the map from input stream
    bool read_solution_map_data(const fs::path& sol_map_in_path, bool primary_map = true);

    // write the map to output stream,
    // sort = output the entries in order, which is helpful when comparing before/after merging
    bool write_solution_map_data(const fs::path& sol_map_out_path,
                                 bool            sort        = true,
                                 bool            primary_map = true);

    // merge solutions from src_file to primary map
    bool merge_solutions_from_file(const fs::path&                src_file,
                                   const std::vector<ProblemKey>& root_probs);
};

class SolutionMapConverter
{
private:
    // ver.0 -> ver.1: remove unused and invalid/incorrect kernels
    //               : caused by using un-supported half_lds in sbrc/sbcr kernels
    bool remove_invalid_half_lds();

    bool remove_callback_nodes();

    void remove_all_marked(ProbSolMap& target);

public:
    SolutionMapConverter()  = default;
    ~SolutionMapConverter() = default;

    bool VersionCheckAndConvert(const std::string& in_map_path, const std::string& out_map_path);
};

#endif // SOLUTION_MAP_H
