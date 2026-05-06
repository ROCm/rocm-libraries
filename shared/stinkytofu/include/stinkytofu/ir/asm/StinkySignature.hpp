/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
/***************************************
 * Enum for Signature Value Kind
 ***************************************/
enum class SignatureValueKind : int { SIG_GLOBALBUFFER = 0, SIG_VALUE = 1 };

/***************************************
 * Helper functions
 ***************************************/
std::string isaVersionToGfx(const std::array<int, 3>& isa);
std::string toHex(uint32_t num);
std::string fieldDesc(const std::string& fieldName, int value, int bits = 0);

/***************************************
 * BitfieldUnion - Base for SRD Values
 ***************************************/
struct BitfieldUnion {
    virtual ~BitfieldUnion() = default;

    virtual std::string toString() const {
        return "0x" + toHex(value);
    }
    virtual int getValue() const {
        return value;
    }
    virtual std::string fieldsDesc() const = 0;
    virtual std::string desc() const = 0;

   protected:
    uint32_t value = 0;
};

/***************************************
 * SRD Upper Value Structures
 ***************************************/

// GFX9 (e.g., gfx90a, gfx942)
union SrdUpperFields9XX {
    struct {
        uint32_t dst_sel_x : 3;
        uint32_t dst_sel_y : 3;
        uint32_t dst_sel_z : 3;
        uint32_t dst_sel_w : 3;
        uint32_t num_format : 3;
        uint32_t data_format : 4;
        uint32_t user_vm_enable : 1;
        uint32_t user_vm_mode : 1;
        uint32_t index_stride : 2;
        uint32_t add_tid_enable : 1;
        uint32_t _unusedA : 3;
        uint32_t nv : 1;
        uint32_t _unusedB : 2;
        uint32_t type : 2;
    };
    uint32_t value;

    SrdUpperFields9XX() : value(0) {}
};

struct SrdUpperValue9XX : public BitfieldUnion {
    SrdUpperFields9XX fields;

    SrdUpperValue9XX() {
        value = 0;
    }

    static SrdUpperValue9XX staticInit();
    std::string fieldsDesc() const override;
    std::string desc() const override;
};

// GFX10 (e.g., gfx1030)
union SrdUpperFields10XX {
    struct {
        uint32_t dst_sel_x : 3;
        uint32_t dst_sel_y : 3;
        uint32_t dst_sel_z : 3;
        uint32_t dst_sel_w : 3;
        uint32_t format : 7;
        uint32_t _unusedA : 2;
        uint32_t index_stride : 2;
        uint32_t add_tid_enable : 1;
        uint32_t resource_level : 1;
        uint32_t _unusedB : 1;
        uint32_t LLC_noalloc : 2;
        uint32_t oob_select : 2;
        uint32_t type : 2;
    };
    uint32_t value;

    SrdUpperFields10XX() : value(0) {}
};

struct SrdUpperValue10XX : public BitfieldUnion {
    SrdUpperFields10XX fields;

    SrdUpperValue10XX() {
        value = 0;
    }

    static SrdUpperValue10XX staticInit();
    std::string fieldsDesc() const override;
    std::string desc() const override;
};

// GFX11 (e.g., gfx1100)
union SrdUpperFields11XX {
    struct {
        uint32_t dst_sel_x : 3;
        uint32_t dst_sel_y : 3;
        uint32_t dst_sel_z : 3;
        uint32_t dst_sel_w : 3;
        uint32_t format : 7;
        uint32_t _unusedA : 2;
        uint32_t index_stride : 2;
        uint32_t add_tid_enable : 1;
        uint32_t resource_level : 1;
        uint32_t _unusedB : 1;
        uint32_t LLC_noalloc : 2;
        uint32_t oob_select : 2;
        uint32_t type : 2;
    };
    uint32_t value;

    SrdUpperFields11XX() : value(0) {}
};

struct SrdUpperValue11XX : public BitfieldUnion {
    SrdUpperFields11XX fields;

    SrdUpperValue11XX() {
        value = 0;
    }

    static SrdUpperValue11XX staticInit();
    std::string fieldsDesc() const override;
    std::string desc() const override;
};

// GFX12 (e.g., gfx1200)
union SrdUpperFields12XX {
    struct {
        uint32_t dst_sel_x : 3;
        uint32_t dst_sel_y : 3;
        uint32_t dst_sel_z : 3;
        uint32_t dst_sel_w : 3;
        uint32_t format : 7;
        uint32_t _unusedA : 2;
        uint32_t index_stride : 2;
        uint32_t add_tid_enable : 1;
        uint32_t resource_level : 1;
        uint32_t _unusedB : 3;
        uint32_t oob_select : 2;
        uint32_t type : 2;
    };
    uint32_t value;

    SrdUpperFields12XX() : value(0) {}
};

struct SrdUpperValue12XX : public BitfieldUnion {
    SrdUpperFields12XX fields;

    SrdUpperValue12XX() {
        value = 0;
    }

    static SrdUpperValue12XX staticInit();
    std::string fieldsDesc() const override;
    std::string desc() const override;
};

// GFX1250 specific
union SrdUpperFields125X {
    struct {
        uint32_t num_records_upper : 6;
        uint32_t reserved : 6;
        uint32_t stride : 14;
        uint32_t stride_scale : 2;
        uint32_t swizzle_enable : 1;
        uint32_t oob_select : 1;
        uint32_t type : 2;
    };
    uint32_t value;

    SrdUpperFields125X() : value(0) {}
};

struct SrdUpperValue125X : public BitfieldUnion {
    SrdUpperFields125X fields;

    SrdUpperValue125X() {
        value = 0;
    }

    static SrdUpperValue125X staticInit();
    std::string fieldsDesc() const override;
    std::string desc() const override;
};

// Factory function to create appropriate SRD value based on ISA version
std::shared_ptr<BitfieldUnion> createSrdUpperValue(const std::array<int, 3>& isa);

/***************************************
 * Signature Argument
 ***************************************/
struct SignatureArgument {
    static const std::unordered_map<std::string, int> ValueTypeSizeDict;

    SignatureValueKind valueKind;
    std::string valueType;
    std::string name;
    int offset;
    int size;
    std::string addrSpaceQual;

    SignatureArgument(int offset, const std::string& name, SignatureValueKind valueKind,
                      const std::string& valueType, const std::string& addrSpaceQual = "");

    std::string toString() const;

   private:
    int valueToSize(SignatureValueKind valueKind, const std::string& valueType) const;
    std::string valueKindToStr() const;
};

/***************************************
 * Signature Kernel Descriptor
 ***************************************/
struct SignatureKernelDescriptor {
    std::string kernelName;
    int totalVgprs;
    int totalAgprs;
    int totalSgprs;
    int originalTotalVgprs;
    int accumOffset;
    int groupSegSize;
    std::array<int, 3> sgprWorkGroup;
    int vgprWorkItem;
    int numSgprPreload;
    std::array<int, 3> isaVersion;
    int wavefrontSize;

    // Optimization config fields
    std::array<int, 2> threadTile = {0, 0};
    std::array<int, 2> subGroup = {0, 0};
    std::array<int, 2> waveGroup = {0, 0};
    int vectorWidthA = 0;
    int vectorWidthB = 0;
    int globalReadVectorWidthA = 0;
    int globalReadVectorWidthB = 0;
    bool directToLdsA = false;
    bool directToLdsB = false;
    int useSgprForGRO = 0;

    // Pass-through .amdhsa_* directives that the structured fields above do
    // not model (e.g. .amdhsa_user_sgpr_count, .amdhsa_fp16_overflow,
    // .amdhsa_inst_pref_size, .amdhsa_user_sgpr_kernarg_preload_*).
    // RawAsmParser appends raw text here when round-tripping; toString()
    // emits each entry verbatim before .end_amdhsa_kernel. Empty by default
    // so rocisa-driven flows (which populate the structured fields directly)
    // are unaffected.
    std::vector<std::string> extraKernelDirectives;

    SignatureKernelDescriptor(const std::string& name, const std::array<int, 3>& isaVersion,
                              int groupSegSize, const std::array<int, 3>& sgprWorkGroup,
                              int vgprWorkItem, int wavefrontSize = 64, int totalVgprs = 0,
                              int totalAgprs = 0, int totalSgprs = 0, int numSgprPreload = 0);

    void setGprs(int totalVgprs, int totalAgprs, int totalSgprs);
    void setOptimizationConfig(const std::array<int, 2>& tt, const std::array<int, 2>& sg,
                               const std::array<int, 2>& wg, int vwA, int vwB, int glvwA, int glvwB,
                               bool d2lA, bool d2lB, int useSgprForGRO);
    int getNextFreeVgpr() const {
        return totalVgprs;
    }
    int getNextFreeSgpr() const {
        return totalSgprs;
    }
    std::string toString() const;
};

/***************************************
 * Signature Code Metadata
 ***************************************/
struct SignatureCodeMeta {
    std::string kernelName;
    int kernArgsVersion;
    int groupSegSize;
    int flatWgSize;
    std::string codeObjectVersion;
    int totalVgprs;
    int totalSgprs;
    int wavefrontSize;
    int offset;
    std::vector<SignatureArgument> argList;

    SignatureCodeMeta(const std::string& name, int kernArgsVersion, int groupSegSize,
                      int flatWgSize, int wavefrontSize, const std::string& codeObjectVersion,
                      int totalVgprs = 0, int totalSgprs = 0);

    void setGprs(int totalVgprs, int totalSgprs);
    void addArg(const std::string& name, SignatureValueKind kind, const std::string& type,
                const std::string& addrSpaceQual = "");

    std::string toString() const;
};

/***************************************
 * Signature Base (Main Class)
 ***************************************/
struct STINKYTOFU_EXPORT SignatureBase {
    SignatureKernelDescriptor kernelDescriptor;
    SignatureCodeMeta codeMeta;

    SignatureBase(const std::string& kernelName, const std::array<int, 3>& isaVersion,
                  int kernArgsVersion, const std::string& codeObjectVersion, int groupSegmentSize,
                  const std::array<int, 3>& sgprWorkGroup, int vgprWorkItem, int flatWorkGroupSize,
                  int wavefrontSize = 64, int totalVgprs = 0, int totalAgprs = 0,
                  int totalSgprs = 0, int numSgprPreload = 0);

    void setGprs(int totalVgprs, int totalAgprs, int totalSgprs);
    void setOptimizationConfig(const std::array<int, 2>& tt, const std::array<int, 2>& sg,
                               const std::array<int, 2>& wg, int vwA, int vwB, int glvwA, int glvwB,
                               bool d2lA, bool d2lB, int useSgprForGRO);
    void addArg(const std::string& name, SignatureValueKind kind, const std::string& type,
                const std::string& addrSpaceQual = "");

    void addDescriptionTopic(const std::string& text);
    void addDescriptionBlock(const std::string& text);
    void addDescription(const std::string& text);
    void clearDescription();

    int getNextFreeVgpr() const {
        return kernelDescriptor.getNextFreeVgpr();
    }
    int getNextFreeSgpr() const {
        return kernelDescriptor.getNextFreeSgpr();
    }

    std::string toString() const;

    // Public utility methods for formatting
    static std::string block3Line(const std::string& text);

   private:
    std::string descriptionTopic;
    std::vector<std::string> descriptionList;

    static std::string block(const std::string& text);
    static std::string slash(const std::string& text);
};

// Forward declaration for StinkyAsmModule
class StinkyAsmModule;

/**
 * IRListModule is an alias for StinkyAsmModule
 * This provides compatibility with rocisa's KernelBody API
 */
using IRListModule = StinkyAsmModule;

/**
 * @brief KernelBody - Complete kernel container combining metadata and instructions
 *
 * This class combines a SignatureBase (metadata) with an IRListModule (instructions)
 * to create a complete, self-contained kernel representation. It manages the lifecycle
 * and coordination between metadata and instruction body.
 *
 * Features:
 * - Automatic metadata + instruction concatenation
 * - Unified register count management
 * - Clean separation of concerns (metadata vs instructions)
 * - Direct rocisa.code.KernelBody API compatibility
 *
 * Example usage:
 * @code
 * // Create kernel
 * auto kernel = std::make_shared<KernelBody>("myKernel");
 *
 * // Add signature
 * auto sig = std::make_shared<SignatureBase>(...);
 * sig->addArg(...);
 * kernel->addSignature(sig);
 *
 * // Add instruction body
 * auto irModule = std::make_shared<IRListModule>(...);
 * // ... add instructions ...
 * kernel->addBody(irModule);
 *
 * // Set final register counts
 * kernel->setGprs(128, 64, 48);
 *
 * // Generate complete kernel
 * std::string completeKernel = kernel->toString();
 * @endcode
 */
class KernelBody {
   public:
    /**
     * @brief Construct a new KernelBody
     * @param name Kernel name
     */
    KernelBody(const std::string& name);

    /**
     * @brief Add signature (metadata) to the kernel
     * @param signature SignatureBase instance containing metadata
     */
    void addSignature(const std::shared_ptr<SignatureBase>& signature);

    /**
     * @brief Add instruction body to the kernel
     * @param body IRListModule containing instructions
     */
    void addBody(const std::shared_ptr<IRListModule>& body);

    /**
     * @brief Set register counts for both signature and tracking
     * @param totalVgprs Total vector GPRs used
     * @param totalAgprs Total accumulator GPRs used
     * @param totalSgprs Total scalar GPRs used
     *
     * This updates both the signature metadata and internal tracking.
     */
    void setGprs(int totalVgprs, int totalAgprs, int totalSgprs);

    /**
     * @brief Get the next free VGPR index
     * @return Next available VGPR index
     */
    int getNextFreeVgpr() const;

    /**
     * @brief Get the next free SGPR index
     * @return Next available SGPR index
     */
    int getNextFreeSgpr() const;

    /**
     * @brief Get the kernel name
     * @return Kernel name
     */
    std::string getName() const;

    /**
     * @brief Get the signature (metadata)
     * @return Shared pointer to SignatureBase
     */
    std::shared_ptr<SignatureBase> getSignature() const;

    /**
     * @brief Get the instruction body
     * @return Shared pointer to IRListModule
     */
    std::shared_ptr<IRListModule> getBody() const;

    /**
     * @brief Generate complete kernel assembly
     *
     * Combines signature metadata with instruction body:
     * - Signature metadata (target, kernel descriptor, code object metadata)
     * - Instruction assembly (generated via emitter)
     *
     * @param emitComments Whether to emit comments in instruction assembly
     * @param emitCycleInfo Whether to emit cycle info in instruction assembly
     * @return Complete kernel assembly string
     */
    std::string toString(bool emitComments = true, bool emitCycleInfo = false) const;

   private:
    std::string name;
    std::shared_ptr<SignatureBase> signature;
    std::shared_ptr<IRListModule> body;
    int totalVgprs;
    int totalAgprs;
    int totalSgprs;
};

}  // namespace stinkytofu
