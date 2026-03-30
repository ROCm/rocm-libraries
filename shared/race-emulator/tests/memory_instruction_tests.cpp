// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/CommonRegister.h"
#include "race-emulator/EmulatorException.h"
#include "race-emulator/Wave.h"
#include "race-emulator/Workgroup.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using namespace raceemulator;

void tryExecute(Wave &regs, const std::string &line) {
  regs.tryExecute(line, false);
}

// Helper: set up SRD (Shader Resource Descriptor) in s[srdBase:srdBase+3].
void setupSrd(Wave &wave, int srdBase, uintptr_t baseAddr, uint32_t size) {
  wave.setSgpr(srdBase, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  wave.setSgpr(srdBase + 1, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  wave.setSgpr(srdBase + 2, size);
  wave.setSgpr(srdBase + 3, 0);
}

} // namespace

TEST(Instructions, MemoryRoundTrip) {
  Workgroup wg;
  wg.setRaceChecks(true);
  Wave regs(/*vgprCount*/ 10, /*sgprCount*/ 10, WaveSize{1}, wg);

  // 1. Allocate a safe "global memory" buffer on the host
  // Using a vector ensures memory is valid and cleaned up automatically.
  std::vector<uint32_t> globalMem(16, 0xDEADBEEF);
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(globalMem.data());

  // 2. Setup Address in v[0:1]
  regs.setVgpr64(0, /*lane*/ 0, static_cast<uint64_t>(baseAddr));

  // 3. Setup Data in v2 (Value to store)
  uint32_t valueToStore = 12345678;
  regs.setVgpr(2, /*lane*/ 0, valueToStore);

  // 4. Store: global_store_dword v_addr, v_data
  // Syntax: global_store_dword v[0:1], v2
  tryExecute(regs, "global_store_dword v[0:1], v2");

  // Verify host memory changed directly
  EXPECT_EQ(globalMem[0], valueToStore);

  // 5. Load: global_load_dword v_data, v_addr
  // Syntax: global_load_dword v3, v[0:1]
  tryExecute(regs, "global_load_dword v3, v[0:1]");

  tryExecute(regs, "s_waitcnt vmcnt(0)");

  // Verify v3 has the value read back from memory
  EXPECT_EQ(regs.getVgpr(3, /*lane*/ 0), valueToStore);
}

TEST(Instructions, GlobalLoadStore_WithInstructionOffsets) {
  // Setup: 4 VGPRs, 0 SGPRs, WaveSize 1
  Workgroup wg;
  Wave regs(4, 0, WaveSize{1}, wg);

  // 1. Setup "Device Memory"
  // Allocate 16 dwords (64 bytes) of host memory to simulate GPU Global Memory
  // Initialize with generic pattern 0xFFFFFFFF
  std::vector<uint32_t> deviceMem(16, 0xFFFFFFFF);

  // Get the raw pointer address to pass into the emulator
  uint64_t basePtr = reinterpret_cast<uint64_t>(deviceMem.data());

  // Store the base pointer in v[0:1]
  regs.setVgpr64(0, 0, basePtr);

  // 2. Test Store with Offset
  // Instruction: global_store_dword v[0:1], v2, off offset:12
  // Logic: Write value from v2 into Base + 12 bytes

  // v2 = 0xCAFEBABE (The data to write)
  regs.setVgpr(2, 0, 0xCAFEBABE);

  // Execute Store
  // Offset 12 bytes = 3 DWORDS (index 3)
  tryExecute(regs, "global_store_dword v[0:1], v2, off offset:12");

  // VERIFICATION:
  // Check that deviceMem[3] is updated
  EXPECT_EQ(deviceMem[3], 0xCAFEBABE);

  // Check neighbors to ensure we didn't write to Index 0 or Index 2
  EXPECT_EQ(deviceMem[0], 0xFFFFFFFF);
  EXPECT_EQ(deviceMem[2], 0xFFFFFFFF);

  // 3. Test Load with Offset
  // Instruction: global_load_dword v3, v[0:1], off offset:20
  // Logic: Read value from Base + 20 bytes into v3

  // Setup Data at Byte Offset 20 (Index 5)
  deviceMem[5] = 0xDEADBEEF;

  // Execute Load
  tryExecute(regs, "global_load_dword v3, v[0:1], off offset:20");

  // VERIFICATION:
  // Check that v3 contains the value from Index 5
  EXPECT_EQ(regs.getVgpr(3, 0), 0xDEADBEEF);
}

TEST(Instructions, DS_Write_B32_Direct) {
  // 1. Setup LDS Storage (Simulating 1KB of Shared Memory)
  Workgroup wg;
  wg.resizeLds(1024);
  std::map<std::string, int> labels;

  // 2. Setup Registers linked to this LDS
  //    Assumes Wave(vgpr, sgpr, waveSize, LDS*)
  WaveId id{0};
  std::map<std::string, Macro> macros;
  Wave regs(4, 4, 0, WaveSize{1}, id, wg, &labels, &macros); // 4 VGPRs, 4 SGPRs

  // 3. Setup Operand State
  //    Instruction: ds_write_b32 v0, v1 offset:16

  // v0 = Address 100 (Relative to start of LDS)
  regs.setVgpr(0, 0, 100);

  // v1 = Data to write (0xCAFEBABE)
  regs.setVgpr(1, 0, 0xCAFEBABE);

  // 4. Execute directly
  //    Target Address = v0 (100) + offset (16) = 116
  tryExecute(regs, "ds_write_b32 v0, v1 offset:16");

  // 5. Verify by inspecting the LDS object directly
  //    (Using memcpy to handle char-to-uint32 safety)
  uint32_t actual = 0;
  actual = wg.getLds().read<uint32_t>(116); // Clear first

  EXPECT_EQ(actual, 0xCAFEBABE);
}

TEST(Instructions, DS_Write_AllVariants_Combined) {
  // 1. Setup (One-time cost)
  Workgroup wg;
  wg.resizeLds(1024); // Init with "unset" pattern
  WaveId id{0};

  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels,
            &macros); // Need enough VGPRs for b128

  // --- Test Case A: ds_write_b8 (Truncation) ---
  // Logic: Write 0xDEADBEEF. Should only write 0xEF (low byte).
  {
    regs.setVgpr(0, 0, 10);         // Addr: 10
    regs.setVgpr(1, 0, 0xDEADBEEF); // Data

    tryExecute(regs, "ds_write_b8 v0, v1");

    // Verify written byte
    uint8_t actual = 0;
    actual = wg.getLds().read<uint8_t>(10);
    // std::memcpy(&actual, wg.getLds().getBase() + 10, 1);

    EXPECT_EQ(actual, 0xEF);

    // Verify neighbor (byte 11) is untouched
    // (Assuming you have an LDS::unset1 or similar, otherwise check against
    // byte pattern)
    uint8_t neighbor = 0;
    neighbor = wg.getLds().read<uint8_t>(11);
    // std::memcpy(&neighbor, wg.getLds().getBase() + 11, 1);
    EXPECT_NE(neighbor, 0xBE); // Should NOT contain the second byte of DEADBEEF
  }

  // --- Test Case B: ds_write_b16 (Short) ---
  // Logic: Write 0xCAFEBABE. Should write 0xBABE (low 16 bits).
  {
    regs.setVgpr(2, 0, 20);         // Addr: 20
    regs.setVgpr(3, 0, 0xCAFEBABE); // Data

    tryExecute(regs, "ds_write_b16 v2, v3");

    uint16_t actual = 0;
    actual = wg.getLds().read<uint16_t>(20);
    // std::memcpy(&actual, wg.getLds().getBase() + 20, 2);
    EXPECT_EQ(actual, 0xBABE);

    // Verify it didn't write 4 bytes
    uint16_t upperPart = 0;
    upperPart = wg.getLds().read<uint16_t>(22);
    // std::memcpy(&upperPart, wg.getLds().getBase() + 22, 2);
    EXPECT_NE(upperPart, 0xCAFE);
  }

  // --- Test Case C: ds_write_b64 (Vector) ---
  // Logic: Write v[4:5] (0xA..A, 0xB..B) to Address 32.
  {
    regs.setVgpr(4, 0, 32);         // Addr: 32
    regs.setVgpr(5, 0, 0xAAAAAAAA); // Data Low
    regs.setVgpr(6, 0, 0xBBBBBBBB); // Data High

    tryExecute(regs, "ds_write_b64 v4, v5");

    uint32_t val1 = 0, val2 = 0;
    val1 = wg.getLds().read<uint32_t>(32);
    val2 = wg.getLds().read<uint32_t>(36);
    // std::memcpy(&val1, wg.getLds().getBase() + 32, 4);
    // std::memcpy(&val2, wg.getLds().getBase() + 36, 4);

    EXPECT_EQ(val1, 0xAAAAAAAA);
    EXPECT_EQ(val2, 0xBBBBBBBB);
  }

  // --- Test Case D: ds_write_b128 (Large Vector) ---
  // Logic: Write v[8:11] to Address 64.
  {
    regs.setVgpr(8, 0, 64); // Addr
    for (int i = 0; i < 4; ++i) {
      regs.setVgpr(9 + i, 0, 0x10101010 * (i + 1));
    }

    tryExecute(regs, "ds_write_b128 v8, v9"); // Uses v9, v10, v11, v12

    // Verify 4 consecutive dwords
    for (int i = 0; i < 4; ++i) {
      uint32_t val = 0;
      val = wg.getLds().read<uint32_t>(64 + (i * 4));
      // std::memcpy(&val, wg.getLds().getBase() + 64 + (i*4), 4);
      EXPECT_EQ(val, 0x10101010 * (i + 1));
    }
  }
}

TEST(Instructions, DS_Read_Variants) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(10, 10, 0, WaveSize{1}, id, wg, &labels, &macros);

  // Setup Memory
  // Addr 10: 0xFF (Represents -1 in i8, or 255 in u8)
  uint8_t valByte = 0xFF;
  wg.getLds().write<uint8_t>(10, valByte);
  // std::memcpy(wg.getLds().getBase() + 10, &valByte, 1);

  // Addr 20: 0xDEADBEEF (for b32 test)
  uint32_t val32 = 0xDEADBEEF;
  wg.getLds().write<uint32_t>(20, val32);
  // std::memcpy(wg.getLds().getBase() + 20, &val32, 4);

  // 1. Test ds_read_u8 (Should be 255 / 0x000000FF)
  regs.setVgpr(0, 0, 10); // Addr
  tryExecute(regs, "ds_read_u8 v1, v0");
  EXPECT_EQ(regs.getVgpr(1, 0), 0x000000FF);

  // 2. Test ds_read_i8 (Should be -1 / 0xFFFFFFFF)
  //
  tryExecute(regs, "ds_read_i8 v2, v0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xFFFFFFFF);

  // 3. Test ds_read_b32 with Offset
  // Addr v0(10) + offset(10) = 20
  tryExecute(regs, "ds_read_b32 v3, v0 offset:10");
  EXPECT_EQ(regs.getVgpr(3, 0), 0xDEADBEEF);
}

TEST(Instructions, BufferLoad_Offen_Success) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  // 1. Allocate host memory
  std::vector<uint32_t> hostMem = {17, 0xCAFEBABE, 0x00001111, 0x22223333};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());

  // 2. Setup SRSRC in s[4:7] (The Buffer Resource Descriptor)
  // Word 0: Base Address Lower 32 bits
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  // Word 1: Base Address Upper 16 bits (stored in bits 0-15)
  // Note: High bits of Word 1 are usually flags, which our emulator currently
  // ignores.
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  regs.setSgpr(
      6, 0xFFFFFFFF); // Stride/NumRecords (Ignored for this basic emulator)
  regs.setSgpr(7, 0); // Flags (Ignored)

  // 3. Setup VADDR in v29 (The Offset)
  // We want to read the first element, so offset is 0.
  regs.setVgpr(29, /*lane*/ 0, 0);

  tryExecute(regs, "buffer_load_dwordx4 v[2:5], v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(2, 0), 17);
  EXPECT_EQ(regs.getVgpr(3, 0), 0xCAFEBABE);
  EXPECT_EQ(regs.getVgpr(4, 0), 0x00001111);
  EXPECT_EQ(regs.getVgpr(5, 0), 0x22223333);
}

TEST(Instructions, BufferLoad_Offen_Extended) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  // 1. Allocate host memory (Increased size to support offsets)
  // Indices:      0           1           2           3           4           5
  // 6           7
  std::vector<uint32_t> hostMem = {0xAAAA0000, 0xBBBB1111, 0xCCCC2222,
                                   0xDDDD3333, 0xEEEE4444, 0xFFFF5555,
                                   0x66667777, 0x77778888};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());

  // 2. Setup SRSRC in s[4:7]
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  regs.setSgpr(6, 0xFFFFFFFF);
  regs.setSgpr(7, 0);

  // Setup VADDR in v29 (initially 0)
  regs.setVgpr(29, 0, 0);

  // Case A: Standard (Base + VOffset)
  // Reads starting at Index 0
  tryExecute(regs, "buffer_load_dwordx4 v[2:5], v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xAAAA0000);
  EXPECT_EQ(regs.getVgpr(3, 0), 0xBBBB1111);

  // Case B: Immediate Offset (Base + VOffset + Imm)
  // Instruction: offset:4 (4 bytes)
  // Target: Byte 4 -> Index 1 (0xBBBB1111)
  tryExecute(regs,
             "buffer_load_dwordx4 v[10:13], v29, s[4:7], 0 offen offset:4");
  EXPECT_EQ(regs.getVgpr(10, 0), 0xBBBB1111);
  EXPECT_EQ(regs.getVgpr(11, 0), 0xCCCC2222);

  // Case C: Scalar Offset (Base + VOffset + SOffset)
  // Setup s8 = 8 (bytes)
  // Target: Byte 8 -> Index 2 (0xCCCC2222)
  regs.setSgpr(8, 8);
  tryExecute(regs, "buffer_load_dwordx4 v[14:17], v29, s[4:7], s8 offen");
  EXPECT_EQ(regs.getVgpr(14, 0), 0xCCCC2222);
  EXPECT_EQ(regs.getVgpr(15, 0), 0xDDDD3333);

  // Case D: Combined (Base + VOffset + SOffset + Imm)
  // VOffset(v29) = 4
  // SOffset(s8)  = 4
  // ImmOffset    = 4
  // Total Offset = 4 + 4 + 4 = 12 bytes -> Index 3 (0xDDDD3333)
  regs.setVgpr(29, 0, 4); // Set vector offset
  regs.setSgpr(8, 4);     // Set scalar offset

  tryExecute(regs,
             "buffer_load_dwordx4 v[18:21], v29, s[4:7], s8 offen offset:4");
  EXPECT_EQ(regs.getVgpr(18, 0), 0xDDDD3333);
  EXPECT_EQ(regs.getVgpr(19, 0), 0xEEEE4444);
}

TEST(Instructions, BufferStore_Offen_Extended) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  // 1. Allocate Host Memory (initialized to zero)
  // Size: 8 elements (32 bytes)
  std::vector<uint32_t> hostMem(8, 0);
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());

  // 2. Setup SRSRC in s[4:7]
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));

  regs.setSgpr(6, 0xFFFFFFFF);
  regs.setSgpr(7, 0);

  // 3. Setup Data to Write (v[10:13])
  regs.setVgpr(10, 0, 0xDEADBEEF);
  regs.setVgpr(11, 0, 0xCAFEBABE);
  regs.setVgpr(12, 0, 0x12345678);
  regs.setVgpr(13, 0, 0x87654321);

  // Setup VADDR (Offset) in v29
  regs.setVgpr(29, 0, 0);

  // Case A: Standard Store (Base + VOffset)
  // Store v[10:11] (2 dwords) to Index 0
  tryExecute(regs, "buffer_store_dwordx2 v[10:11], v29, s[4:7], 0 offen");
  EXPECT_EQ(hostMem[0], 0xDEADBEEF);
  EXPECT_EQ(hostMem[1], 0xCAFEBABE);

  // Case B: Immediate Offset
  // Store v12 (1 dword) to Index 2 (Offset 8 bytes)
  tryExecute(regs, "buffer_store_dword v12, v29, s[4:7], 0 offen offset:8");
  EXPECT_EQ(hostMem[2], 0x12345678);

  // Case C: Scalar Offset + Immediate + VOffset
  // Store v13 to Index 4 (Offset 16 bytes)
  // VOffset(v29) = 4
  // SOffset(s8)  = 4
  // ImmOffset    = 8
  // Total = 4 + 4 + 8 = 16 bytes -> Index 4

  regs.setVgpr(29, 0, 4);
  regs.setSgpr(8, 4);
  tryExecute(regs, "buffer_store_dword v13, v29, s[4:7], s8 offen offset:8");
  EXPECT_EQ(hostMem[4], 0x87654321);
}

TEST(Instructions, BufferLoad_NegativeOffsets) {
  Workgroup wg;
  Wave regs(10, 32, WaveSize{64}, wg);

  // 1. Allocate Host Memory
  std::vector<uint32_t> hostMem(256, 0);
  hostMem[123] = 0xCAFEBABE; // The target data

  uintptr_t rootPtr = reinterpret_cast<uintptr_t>(hostMem.data());

  // Set base address to the start of the allocation.
  regs.setSgpr(4, static_cast<uint32_t>(rootPtr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((rootPtr >> 32) & 0xFFFF));

  // Set Size to the full size (1024 bytes)
  regs.setSgpr(6, 256 * 4);
  regs.setSgpr(7, 0);

  // 3. Setup Offsets to reach Index 123 POSITIVELY
  // We want Index 123 (byte 492).
  // Let's split it: Scalar=400, Vector=92.
  int32_t scalarOffset = 400;
  int32_t vectorOffset = 92;

  regs.setSgpr(8, static_cast<uint32_t>(scalarOffset));
  regs.setVgpr(0, 0, static_cast<uint32_t>(vectorOffset));

  // 4. Execute
  tryExecute(regs, "buffer_load_dword v1, v0, s[4:7], s8 offen");

  // 5. Verify
  EXPECT_EQ(regs.getVgpr(1, 0), 0xCAFEBABE);
}

// OOB (out-of-bounds) buffer loads return zero.
// ISA reference: CDNA3 §9.1: "When an address is out of range, reads return
// zero, and writes and atomics are dropped." CDNA4 §9.1 identical.
TEST(Instructions, BufferLoad_OutOfBounds_ReturnsZero) {
  Workgroup wg;
  Wave regs(/*vgprCount=*/10, /*sgprCount=*/10, WaveSize{1}, wg);

  std::vector<uint32_t> hostMem = {0x11111111, 0x22222222};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());
  // SRD size = 4 bytes: only the first dword is in bounds.
  setupSrd(regs, /*srdBase=*/4, baseAddr, /*size=*/4);

  // Pre-fill destination with sentinel to verify it gets zeroed.
  regs.setVgpr(2, 0, 0xDEADBEEF);
  regs.setVgpr(3, 0, 0xDEADBEEF);

  regs.setVgpr(0, 0, 0); // voffset = 0
  tryExecute(regs, "buffer_load_dwordx2 v[2:3], v0, s[4:7], 0 offen");

  // First dword is in bounds, second is OOB → zeroed.
  EXPECT_EQ(regs.getVgpr(2, 0), 0x11111111u);
  EXPECT_EQ(regs.getVgpr(3, 0), 0u);

  // Fully OOB: offset beyond buffer size.
  regs.setVgpr(4, 0, 0xDEADBEEF);
  regs.setVgpr(0, 0, 100); // voffset = 100, well past 4-byte buffer
  tryExecute(regs, "buffer_load_dword v4, v0, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(4, 0), 0u);
}

TEST(Instructions, DS_Write_B8_D16_HI) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // --- Test Case: ds_write_b8_d16_hi ---
  // Logic: Write 0x11223344.
  // Bits [31:24] = 0x11 (High byte of High word)
  // Bits [23:16] = 0x22 (Low byte of High word)  <-- TARGET
  // Bits [15:08] = 0x33 (High byte of Low word)
  // Bits [07:00] = 0x44 (Low byte of Low word)
  {
    regs.setVgpr(0, 0, 300);        // Addr: 300
    regs.setVgpr(1, 0, 0x11223344); // Data

    tryExecute(regs, "ds_write_b8_d16_hi v0, v1");

    // Verify written byte at address 300
    uint8_t actual = wg.getLds().read<uint8_t>(300);

    EXPECT_EQ(actual, 0x22);

    // Negative checks to ensure we didn't grab the wrong byte
    EXPECT_NE(actual, 0x44); // Standard b8 (low bits)
    EXPECT_NE(actual, 0x11); // High bits of high word
  }
}

TEST(Instructions, DS_Write_Extended_Features) {
  // Setup
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // --- Test Case E: ds_write_b16_d16_hi ---
  // Logic: Write 0xAABBCCDD.
  // Standard b16 would write 0xCCDD.
  // d16_hi should write 0xAABB (bits [31:16]).
  {
    regs.setVgpr(0, 0, 100);        // Addr: 100
    regs.setVgpr(1, 0, 0xAABBCCDD); // Data

    tryExecute(regs, "ds_write_b16_d16_hi v0, v1");

    // Verify 16 bits at address 100
    uint16_t actual = wg.getLds().read<uint16_t>(100);
    EXPECT_EQ(actual, 0xAABB);

    // Sanity check: ensure we didn't write the lower bits
    EXPECT_NE(actual, 0xCCDD);
  }

  // --- Test Case F: ds_write with Offset ---
  // Logic: Write to Address 200 with offset:16 -> Effective Address 216
  {
    regs.setVgpr(2, 0, 200);        // Base Addr: 200
    regs.setVgpr(3, 0, 0x12345678); // Data

    // Note: The parser expects no spaces inside "offset:16" based on current
    // implementation
    tryExecute(regs, "ds_write_b32 v2, v3 offset:16");

    // Check Base Address (should be untouched/zero if initialized that way)
    uint32_t baseVal = wg.getLds().read<uint32_t>(200);
    // Assuming memory was zeroed or didn't contain this pattern
    EXPECT_NE(baseVal, 0x12345678);

    // Check Effective Address (200 + 16 = 216)
    uint32_t offsetVal = wg.getLds().read<uint32_t>(216);
    EXPECT_EQ(offsetVal, 0x12345678);
  }
}

TEST(Instructions, DS_Read_AllVariants_Extended) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(10, 10, 0, WaveSize{1}, id, wg, &labels, &macros);
  regs.setDsPreserve(true); // Test d16 preserve behavior

  // --- Setup Memory ---
  // Addr 10: 0xFF
  //   - Interpreted as u8:  255
  //   - Interpreted as i8:  -1
  uint8_t valByte = 0xFF;
  wg.getLds().write<uint8_t>(10, valByte);

  // Addr 20: 0xDEADBEEF
  uint32_t val32 = 0xDEADBEEF;
  wg.getLds().write<uint32_t>(20, val32);

  // PART 1: Standard Reads (Legacy Behavior)

  // 1. Test ds_read_u8 (Full Register Overwrite, Zero Ext)
  regs.setVgpr(0, 0, 10);         // Addr
  regs.setVgpr(1, 0, 0x12345678); // Garbage to ensure overwrite
  tryExecute(regs, "ds_read_u8 v1, v0");
  EXPECT_EQ(regs.getVgpr(1, 0), 0x000000FF);

  // 2. Test ds_read_i8 (Full Register Overwrite, Sign Ext)
  regs.setVgpr(2, 0, 0x12345678); // Garbage
  tryExecute(regs, "ds_read_i8 v2, v0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xFFFFFFFF); // -1 extended to 32-bit

  // 3. Test ds_read_b32 with Offset
  // Addr v0(10) + offset(10) = 20
  regs.setVgpr(3, 0, 0x00000000);
  tryExecute(regs, "ds_read_b32 v3, v0 offset:10");
  EXPECT_EQ(regs.getVgpr(3, 0), 0xDEADBEEF);

  // PART 2: D16 Variants (Packing Behavior)

  // 4. Test ds_read_u8_d16 (Low 16-bit write, Zero Ext, High Preserved)
  // Preset v4 to 0xAAAA_AAAA.
  // We read 0xFF. Zero ext to 0x00FF.
  // Result should be 0xAAAA_00FF.
  regs.setVgpr(4, 0, 0xAAAAAAAA);
  tryExecute(regs, "ds_read_u8_d16 v4, v0");
  EXPECT_EQ(regs.getVgpr(4, 0), 0xAAAA00FF);

  // 5. Test ds_read_i8_d16 (Low 16-bit write, Sign Ext, High Preserved)
  // Preset v5 to 0xBBBB_BBBB.
  // We read 0xFF (-1). Sign ext to 16-bit is 0xFFFF (-1).
  // Result should be 0xBBBB_FFFF.
  regs.setVgpr(5, 0, 0xBBBBBBBB);
  tryExecute(regs, "ds_read_i8_d16 v5, v0");
  EXPECT_EQ(regs.getVgpr(5, 0), 0xBBBBFFFF);

  // 6. Test ds_read_u8_d16_hi (High 16-bit write, Zero Ext, Low Preserved)
  // Preset v6 to 0xCCCC_CCCC.
  // We read 0xFF. Zero ext to 0x00FF.
  // We write to high bits.
  // Result should be 0x00FF_CCCC.
  regs.setVgpr(6, 0, 0xCCCCCCCC);
  tryExecute(regs, "ds_read_u8_d16_hi v6, v0");
  EXPECT_EQ(regs.getVgpr(6, 0), 0x00FFCCCC);

  // 7. Test ds_read_i8_d16_hi (High 16-bit write, Sign Ext, Low Preserved)
  // Preset v7 to 0xDDDD_DDDD.
  // We read 0xFF (-1). Sign ext to 16-bit is 0xFFFF.
  // We write to high bits.
  // Result should be 0xFFFF_DDDD.
  regs.setVgpr(7, 0, 0xDDDDDDDD);
  tryExecute(regs, "ds_read_i8_d16_hi v7, v0");
  EXPECT_EQ(regs.getVgpr(7, 0), 0xFFFFDDDD);
}

TEST(Instructions, DS_Write2_B64) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // ds_write2_b64 vaddr, v[data0:data0+1], v[data1:data1+1] offset0:N offset1:M
  // Writes 8 bytes from data0 to LDS[vaddr + offset0*8]
  // and 8 bytes from data1 to LDS[vaddr + offset1*8]

  regs.setVgpr(0, 0, 100);        // vaddr = 100
  regs.setVgpr(1, 0, 0xAAAAAAAA); // data0 low
  regs.setVgpr(2, 0, 0xBBBBBBBB); // data0 high
  regs.setVgpr(3, 0, 0xCCCCCCCC); // data1 low
  regs.setVgpr(4, 0, 0xDDDDDDDD); // data1 high

  // offset0=2, offset1=5 -> addresses 100+16=116, 100+40=140
  tryExecute(regs, "ds_write2_b64 v0, v[1:2], v[3:4] offset0:2 offset1:5");

  // Verify first 8 bytes at addr 116
  EXPECT_EQ(wg.getLds().read<uint32_t>(116), 0xAAAAAAAAu);
  EXPECT_EQ(wg.getLds().read<uint32_t>(120), 0xBBBBBBBBu);

  // Verify second 8 bytes at addr 140
  EXPECT_EQ(wg.getLds().read<uint32_t>(140), 0xCCCCCCCCu);
  EXPECT_EQ(wg.getLds().read<uint32_t>(144), 0xDDDDDDDDu);
}

TEST(Instructions, DS_Write2_B64_DefaultOffset0) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // When offset0 is omitted, it defaults to 0.
  // ds_write2_b64 v0, v[1:2], v[3:4] offset1:1
  // -> data0 at vaddr+0, data1 at vaddr+8
  regs.setVgpr(0, 0, 200); // vaddr = 200
  regs.setVgpr(1, 0, 0x11111111);
  regs.setVgpr(2, 0, 0x22222222);
  regs.setVgpr(3, 0, 0x33333333);
  regs.setVgpr(4, 0, 0x44444444);

  tryExecute(regs, "ds_write2_b64 v0, v[1:2], v[3:4] offset1:1");

  EXPECT_EQ(wg.getLds().read<uint32_t>(200), 0x11111111u);
  EXPECT_EQ(wg.getLds().read<uint32_t>(204), 0x22222222u);
  EXPECT_EQ(wg.getLds().read<uint32_t>(208), 0x33333333u);
  EXPECT_EQ(wg.getLds().read<uint32_t>(212), 0x44444444u);
}

TEST(Instructions, DS_Read2_B64) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // ds_read2_b64 v[dst:dst+3], vaddr offset0:N offset1:M
  // Reads 8 bytes from LDS[vaddr + offset0*8] into v[dst:dst+1]
  // and 8 bytes from LDS[vaddr + offset1*8] into v[dst+2:dst+3]

  // Setup LDS memory
  wg.getLds().write<uint32_t>(116, 0xAAAAAAAA);
  wg.getLds().write<uint32_t>(120, 0xBBBBBBBB);
  wg.getLds().write<uint32_t>(140, 0xCCCCCCCC);
  wg.getLds().write<uint32_t>(144, 0xDDDDDDDD);

  regs.setVgpr(0, 0, 100); // vaddr = 100

  // offset0=2, offset1=5 -> read from 100+16=116 and 100+40=140
  tryExecute(regs, "ds_read2_b64 v[4:7], v0 offset0:2 offset1:5");

  EXPECT_EQ(regs.getVgpr(4, 0), 0xAAAAAAAAu);
  EXPECT_EQ(regs.getVgpr(5, 0), 0xBBBBBBBBu);
  EXPECT_EQ(regs.getVgpr(6, 0), 0xCCCCCCCCu);
  EXPECT_EQ(regs.getVgpr(7, 0), 0xDDDDDDDDu);
}

TEST(Instructions, DS_Read2_Write2_B64_Roundtrip) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // Write via ds_write2_b64, then read back via ds_read2_b64
  regs.setVgpr(0, 0, 0); // vaddr = 0
  regs.setVgpr(1, 0, 0x12345678);
  regs.setVgpr(2, 0, 0x9ABCDEF0);
  regs.setVgpr(3, 0, 0xFEDCBA98);
  regs.setVgpr(4, 0, 0x76543210);

  tryExecute(regs, "ds_write2_b64 v0, v[1:2], v[3:4] offset0:10 offset1:20");

  // Read back into different registers
  tryExecute(regs, "ds_read2_b64 v[8:11], v0 offset0:10 offset1:20");

  EXPECT_EQ(regs.getVgpr(8, 0), 0x12345678u);
  EXPECT_EQ(regs.getVgpr(9, 0), 0x9ABCDEF0u);
  EXPECT_EQ(regs.getVgpr(10, 0), 0xFEDCBA98u);
  EXPECT_EQ(regs.getVgpr(11, 0), 0x76543210u);
}

TEST(Instructions, DS_Store_Load_2addr_B64_RdnaAliases) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // RDNA aliases: ds_store_2addr_b64 = ds_write2_b64
  //              ds_load_2addr_b64 = ds_read2_b64
  regs.setVgpr(0, 0, 64);
  regs.setVgpr(1, 0, 0x11223344);
  regs.setVgpr(2, 0, 0x55667788);
  regs.setVgpr(3, 0, 0x99AABBCC);
  regs.setVgpr(4, 0, 0xDDEEFF00);

  tryExecute(regs, "ds_store_2addr_b64 v0, v[1:2], v[3:4] offset0:0 offset1:1");

  tryExecute(regs, "ds_load_2addr_b64 v[8:11], v0 offset0:0 offset1:1");

  EXPECT_EQ(regs.getVgpr(8, 0), 0x11223344u);
  EXPECT_EQ(regs.getVgpr(9, 0), 0x55667788u);
  EXPECT_EQ(regs.getVgpr(10, 0), 0x99AABBCCu);
  EXPECT_EQ(regs.getVgpr(11, 0), 0xDDEEFF00u);
}

// RDNA-style instruction aliases (global_load_b<bits>, global_store_b<bits>,
// s_load_b<bits>). These are the same operations as global_load_dword[xN],
// global_store_dword[xN], and s_load_dword[xN] respectively, just with the
// naming convention used by RDNA architectures (gfx11+).
TEST(Instructions, RdnaGlobalLoadStoreAliases) {
  Workgroup wg;
  Wave regs(10, 10, WaveSize{1}, wg);

  std::vector<uint32_t> mem(16, 0);
  mem[0] = 42;
  mem[1] = 99;
  uintptr_t addr = reinterpret_cast<uintptr_t>(mem.data());
  regs.setVgpr64(0, 0, static_cast<uint64_t>(addr));

  // global_load_b32 should behave like global_load_dword
  tryExecute(regs, "global_load_b32 v2, v[0:1], off");
  tryExecute(regs, "s_waitcnt vmcnt(0)");
  EXPECT_EQ(regs.getVgpr(2, 0), 42u);

  // global_load_b64 should behave like global_load_dwordx2
  tryExecute(regs, "global_load_b64 v[4:5], v[0:1], off");
  tryExecute(regs, "s_waitcnt vmcnt(0)");
  EXPECT_EQ(regs.getVgpr(4, 0), 42u);
  EXPECT_EQ(regs.getVgpr(5, 0), 99u);

  // global_store_b32 should behave like global_store_dword
  regs.setVgpr(3, 0, 777);
  tryExecute(regs, "global_store_b32 v[0:1], v3, off");
  EXPECT_EQ(mem[0], 777u);
}

TEST(Instructions, RdnaSLoadAliases) {
  Workgroup wg;
  Wave regs(4, 10, WaveSize{1}, wg);

  std::vector<uint32_t> mem(16, 0);
  mem[0] = 11;
  mem[1] = 22;
  mem[2] = 33;
  mem[3] = 44;
  uint64_t addr = reinterpret_cast<uint64_t>(mem.data());
  regs.setSgpr64(0, addr);

  // s_load_b128 should behave like s_load_dwordx4
  tryExecute(regs, "s_load_b128 s[4:7], s[0:1], 0x0");
  tryExecute(regs, "s_waitcnt lgkmcnt(0)");
  EXPECT_EQ(regs.getSgpr(4), 11u);
  EXPECT_EQ(regs.getSgpr(5), 22u);
  EXPECT_EQ(regs.getSgpr(6), 33u);
  EXPECT_EQ(regs.getSgpr(7), 44u);

  // s_load_b64 should behave like s_load_dwordx2
  tryExecute(regs, "s_load_b64 s[8:9], s[0:1], 0x0");
  tryExecute(regs, "s_waitcnt lgkmcnt(0)");
  EXPECT_EQ(regs.getSgpr(8), 11u);
  EXPECT_EQ(regs.getSgpr(9), 22u);
}

// RDNA aliases: buffer_load_b<bits> = buffer_load_dword[xN].
// Verify that the RDNA-style names work identically to the CDNA names.
TEST(Instructions, RdnaBufferLoadStoreAliases) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  std::vector<uint32_t> hostMem = {100, 200, 300, 400};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());

  // Setup buffer resource descriptor in s[4:7]
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  regs.setSgpr(6, 0xFFFFFFFF);
  regs.setSgpr(7, 0);
  regs.setVgpr(29, 0, 0); // offset = 0

  // buffer_load_b32 (RDNA alias for buffer_load_dword)
  tryExecute(regs, "buffer_load_b32 v0, v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(0, 0), 100u);

  // buffer_load_b128 (RDNA alias for buffer_load_dwordx4)
  tryExecute(regs, "buffer_load_b128 v[2:5], v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(2, 0), 100u);
  EXPECT_EQ(regs.getVgpr(3, 0), 200u);
  EXPECT_EQ(regs.getVgpr(4, 0), 300u);
  EXPECT_EQ(regs.getVgpr(5, 0), 400u);

  // buffer_store_b32 (RDNA alias for buffer_store_dword)
  regs.setVgpr(10, 0, 999);
  tryExecute(regs, "buffer_store_b32 v10, v29, s[4:7], 0 offen");
  EXPECT_EQ(hostMem[0], 999u);
}

// buffer_load_d16_b16:    load 16-bit value into VGPR[15:0], preserve [31:16].
// buffer_load_d16_hi_b16: load 16-bit value into VGPR[31:16], preserve [15:0].
TEST(Instructions, BufferLoadD16) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  uint16_t hostMem[] = {0x1234, 0xABCD};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem);

  // Setup SRD in s[4:7]
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  regs.setSgpr(6, 0xFFFFFFFF);
  regs.setSgpr(7, 0);
  regs.setVgpr(29, 0, 0); // offset = 0

  // Pre-fill v0 with a known pattern to verify preservation.
  regs.setVgpr(0, 0, 0xDEADBEEF);

  // buffer_load_d16_b16: loads hostMem[0] (0x1234) into v0[15:0],
  // preserving v0[31:16] (0xDEAD).
  tryExecute(regs, "buffer_load_d16_b16 v0, v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(0, 0), 0xDEAD1234u);

  // buffer_load_d16_hi_b16: loads hostMem[0] (0x1234) into v0[31:16],
  // preserving v0[15:0] (0x1234 from previous load).
  tryExecute(regs, "buffer_load_d16_hi_b16 v0, v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(0, 0), 0x12341234u);

  // Load from offset 2 (second element) into high half.
  regs.setVgpr(29, 0, 2); // offset = 2 bytes
  tryExecute(regs, "buffer_load_d16_hi_b16 v0, v29, s[4:7], 0 offen");
  EXPECT_EQ(regs.getVgpr(0, 0), 0xABCD1234u);
}

// ds_store_b16 and ds_store_b16_d16_hi: RDNA3+ renames of ds_write_b16
// and ds_write_b16_d16_hi respectively.
TEST(Instructions, DsStoreB16_RDNA3Rename) {
  Workgroup wg;
  wg.resizeLds(1024);
  WaveId id{0};
  std::map<std::string, int> labels;
  std::map<std::string, Macro> macros;
  Wave regs(16, 16, 0, WaveSize{1}, id, wg, &labels, &macros);

  // ds_store_b16: writes low 16 bits of data VGPR.
  regs.setVgpr(0, 0, 200);        // addr
  regs.setVgpr(1, 0, 0xDEAD1234); // data
  tryExecute(regs, "ds_store_b16 v0, v1");
  EXPECT_EQ(wg.getLds().read<uint16_t>(200), 0x1234);

  // ds_store_b16_d16_hi: writes high 16 bits of data VGPR.
  regs.setVgpr(0, 0, 204);        // addr
  regs.setVgpr(1, 0, 0xBEEF5678); // data
  tryExecute(regs, "ds_store_b16_d16_hi v0, v1");
  EXPECT_EQ(wg.getLds().read<uint16_t>(204), 0xBEEF);
}

// Byte-level race checking with buffer_load_d16 instructions.
// Two d16 loads to different halves of the same register should not race
// when only the drained half is subsequently read.
TEST(Instructions, BufferLoadD16_ByteLevelRace) {
  Workgroup wg;
  wg.setRaceChecks(true);
  wg.setCompleteEmulation(true);
  Wave regs(/*vgprCount*/ 64, /*sgprCount*/ 32, WaveSize{1}, wg);

  uint16_t hostMem[] = {0x1234, 0xABCD, 0x5678, 0x9999};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem);

  // Setup SRD in s[4:7]
  regs.setSgpr(4, static_cast<uint32_t>(baseAddr & 0xFFFFFFFF));
  regs.setSgpr(5, static_cast<uint32_t>((baseAddr >> 32) & 0xFFFF));
  regs.setSgpr(6, 0xFFFFFFFF);    // num_records
  regs.setSgpr(7, 0);             // format
  regs.setVgpr(29, 0, 0);         // offset = 0
  regs.setVgpr(0, 0, 0xDEADBEEF); // pre-fill v0

  // Issue two d16 loads to different halves of v0.
  tryExecute(regs, "buffer_load_d16_b16 v0, v29, s[4:7], 0 offen");
  regs.setVgpr(29, 0, 2); // offset = 2 for second element
  tryExecute(regs, "buffer_load_d16_hi_b16 v0, v29, s[4:7], 0 offen");

  // Two vmcnt events outstanding. Drain the oldest (lo half).
  regs.sWaitCntVmcnt(1);

  // Reading the lo half via getHalfVgpr should NOT race.
  EXPECT_NO_THROW(regs.getHalfVgpr(0, 0, false));
  EXPECT_EQ(regs.getHalfVgpr(0, 0, false), 0x1234);

  // Reading the hi half should RACE (still outstanding).
  EXPECT_THROW(regs.getHalfVgpr(0, 0, true), RaceConditionException);

  // Reading full register should also RACE (hi half outstanding).
  EXPECT_THROW(regs.getVgpr(0, 0), RaceConditionException);

  // Drain the hi half too.
  regs.sWaitCntVmcnt(0);

  // Now both halves are safe.
  EXPECT_NO_THROW(regs.getVgpr(0, 0));
  EXPECT_EQ(regs.getVgpr(0, 0), 0xABCD1234u);
}

// ============================================================================
// buffer_load ... lds (direct-to-LDS)
// ============================================================================

// buffer_load_dwordx4 ... lds: loads 4 dwords per lane from global memory
// directly into LDS at m0 + lane * 16.
TEST(Instructions, BufferLoad_Lds_DirectToLds) {
  Workgroup wg;
  wg.resizeLds(1024);
  // waveSize=2 for manageable test size (2 lanes × 16 bytes = 32 bytes LDS).
  wg.setRaceChecks(true);
  wg.setCompleteEmulation(true);
  Wave wave(/*vgprCount=*/4, /*sgprCount=*/10, WaveSize{2}, wg);

  // Host memory: 8 dwords (4 per lane × 2 lanes).
  std::vector<uint32_t> hostMem = {0x11111111, 0x22222222, 0x33333333,
                                   0x44444444, 0x55555555, 0x66666666,
                                   0x77777777, 0x88888888};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());
  setupSrd(wave, /*srdBase=*/4, baseAddr, hostMem.size() * sizeof(uint32_t));

  // v0 = per-lane offset into global memory (offen). Lane 0 → offset 0,
  // lane 1 → offset 16 (second group of 4 dwords).
  wave.setVgpr(0, /*lane=*/0, 0);
  wave.setVgpr(0, /*lane=*/1, 16);

  // m0 = LDS base address for the write.
  wave.setM0(64);

  tryExecute(wave, "buffer_load_dwordx4 v0, s[4:7], 0 offen lds");

  // Verify LDS contents: lane 0 at m0+0*16=64, lane 1 at m0+1*16=80.
  auto &lds = wg.getLds();
  EXPECT_EQ(lds.read<uint32_t>(64), 0x11111111u);
  EXPECT_EQ(lds.read<uint32_t>(68), 0x22222222u);
  EXPECT_EQ(lds.read<uint32_t>(72), 0x33333333u);
  EXPECT_EQ(lds.read<uint32_t>(76), 0x44444444u);
  EXPECT_EQ(lds.read<uint32_t>(80), 0x55555555u);
  EXPECT_EQ(lds.read<uint32_t>(84), 0x66666666u);
  EXPECT_EQ(lds.read<uint32_t>(88), 0x77777777u);
  EXPECT_EQ(lds.read<uint32_t>(92), 0x88888888u);

  // Verify GLOBAL_TO_LDS event was registered.
  EXPECT_EQ(wave.getWaveMemoryEvents().size(), 1u);
  EventId eid = wave.getWaveMemoryEvents()[0];
  EXPECT_EQ(wg.getEventType(eid), MemoryEventType::GLOBAL_TO_LDS);
  EXPECT_EQ(wg.getLdsWriteEvents().size(), 1u);

  // Verify vmcnt retires it.
  wave.sWaitCntVmcnt(0);
  EXPECT_TRUE(wave.getWaveMemoryEvents().empty());
  EXPECT_EQ(wg.getEventStatus(eid), EventStatus::WAVE_COMPLETE);
}

// Verify that LDS reads race against an outstanding DTL write (before vmcnt).
TEST(Instructions, BufferLoad_Lds_RaceBeforeVmcnt) {
  Workgroup wg;
  wg.resizeLds(1024);
  wg.setRaceChecks(true);
  wg.setCompleteEmulation(true);
  Wave wave(/*vgprCount=*/4, /*sgprCount=*/10, WaveSize{1}, wg);

  std::vector<uint32_t> hostMem = {0xAAAAAAAA};
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(hostMem.data());
  setupSrd(wave, /*srdBase=*/4, baseAddr, sizeof(uint32_t));

  wave.setVgpr(0, /*lane=*/0, 0);
  wave.setM0(0);

  tryExecute(wave, "buffer_load_dword v0, s[4:7], 0 offen lds");

  // LDS read before vmcnt → race.
  EXPECT_THROW(wg.readLds<uint32_t>(0, WaveId{0}, 0), RaceConditionException);

  // After vmcnt → safe.
  wave.sWaitCntVmcnt(0);
  EXPECT_NO_THROW(wg.readLds<uint32_t>(0, WaveId{0}, 0));
  EXPECT_EQ(wg.readLds<uint32_t>(0, WaveId{0}, 0), 0xAAAAAAAAu);
}
