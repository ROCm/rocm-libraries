#include "race-emulator/Emulator.h"
#include <cstring> // For std::memcpy
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

using namespace raceemulator;

// Setup the assembly for a function of the form

/*
 *
  __global__ void foo(void *data) {
      int tid = threadIdx.x;
      // stuff

 }
*/

static constexpr std::string_view boiler = R"BOILER(
	.amdhsa_kernel foo
 		.amdhsa_group_segment_fixed_size 1024
 		.amdhsa_private_segment_fixed_size 0
 		.amdhsa_kernarg_size 8
 	  .amdhsa_user_sgpr_count 2
  	  .amdhsa_user_sgpr_dispatch_ptr 0
  	  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  	  .amdhsa_user_sgpr_dispatch_id 0
  	  .amdhsa_enable_private_segment 0
  	  .amdhsa_system_sgpr_workgroup_id_x 1
      .amdhsa_next_free_sgpr 10 ; might be more that 10!
      .amdhsa_next_free_vgpr 10 ; might be more that 10!
      .amdhsa_accum_offset 10
	.end_amdhsa_kernel
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .name:           foo
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
...
)BOILER";

struct RaceVerifier {
  std::optional<RaceConditionException::Space> space;
  std::optional<int> address;  // LDS Byte or Register Index
  std::optional<bool> isWrite; // Did the crash happen on a Write?
  std::optional<int> waveId;   // Which wave crashed?

  std::optional<std::string> instructionSubstring;

  static RaceVerifier LdsAccess(int addr) {
    RaceVerifier v;
    v.space = RaceConditionException::Space::LDS;
    v.address = addr;
    return v;
  }

  static RaceVerifier VgprAccess(int regIdx) {
    RaceVerifier v;
    v.space = RaceConditionException::Space::VGPR;
    v.address = regIdx;
    return v;
  }

  RaceVerifier &onWrite() {
    isWrite = true;
    return *this;
  }
  RaceVerifier &onRead() {
    isWrite = false;
    return *this;
  }
  RaceVerifier &inWave(int w) {
    waveId = w;
    return *this;
  }
  RaceVerifier &onInstruction(std::string text) {
    instructionSubstring = text;
    return *this;
  }
};

class RaceTestFixture : public ::testing::Test {
protected:
  void ExpectRace(const std::string &assemblyBody,
                  const std::string &expectedMsgPart, int nGlobalBytes = 16,
                  int nWaves = 1,
                  std::optional<RaceVerifier> verifier = std::nullopt) {
    Emulator emulator = getBoilerEmulator(assemblyBody, nGlobalBytes);
    try {
      emulator.enableRaceChecks(true);
      emulator.run(/* wgId= */ 0,
                   {nWaves * 64, 1, 1}); // nWaves waves * 64 threads/wave
      FAIL() << "Expected RaceConditionException, but simulation completed "
                "successfully.";
    } catch (const RaceConditionException &e) {

      if (verifier.has_value()) {
        const RaceVerifier &v = verifier.value();

        if (v.space.has_value()) {
          EXPECT_EQ(e.space, v.space.value());
        }
        if (v.address.has_value()) {
          EXPECT_EQ(e.index, v.address.value());
        }
        if (v.isWrite.has_value()) {
          EXPECT_EQ(e.isWrite, v.isWrite.value());
        }
        if (v.waveId.has_value()) {
          EXPECT_EQ(e.wave, v.waveId.value());
        }
        if (v.instructionSubstring.has_value()) {
          // TODO(newling) need to capture instruction text in exception
          // EXPECT_PRED_FORMAT2(::testing::IsSubstring,
          //                    v.instructionSubstring.value(),
          //                    e.instructionText);
        }
      }
      // We caught it! Now verify it's the *right* race.
      std::string report = e.what();
      EXPECT_PRED_FORMAT2(::testing::IsSubstring, expectedMsgPart, report);

    } catch (const std::exception &e) {
      FAIL() << "Expected RaceConditionException, but got generic error: "
             << e.what();
    }
  }

  void ExpectSuccess(const std::string &assemblyBody, int nGlobalBytes = 16,
                     int nWaves = 1) {

    try {
      Emulator emulator = getBoilerEmulator(assemblyBody, nGlobalBytes);
      emulator.enableRaceChecks(true);
      emulator.run(/* wgId= */ 0,
                   {nWaves * 64, 1, 1}); // nWaves waves * 64 threads/wave
    } catch (const EmulatorException &e) {
      FAIL() << "Expected successful execution, but got EmulatorException: "
             << e.what();
    }
    // catch all others:
    catch (const std::exception &e) {
      FAIL() << "Expected successful execution, but got generic error: "
             << e.what();
    }
  }

private:
  Emulator getBoilerEmulator(std::string_view assembly, int nGlobalBytes) {
    // Construct a string that is "race" + "boiler":
    std::string combined =
        "foo:\n" + std::string(assembly) + std::string(boiler);
    auto emulator = Emulator::createGfx942(combined);
    emulator.enableRaceChecks(true);

    h_data.resize(nGlobalBytes / 4 + 1);
    std::iota(h_data.begin(), h_data.end(), 0);
    int *d_data = h_data.data();
    emulator.addKernarg(0, &d_data);
    return emulator;
  }

  std::vector<int> h_data;
};

TEST_F(RaceTestFixture, DsWriteToDsReadMissingBarrier) {

  // Based on HIP code:
  //
  //   __shared__ int temp[256];
  //   int tid = threadIdx.x;
  //   temp[tid] = data[threadIdx.x];
  //   __syncthreads();
  //   data[tid] = temp[256 - tid - 1];
  //

  const auto code = R"ASM(

  ; Each thread loads a distinct 4 bytes from global to a vector register.
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  v_sub_u32_e32 v2, 0, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  s_waitcnt vmcnt(0)

  ; Each thread writes its 4 bytes to LDS.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)

  ;  s_barrier <--- MISSING BARRIER
  ; Each threads reads from LDS, from an address written by another wave.
  ds_read_b32 v1, v2 offset:1020
  s_waitcnt lgkmcnt(0)
  global_store_dword v0, v1, s[0:1]
  s_endpgm
    )ASM";

  // We have 1 exact character level test for each type, the others must be
  // testing the underlying logc not string details.
  auto msg0 = R"MSG(
LDS race in byte 512 detected. Race between a pair in:

Wave 2 Lane 0:
11     |   ; Each thread writes its 4 bytes to LDS.
12 --> |   ds_write_b32 v0, v1
13     |   s_waitcnt lgkmcnt(0)

Wave 1 Lane 63:
16     |   ; Each threads reads from LDS, from an address written by another wave.
17 --> |   ds_read_b32 v1, v2 offset:1020
18     |   s_waitcnt lgkmcnt(0)
)MSG";

  // With 4 waves, we need a barrier because threads in different waves
  // read/write the same byte.
  int nGlobalBytes = 1024;
  int nWaves = 4;
  ExpectRace(code, msg0, nGlobalBytes, nWaves);

  // With 2 waves, not a problem, different addresses Although unitialised LDS).
  nWaves = 2;
  ExpectSuccess(code, nGlobalBytes, nWaves);
}

TEST_F(RaceTestFixture, GlobalLoadToLdsWriteMissingVmcnt) {

  // Based on HIP code:
  //
  //   __shared__ int temp[256];
  //   int tid = threadIdx.x;
  //   temp[tid] = data[threadIdx.x];
  //   __syncthreads();
  //   data[tid] = temp[256 - tid - 1];
  //

  const auto code = R"ASM(

  ; Initialization
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	v_sub_u32_e32 v2, 0, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  ; s_waitcnt vmcnt(0) <-- MISSING!

  ; Write to LDS
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)

  ;  Read from LDS, from an address written by another wave.
 	s_barrier
	ds_read_b32 v1, v2 offset:1020
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
    )ASM";

  auto msg0 = R"MSG(
VGPR race detected on line 12 (wave 0, lane 0). Conflicting events:

7     |   s_waitcnt lgkmcnt(0)
8 --> |   global_load_dword v1, v0, s[0:1]
9     |   ; s_waitcnt vmcnt(0) <-- MISSING!

11     |   ; Write to LDS
12 --> |   ds_write_b32 v0, v1
13     |   s_waitcnt lgkmcnt(0)
)MSG";

  // With 4 waves, we need a barrier because threads in different waves
  // read/write the same byte.
  ExpectRace(code, msg0, 1024, 4);
}

// This is the more sustainable way of testing, less brittle to string changes.
TEST_F(RaceTestFixture, GlobalLoadToMathInsufficientVmcnt) {

  // Based on HIP code:
  //
  //   int tid = threadIdx.x;
  //   auto v1 = data[threadIdx.x];
  //   auto v2 = data[threadIdx.x+17];
  //   auto v3 =  v1 + v2;
  //   data[threadIdx.x] = v3;

  const auto code = R"ASM(
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v0, s[0:1]
	global_load_dword v2, v0, s[0:1] offset:68
	s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
	v_add_u32_e32 v1, v2, v1
	global_store_dword v0, v1, s[0:1]
	s_endpgm
    )ASM";

  // ExpectSuccess(code, 512, 1);

  ExpectRace(code, "", 512, 1,
             RaceVerifier::VgprAccess(2).onInstruction("v_add_u32_e32"));
}

// Verify that the VGPR race message includes wave and lane information.
TEST_F(RaceTestFixture, VgprRaceMessageIncludesWaveAndLane) {

  const auto code = R"ASM(
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_load_dword v1, v0, s[0:1]
	global_load_dword v2, v0, s[0:1] offset:68
	s_waitcnt vmcnt(1) ; <--- SHOULD WAIT FOR BOTH!
	v_add_u32_e32 v1, v2, v1
	global_store_dword v0, v1, s[0:1]
	s_endpgm
    )ASM";

  ExpectRace(code, "VGPR race detected on line 8 (wave 0, lane 0)", 512, 1);
}

TEST_F(RaceTestFixture, DsWriteToDsReadInsufficientLgkmCnt) {

  const std::string code = R"ASM(
  ; v0 is initialized with the thread id.
  ; write to LDS at byte address threadId * 4:
  v_lshlrev_b32_e32 v0, 2, v0
  ds_write_b32 v0, v1

  ; Wait for the write to complete:
  s_waitcnt lgkmcnt(0)

  ; Read the value back:
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)

  s_endpgm
    )ASM";

  ExpectSuccess(code, 0, 1);

  const std::string codeSansWait = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_write_b32 v0, v1
  ; Missing waitcnt here!
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
    )ASM";

  ExpectRace(codeSansWait, "", 0, 1,
             RaceVerifier::LdsAccess(0).onInstruction("ds_read_b32"));
}

TEST_F(RaceTestFixture, DsWriteOverWriteIsFine) {

  const std::string writeWaitWrite = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ds_write_b32 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
    )ASM";

  ExpectSuccess(writeWaitWrite, 0, 1);

  const std::string writeWrite = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_write_b32 v0, v1
  ds_write_b32 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
    )ASM";

  ExpectSuccess(writeWrite, 0, 1);
}

TEST_F(RaceTestFixture, DSReadOverReadIsFine) {

  // Case where the sources are the same:
  const std::string readRead = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_read_b32 v1, v0
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
    )ASM";
  ExpectSuccess(readRead, 0, 1);

  // Case where the destination is the same:
  const std::string readReadSameDst = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  print int v0 0

  ; set v1 to be v0 plus 4:
  v_add_u32_e32 v1, 4, v0
  print int v1 0
  ds_read_b32 v3, v0
  ds_read_b32 v3, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
    )ASM";
  ExpectSuccess(readReadSameDst, 0, 1);
}

// TODO(newling) confirm racey:
//   ds_read_b32 v1, v0
//   ds_write_b32 v0, v2

TEST_F(RaceTestFixture, DsReadToDsWriteInsufficientLgkmCnt) {
  const std::string code = R"ASM(
  ; v0 is initialized with the thread id.
  ; read from LDS at byte address threadId * 4:
  v_lshlrev_b32_e32 v0, 2, v0
  ds_read_b32 v1, v0

  ; Wait for read to complete.
  s_waitcnt lgkmcnt(0)

  ; Write to LDS at same address.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);

  const std::string codeSansWait = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_read_b32 v1, v0
  ds_write_b32 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(codeSansWait, "", 0, 1,
             RaceVerifier::LdsAccess(0).onInstruction("ds_write_b32"));

  const std::string foo = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  ds_read_b32 v1, v0
  ds_read_b32 v1, v0
  ds_read_b32 v1, v0
  ds_read_b32 v1, v0
  ds_write_b32 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(foo, "", 0, 1,
             RaceVerifier::LdsAccess(0).onInstruction("ds_write_b32"));
}

// TODO(newling): test where global_load and ds_read to same destination.
// TODO(newling): error message should just have 2 racing lines.
//                1) where it is first detected
//                2) most recent incomplete conflictor.
