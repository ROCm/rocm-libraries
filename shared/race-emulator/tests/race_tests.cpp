// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Emulator.h"
#include "race-emulator/IntervalSet.h"
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
      emulator.run(Dim3d(0), {nWaves * 64, 1, 1},
                   {.raceChecks = true}); // nWaves waves * 64 threads/wave
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
    Emulator emulator = getBoilerEmulator(assemblyBody, nGlobalBytes);
    emulator.run(Dim3d(0), {nWaves * 64, 1, 1},
                 {.raceChecks = true}); // nWaves waves * 64 threads/wave
  }

private:
  Emulator getBoilerEmulator(std::string_view assembly, int nGlobalBytes) {
    // Construct a string that is "race" + "boiler":
    std::string combined =
        "foo:\n" + std::string(assembly) + std::string(boiler);
    auto emulator = Emulator::createGfx942(combined);

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

Wave 1:
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

// All 64 lanes write the same value (42) to LDS[0], then all read it back.
// This creates 64 LDS_TO_VGPR events on byte 0. s_waitcnt lgkmcnt(0) must
// mark all 64 as WAVE_COMPLETE so the subsequent write doesn't falsely race.
TEST_F(RaceTestFixture, DsReadMultipleLanesSameByteNoRace) {
  const std::string code = R"ASM(
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  ds_write_b32 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// Scatter-then-broadcast: each lane writes its thread ID to LDS[lane*4],
// then all lanes read from LDS[0] (the value lane 0 wrote).
TEST_F(RaceTestFixture, DsScatterThenBroadcastReadNoRace) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 7
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v0, 0
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// All lanes write different values (their thread ID) to the same LDS address.
// This is a WAW race: the final value depends on which lane writes last.
// TODO: this should be ExpectRace once WAW detection is implemented. The race
// checker currently only detects RAW and WAR hazards on LDS, not WAW.
TEST_F(RaceTestFixture, DsAllLanesWriteSameAddressIsRace) {
  const std::string code = R"ASM(
  v_mov_b32_e32 v1, v0
  v_mov_b32_e32 v0, 0
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// TODO(newling): test where global_load and ds_read to same destination.
// TODO(newling): error message should just have 2 racing lines.
//                1) where it is first detected
//                2) most recent incomplete conflictor.

// --- Byte-level VGPR race tracking tests ---
// These verify that d16 instructions (which touch only 16 bits of a VGPR)
// create events on the correct byte mask, so that accesses to the other
// half of the register are not flagged as races.

// Two LDS d16 loads to different halves of the same register.
// Drain the lo-half event, then read the lo half. No race.
TEST_F(RaceTestFixture, D16_LdsLoad_NoFalsePositive) {
  const auto code = R"ASM(
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)

  ; Write some data to LDS first
  v_mov_b32_e32 v1, 0x12345678
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_barrier

  ; Load lo half and hi half of v2 from LDS
  ds_load_u16_d16 v2, v0         ; event on v2 bytes [0,1]
  ds_load_u16_d16_hi v2, v0      ; event on v2 bytes [2,3]
  s_waitcnt lgkmcnt(1)           ; drain oldest (lo half)

  ; Read lo half of v2 — should NOT race (lo event drained)
  ; Write to a different LDS address to avoid LDS WAR conflict.
  ds_write_b16 v0, v2 offset:512
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// D16 hi load outstanding, but ds_store_b16 reads only lo half. No race.
TEST_F(RaceTestFixture, D16_HiLoadOutstanding_LoReadSafe) {
  const auto code = R"ASM(
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)

  v_mov_b32_e32 v1, 0xAABBCCDD
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_barrier

  ; Load only hi half of v2
  ds_load_u16_d16_hi v2, v0      ; event on v2 bytes [2,3]

  ; Read lo half of v2 — no event on bytes [0,1], so safe.
  ; Write to a different LDS address to avoid LDS WAR conflict.
  ds_write_b16 v0, v2 offset:512
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// Full-register LDS load, then read without waitcnt — should race.
// (backward compatibility: full-register events still work)
TEST_F(RaceTestFixture, D16_FullLoadStillRaces) {
  const auto code = R"ASM(
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)

  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_barrier

  ; Full 32-bit load to v2 — event on all 4 bytes
  ds_load_b32 v2, v0
  ; NO waitcnt — reading v2 should race
  ds_write_b16 v0, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "VGPR race", 0, 1, RaceVerifier::VgprAccess(2).onRead());
}

// Cross-wave LDS race detected regardless of wave scheduling order.
//
// Two waves: wave 0 writes to LDS[0..255], wave 1 writes to LDS[256..511].
// Both waves then read LDS[0] (a cross-wave read for wave 1) without a
// barrier. A later s_barrier appears before s_endpgm.
//
// The race must be detected regardless of wave scheduling order, because
// flushWaveCompleteMemoryEvents (which removes events from LDS tracking)
// only runs when ALL waves have reached the barrier, not when each wave
// individually arrives.
TEST(WaveScheduleTest, CrossWaveLdsRaceDetectedForAllSchedules) {

  const std::string code = std::string("foo:\n") + R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0   ; v0 = threadId * 4 (per-wave LDS address)
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1            ; each wave writes to its own LDS region
  s_waitcnt lgkmcnt(0)
  ; s_barrier                     ; <--- MISSING (should sync before read)
  v_mov_b32_e32 v2, 0
  ds_read_b32 v3, v2             ; all lanes read LDS[0] (cross-wave)
  s_waitcnt lgkmcnt(0)
  s_barrier                       ; a later barrier before termination
  s_endpgm
  )ASM" + std::string(boiler);

  auto runWith = [&](WaveSchedule sched) -> bool {
    auto emulator = Emulator::createGfx942(code);
    std::vector<int> h(2, 0);
    int *p = h.data();
    emulator.addKernarg(0, &p);
    try {
      emulator.run({0, 0, 0}, {128, 1, 1},
                   {.raceChecks = true, .waveSchedule = sched});
      return false;
    } catch (...) {
      return true;
    }
  };

  EXPECT_TRUE(runWith(WaveSchedule::Sequential));
  EXPECT_TRUE(runWith(WaveSchedule::RoundRobin));
}

// D16 lo load outstanding, full-register read — should race.
TEST_F(RaceTestFixture, D16_LoLoadOutstanding_FullReadRaces) {
  const auto code = R"ASM(
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)

  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_barrier

  ; Load lo half of v2
  ds_load_u16_d16 v2, v0         ; event on v2 bytes [0,1]
  ; NO waitcnt — full read of v2 should race (bytes [0,1] outstanding)
  v_mov_b32_e32 v3, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "VGPR race", 0, 1, RaceVerifier::VgprAccess(2).onRead());
}

// ---------------------------------------------------------------------------
// Same-wave LDS race tests: different lanes access overlapping LDS addresses.
// ---------------------------------------------------------------------------

// Lane 0 writes to LDS[0], then all lanes (including lane 0) read LDS[0]
// WITHOUT s_waitcnt. The write is still ACTIVE → race.
TEST_F(RaceTestFixture, SameWave_WriteReadNoWaitcnt_IsRace) {
  const std::string code = R"ASM(
  ; Lane 0 writes 42 to LDS[0]. All other lanes write to distinct addresses.
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  ; NO s_waitcnt — the write is still ACTIVE
  ; All lanes read from LDS[0]
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// Lane 0 writes to LDS[0], s_waitcnt, then all lanes read LDS[0].
// Same wave + WAVE_COMPLETE → safe.
TEST_F(RaceTestFixture, SameWave_WriteWaitcntRead_NoRace) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ; All lanes read from LDS[0] (lane 0's write address)
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// Two different instructions (different PCs): lane 0 writes LDS[0],
// s_waitcnt, then lane 0 writes LDS[0] again without s_waitcnt before
// reading. The second write is ACTIVE → race on read.
TEST_F(RaceTestFixture, SameWave_DoubleWriteThenRead_IsRace) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ; First write at each lane's own address
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ; Second write at lane's own address — no waitcnt after this
  ds_write_b32 v0, v1
  ; Read from LDS[0] while lane 0's second write is still ACTIVE
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// ---------------------------------------------------------------------------
// Cross-wave LDS race tests with precise error messages.
// ---------------------------------------------------------------------------

// Two waves: wave 0 writes LDS[0..255], wave 1 reads LDS[0] without barrier.
// Verify the error message reports correct waves and lanes.
TEST(WaveScheduleTest, CrossWaveLdsRace_ErrorMessageWaves) {
  const std::string code = std::string("foo:\n") + R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0   ; v0 = threadId * 4
  v_mov_b32_e32 v1, 99
  ds_write_b32 v0, v1            ; each lane writes to its own 4 bytes
  s_waitcnt lgkmcnt(0)
  ; s_barrier                     ; <--- MISSING
  v_mov_b32_e32 v2, 0
  ds_read_b32 v3, v2             ; all lanes read LDS[0]
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_endpgm
  )ASM" + std::string(boiler);

  auto emulator = Emulator::createGfx942(code);
  std::vector<int> h(2, 0);
  int *p = h.data();
  emulator.addKernarg(0, &p);
  try {
    emulator.run({0, 0, 0}, {128, 1, 1}, {.raceChecks = true});
    FAIL() << "Expected race exception";
  } catch (const RaceConditionException &e) {
    std::string msg = e.what();
    // The race is between wave 0's write to LDS[0] and wave 1's read of LDS[0].
    EXPECT_NE(msg.find("LDS race in byte 0"), std::string::npos);
    // Both waves should be mentioned in the report.
    EXPECT_NE(msg.find("Wave 0"), std::string::npos);
    EXPECT_NE(msg.find("Wave 1"), std::string::npos);
  }
}

// Two waves write to the same LDS address (WAR: wave 0 reads, wave 1 writes
// without barrier). Verify waves and instruction context in the message.
TEST(WaveScheduleTest, CrossWaveLdsWriteAfterRead_ErrorMessage) {
  const std::string code = std::string("foo:\n") + R"ASM(
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  ; All lanes write to LDS[0]
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  ; s_barrier                     ; <--- MISSING
  ; All lanes read from LDS[0] — wave 0 reads, but wave 1's write is pending
  ds_read_b32 v2, v0
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_endpgm
  )ASM" + std::string(boiler);

  auto emulator = Emulator::createGfx942(code);
  std::vector<int> h(2, 0);
  int *p = h.data();
  emulator.addKernarg(0, &p);
  try {
    emulator.run({0, 0, 0}, {128, 1, 1}, {.raceChecks = true});
    FAIL() << "Expected race exception";
  } catch (const RaceConditionException &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("LDS race in byte 0"), std::string::npos);
    // Should mention ds_write_b32 and ds_read_b32 in the code context.
    EXPECT_NE(msg.find("ds_write_b32"), std::string::npos);
    EXPECT_NE(msg.find("ds_read_b32"), std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// s_barrier does not substitute for s_waitcnt within a single wave.
// ---------------------------------------------------------------------------

// ds_write, s_barrier (no s_waitcnt), ds_read → still races.
// The barrier flushes WAVE_COMPLETE events, but without s_waitcnt the write
// is still ACTIVE and stays in the event tracking.
TEST_F(RaceTestFixture, SameWave_BarrierWithoutWaitcnt_StillRaces) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_barrier
  ; Write is still ACTIVE (no s_waitcnt), barrier only flushes WAVE_COMPLETE.
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// ds_write, s_waitcnt, s_barrier, ds_read → safe.
// s_waitcnt makes it WAVE_COMPLETE, barrier removes it entirely.
TEST_F(RaceTestFixture, SameWave_WaitcntThenBarrier_NoRace) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_barrier
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// ---------------------------------------------------------------------------
// Deep event stack: multiple outstanding LDS events with partial retirement.
// ---------------------------------------------------------------------------

// Three ds_writes to different addresses (via different lanes), then
// s_waitcnt lgkmcnt(1) retires the oldest 2, leaving 1 outstanding.
// Read from the newest write's address → race (still ACTIVE).
// Read from an old write's address → safe (WAVE_COMPLETE, same wave).
TEST_F(RaceTestFixture, DeepEventStack_PartialWaitcnt_NewestRaces) {
  const std::string code = R"ASM(
  ; Three writes to distinct LDS addresses using lane 0's address.
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 10
  ds_write_b32 v0, v1              ; event 0: each lane writes
  v_mov_b32_e32 v1, 20
  ds_write_b32 v0, v1              ; event 1: same addresses
  v_mov_b32_e32 v1, 30
  ds_write_b32 v0, v1              ; event 2: same addresses
  ; Retire oldest 2, keep newest 1 outstanding.
  s_waitcnt lgkmcnt(1)
  ; Read from LDS[0] — the newest write (event 2) is still ACTIVE.
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// Three ds_writes, s_waitcnt lgkmcnt(0) retires all, then read → safe.
TEST_F(RaceTestFixture, DeepEventStack_FullWaitcnt_AllSafe) {
  const std::string code = R"ASM(
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 10
  ds_write_b32 v0, v1
  v_mov_b32_e32 v1, 20
  ds_write_b32 v0, v1
  v_mov_b32_e32 v1, 30
  ds_write_b32 v0, v1
  ; Retire all.
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// ---------------------------------------------------------------------------
// Exec mask tests: partial lane activity.
// ---------------------------------------------------------------------------

// Only lane 0 active: writes to LDS[0]. Then full exec reads LDS[0] without
// s_waitcnt → race on byte 0.
TEST_F(RaceTestFixture, ExecMask_PartialWrite_FullRead_IsRace) {
  const std::string code = R"ASM(
  ; Only lane 0 active.
  s_mov_b64 exec, 1
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  ; Restore full exec.
  s_mov_b64 exec, -1
  ; All lanes read LDS[0] — lane 0's write is still ACTIVE.
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// Only lane 0 active: writes to LDS[0], s_waitcnt, then full exec reads
// LDS[0] → safe (same wave, WAVE_COMPLETE).
TEST_F(RaceTestFixture, ExecMask_PartialWrite_Waitcnt_FullRead_NoRace) {
  const std::string code = R"ASM(
  s_mov_b64 exec, 1
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, -1
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// Lanes 0-31 write to LDS[0..127], then lanes 32-63 read LDS[0] without
// s_waitcnt → race (lane 0's write is ACTIVE, lane 32 reads it).
TEST_F(RaceTestFixture, ExecMask_DisjointLanes_OverlappingAddresses_IsRace) {
  const std::string code = R"ASM(
  ; Lanes 0-31 active: each writes to LDS[lane*4].
  s_mov_b64 exec, 0x00000000FFFFFFFF
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  ; NO s_waitcnt
  ; Lanes 32-63 active: all read from LDS[0].
  s_mov_b64 exec, 0xFFFFFFFF00000000
  v_mov_b32_e32 v3, 0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, -1
  s_endpgm
  )ASM";
  ExpectRace(code, "LDS race in byte 0", 0, 1,
             RaceVerifier::LdsAccess(0).onRead());
}

// Lanes 0-31 write to LDS[0..127], then lanes 32-63 read from LDS[128..255]
// (non-overlapping addresses) → no race, even without s_waitcnt.
TEST_F(RaceTestFixture, ExecMask_DisjointLanes_DisjointAddresses_NoRace) {
  const std::string code = R"ASM(
  ; Lanes 0-31 active: each writes to LDS[lane*4] (addresses 0..124).
  s_mov_b64 exec, 0x00000000FFFFFFFF
  v_lshlrev_b32_e32 v0, 2, v0
  v_mov_b32_e32 v1, 42
  ds_write_b32 v0, v1
  ; Lanes 32-63 active: each reads from LDS[lane*4] (addresses 128..252).
  ; These addresses don't overlap with the writes.
  s_mov_b64 exec, 0xFFFFFFFF00000000
  v_lshlrev_b32_e32 v3, 2, v0
  ds_read_b32 v2, v3
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, -1
  s_endpgm
  )ASM";
  ExpectSuccess(code, 0, 1);
}

// ---------------------------------------------------------------------------
// Cross-wave test with partial exec masks and selective s_waitcnt.
// ---------------------------------------------------------------------------
//
// Two waves, wave-64. Each wave does 4 writes with 16 lanes active per write
// (cycling through quarters of the exec mask), writing to disjoint LDS regions.
// After s_waitcnt + s_barrier, each wave does 2 reads with 32 lanes active per
// read, reading from the OTHER wave's data. The s_waitcnt before each read
// retires a different number of the wave's own writes.
//
// LDS layout (each wave writes 256 bytes = 4 writes × 16 lanes × 4 bytes):
//   Wave 0 writes to LDS [0, 256):
//     Write 0 (lanes 0-15):   [0, 64)
//     Write 1 (lanes 16-31):  [64, 128)
//     Write 2 (lanes 32-47):  [128, 192)
//     Write 3 (lanes 48-63):  [192, 256)
//   Wave 1 writes to LDS [256, 512):
//     Write 0 (lanes 0-15):   [256, 320)
//     Write 1 (lanes 16-31):  [320, 384)
//     Write 2 (lanes 32-47):  [384, 448)
//     Write 3 (lanes 48-63):  [448, 512)
//
// After barrier, each wave reads from the other wave's region:
//   Read 0 (lanes 0-31): reads [otherBase, otherBase+128)
//   Read 1 (lanes 32-63): reads [otherBase+128, otherBase+256)
//
// Parameterized helper for cross-wave partial-exec tests.
// Two waves, each does 4 ds_write with 16 lanes active per write (quarters
// of exec mask), writing to disjoint LDS regions. Two barriers separate the
// reads: read 0 (lanes 0-31) reads other wave's writes 0+1, read 1
// (lanes 32-63) reads other wave's writes 2+3.
//
// lgkmcnt1: s_waitcnt value before barrier 1. With 4 outstanding writes,
//           lgkmcnt(N) waits until at most N remain (retires the 4-N oldest).
// lgkmcnt2: s_waitcnt value before barrier 2. Outstanding ops at that point
//           are remaining writes + read 0. Set to -1 to omit the s_waitcnt.
// barrier1/barrier2: whether to include each s_barrier.
//
// Correct: lgkmcnt1=2 keeps writes 2+3, retires 0+1. lgkmcnt2=0 retires all.
// Over-retiring is safe: lgkmcnt1=0 retires all writes before barrier 1.
// Under-retiring races: lgkmcnt1=3 keeps 3 writes, only retires write 0,
//   so write 1 is still ACTIVE when read 0 touches its LDS region.
//
static std::string crossWavePartialExecCode(int lgkmcnt1, int lgkmcnt2,
                                            bool barrier1 = true,
                                            bool barrier2 = true) {
  std::string code = R"ASM(
  ; v0 = threadIdx.x (set by emulator, bits [9:0])
  ; Compute waveId in s4: threadIdx.x / 64
  v_readfirstlane_b32 s4, v0
  s_lshr_b32 s4, s4, 6

  ; Compute lane-local ID in v5: threadIdx.x % 64
  v_and_b32_e32 v5, 63, v0

  ; Compute LDS write base: waveId * 256 + (lane % 16) * 4
  s_lshl_b32 s5, s4, 8
  v_and_b32_e32 v6, 15, v5
  v_lshlrev_b32_e32 v6, 2, v6
  v_mov_b32_e32 v1, s5
  v_add_u32_e32 v6, v1, v6

  ; Write value (arbitrary non-zero)
  v_mov_b32_e32 v7, 42

  ; Write 0: lanes 0-15 active, LDS [waveBase, waveBase+64)
  s_mov_b64 exec, 0x000000000000FFFF
  ds_write_b32 v6, v7

  ; Write 1: lanes 16-31, LDS [waveBase+64, waveBase+128)
  s_mov_b64 exec, 0x00000000FFFF0000
  ds_write_b32 v6, v7 offset:64

  ; Write 2: lanes 32-47, LDS [waveBase+128, waveBase+192)
  s_mov_b64 exec, 0x0000FFFF00000000
  ds_write_b32 v6, v7 offset:128

  ; Write 3: lanes 48-63, LDS [waveBase+192, waveBase+256)
  s_mov_b64 exec, 0xFFFF000000000000
  ds_write_b32 v6, v7 offset:192

  )ASM";
  code += "  s_waitcnt lgkmcnt(" + std::to_string(lgkmcnt1) + ")\n";
  code += "  s_mov_b64 exec, -1\n";
  if (barrier1) {
    code += "  s_barrier\n";
  }
  code += R"ASM(
  ; Compute read base: otherWaveId * 256
  s_xor_b32 s6, s4, 1
  s_lshl_b32 s6, s6, 8

  ; Read address: otherBase + (lane % 32) * 4
  v_and_b32_e32 v8, 31, v5
  v_lshlrev_b32_e32 v8, 2, v8
  v_mov_b32_e32 v1, s6
  v_add_u32_e32 v8, v1, v8

  ; Read 0: lanes 0-31 read [otherBase, otherBase+128)
  s_mov_b64 exec, 0x00000000FFFFFFFF
  ds_read_b32 v9, v8
  )ASM";
  if (lgkmcnt2 >= 0) {
    code += "  s_waitcnt lgkmcnt(" + std::to_string(lgkmcnt2) + ")\n";
  }
  code += "  s_mov_b64 exec, -1\n";
  if (barrier2) {
    code += "  s_barrier\n";
  }
  code += R"ASM(
  ; Read 1: lanes 32-63 read [otherBase+128, otherBase+256)
  s_mov_b64 exec, 0xFFFFFFFF00000000
  ds_read_b32 v9, v8 offset:128
  s_waitcnt lgkmcnt(0)

  s_mov_b64 exec, -1
  s_endpgm
  )ASM";
  return code;
}

// lgkmcnt(2): 2 remain (writes 2+3), 2 retired (writes 0+1). Barrier 1
// flushes writes 0+1. lgkmcnt(0): all retired. Barrier 2 flushes writes 2+3.
TEST_F(RaceTestFixture, ExecMask_CrossWave_Correct_2_0) {
  ExpectSuccess(crossWavePartialExecCode(2, 0), 0, 2);
}

// lgkmcnt(0): all 4 writes retired before barrier 1. Conservative but safe.
TEST_F(RaceTestFixture, ExecMask_CrossWave_OverRetire_0_0) {
  ExpectSuccess(crossWavePartialExecCode(0, 0), 0, 2);
}

// lgkmcnt(1): 1 remains (write 3), 3 retired (writes 0+1+2). After read 0,
// outstanding = write 3 + read 0 = 2. lgkmcnt(1): 1 remains (read 0),
// write 3 retired. Barrier 2 flushes write 3. All writes flushed.
TEST_F(RaceTestFixture, ExecMask_CrossWave_OverRetire_1_1) {
  ExpectSuccess(crossWavePartialExecCode(1, 1), 0, 2);
}

// lgkmcnt(3): 3 remain (writes 1+2+3), only write 0 retired. Barrier 1
// flushes write 0. Read 0 overlaps other wave's write 1 (still ACTIVE) → race.
TEST_F(RaceTestFixture, ExecMask_CrossWave_UnderRetire_3_0) {
  ExpectRace(crossWavePartialExecCode(3, 0), "LDS race", 0, 2);
}

// lgkmcnt(2) correct for barrier 1. After read 0, outstanding = writes 2+3
// + read 0 = 3. lgkmcnt(2): 2 remain (write 3 + read 0), write 2 retired.
// Barrier 2 flushes write 2 only. Read 1 overlaps write 3 (ACTIVE) → race.
TEST_F(RaceTestFixture, ExecMask_CrossWave_UnderRetire_2_2) {
  ExpectRace(crossWavePartialExecCode(2, 2), "LDS race", 0, 2);
}

// Missing barrier 1: writes are WAVE_COMPLETE but not flushed for cross-wave
// access. Read 0 sees other wave's writes as unsafe → race.
TEST_F(RaceTestFixture, ExecMask_CrossWave_MissingBarrier1) {
  ExpectRace(crossWavePartialExecCode(0, 0, false, true), "LDS race", 0, 2);
}

// Missing barrier 2: writes 2+3 WAVE_COMPLETE but not flushed. Read 1 sees
// other wave's write 2 as unsafe → race.
TEST_F(RaceTestFixture, ExecMask_CrossWave_MissingBarrier2) {
  ExpectRace(crossWavePartialExecCode(2, 0, true, false), "LDS race", 0, 2);
}

// ============================================================================
// GLOBAL_TO_LDS (direct-to-LDS) cross-wave race detection tests.
//
// Single-wave DTL event lifecycle (registration, vmcnt retirement, race
// before/after vmcnt) is tested via buffer_load ... lds instructions in
// memory_instruction_tests.cpp. These tests cover cross-wave scenarios
// that require multiple Wave objects sharing a Workgroup.
// ============================================================================

// Cross-wave race: wave 0 does GLOBAL_TO_LDS, wave 1 reads same LDS range.
// Even after vmcnt, the event is only WAVE_COMPLETE (safe for wave 0) but
// still unsafe for wave 1 until barrier.
TEST(GlobalToLds, CrossWaveRace) {
  const std::map<std::string, int> labels;
  const std::map<std::string, Macro> macros;
  Workgroup wg;
  wg.resizeLds(1024);
  wg.setRaceChecks(true);

  // Create two waves sharing the same workgroup.
  Wave wave0(/*vgprCount=*/4, /*agprOffset=*/4, /*sgprCount=*/4, WaveSize{64},
             WaveId{0}, wg, &labels, &macros);
  Wave wave1(/*vgprCount=*/4, /*agprOffset=*/4, /*sgprCount=*/4, WaveSize{64},
             WaveId{1}, wg, &labels, &macros);

  IntervalSet intervals;
  intervals.append(0, 64);
  wave0.registerGlobalToLdsEvent(/*pc=*/10, intervals);
  wave0.sWaitCntVmcnt(0);

  // Wave 0 can read safely (WAVE_COMPLETE for its own wave).
  EXPECT_NO_THROW(wg.readLds<uint32_t>(0, WaveId{0}, 0));

  // Wave 1 cannot — event is WAVE_COMPLETE but not RETIRED (no barrier).
  EXPECT_THROW(wg.readLds<uint32_t>(0, WaveId{1}, 0), RaceConditionException);
}

// After barrier (flushWaveCompleteMemoryEvents), the event is fully retired
// and other waves can safely access the LDS range.
TEST(GlobalToLds, CrossWaveSafeAfterBarrier) {
  const std::map<std::string, int> labels;
  const std::map<std::string, Macro> macros;
  Workgroup wg;
  wg.resizeLds(1024);
  wg.setRaceChecks(true);

  Wave wave0(/*vgprCount=*/4, /*agprOffset=*/4, /*sgprCount=*/4, WaveSize{64},
             WaveId{0}, wg, &labels, &macros);
  Wave wave1(/*vgprCount=*/4, /*agprOffset=*/4, /*sgprCount=*/4, WaveSize{64},
             WaveId{1}, wg, &labels, &macros);

  IntervalSet intervals;
  intervals.append(0, 64);
  wave0.registerGlobalToLdsEvent(/*pc=*/10, intervals);
  wave0.sWaitCntVmcnt(0);

  // Simulate barrier: flush all wave-complete events.
  wave0.flushWaveCompleteMemoryEvents();

  // Now wave 1 can safely read.
  EXPECT_NO_THROW(wg.readLds<uint32_t>(0, WaveId{1}, 0));
  EXPECT_TRUE(wg.getLdsWriteEvents().empty());
}
