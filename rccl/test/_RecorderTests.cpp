/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "RcclMockFuncs.hpp"
//#include "TestBed.hpp"

namespace RcclUnitTesting
{
  /**
   * \brief Verify correctness of Recorder record() correctness in binary mode
   * ******************************************************************************************/
  TEST(Recorder, ParseBinary)
  {
    // to add after binary export of logging is supported
  }

  /**
   * \brief Verify correctness of Recorder record() correctness in json mode
   * ******************************************************************************************/
  TEST(Recorder, ParseJson)
  {
    setenv("RCCL_REPLAY_FILE", "test.json", 1);

    int pid = getpid();
    hipStream_t stream;
    hipStreamCreate(&stream);

    int array[] = {2, 3, 5};
    ncclComm comm{.nRanks = 1, .localRank = 1, .localRankToRank = array, .opCount = 8, .planner = {.nTasksColl = 13, .nTasksP2p = 21}};  
    rccl::rcclApiCall call(rccl::rrAllToAllv, {.sendbuff = (void*)0x7f22f9600000, .recvbuff = (void*)0x7f22f9601000, .count = 0, .datatype = ncclFloat32, .comm = &comm, .stream = stream});
    rccl::Recorder::instance().record(call);

    std::vector<rccl::rcclApiCall> calls;
    char entry[4096];
    //parse the outfile
    std::string filename = "test" + std::to_string(pid) + ".json";
    std::ifstream fp("test" + std::to_string(pid) + ".json");
    fp.getline(entry, 4096);
    fp.getline(entry, 4096);
    fp.getline(entry, 4096);
    parseJsonEntry(entry, calls);
    int result = memcmp((char*)&calls[0]+4, (char*)&call+4, sizeof(rccl::rcclApiCall)-4);
    fp.close(); // care that recorder is not designed to anticipate fp closing before destructor
    remove(filename.c_str());
    unsetenv("RCCL_REPLAY_FILE");
    assert(!result);
  }

  /**
   * \brief Verify RCCL Recorder's integrity in multithread context by comparing Recorder
   * instance across different threads.
   * ******************************************************************************************/
  static void recorderCmp(void** recorder)
  {
    *recorder = &(rccl::Recorder::instance());
  }
  TEST(Recorder, VerifyMultithread)
  {
    void *p1, *p2;
    std::thread t1(recorderCmp, &p1);
    std::thread t2(recorderCmp, &p2);
    t1.join();
    t2.join();
    assert(p1 == p2);
  }
}
