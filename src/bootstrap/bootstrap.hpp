// Copyright (c) Microsoft Corporation.
// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Licensed under the MIT license.

#ifndef ROCSHMEM_BOOTSTRAP_HPP_
#define ROCSHMEM_BOOTSTRAP_HPP_


#include <array>
#include <bitset>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "rocshmem/rocshmem_common.hpp"

namespace rocshmem {

/// Return a version string.
std::string version();

/// Base class for bootstraps.
class Bootstrap {
 public:
  Bootstrap(){};
  virtual ~Bootstrap() = default;
  virtual int getRank() = 0;
  virtual int getNranks() = 0;
  virtual int getNranksPerNode() = 0;
  virtual void send(void* data, int size, int peer, int tag) = 0;
  virtual void recv(void* data, int size, int peer, int tag) = 0;
  virtual void allGather(void* allData, int size) = 0;
  virtual void barrier() = 0;

  void groupBarrier(const std::vector<int>& ranks);
  void send(const std::vector<char>& data, int peer, int tag);
  void recv(std::vector<char>& data, int peer, int tag);
};

/// A native implementation of the bootstrap using TCP sockets.
class TcpBootstrap : public Bootstrap {
 public:
  /// Create a random unique ID.
  /// @return The created unique ID.
  static rocshmem_uniqueid_t createUniqueId();

  /// Constructor.
  /// @param rank The rank of the process.
  /// @param nRanks The total number of ranks.
  TcpBootstrap(int rank, int nRanks);

  /// Destructor.
  ~TcpBootstrap();

  /// Return the unique ID stored in the @ref TcpBootstrap.
  /// @return The unique ID stored in the @ref TcpBootstrap.
  rocshmem_uniqueid_t getUniqueId() const;

  /// Initialize the @ref TcpBootstrap with a given unique ID.
  /// @param uniqueId The unique ID to initialize the @ref TcpBootstrap with.
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(rocshmem_uniqueid_t uniqueId, int64_t timeoutSec = 30);

  /// Initialize the @ref TcpBootstrap with a string formatted as "ip:port" or "interface:ip:port".
  /// @param ifIpPortTrio The string formatted as "ip:port" or "interface:ip:port".
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(const std::string& ifIpPortTrio, int64_t timeoutSec = 30);

  /// Return the rank of the process.
  int getRank() override;

  /// Return the total number of ranks.
  int getNranks() override;

  /// Return the total number of ranks per node.
  int getNranksPerNode() override;

  /// Send data to another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
  ///
  /// @param data The data to send.
  /// @param size The size of the data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  void send(void* data, int size, int peer, int tag) override;

  /// Receive data from another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
  ///
  /// @param data The buffer to write the received data to.
  /// @param size The size of the data to receive.
  /// @param peer The rank of the process to receive the data from.
  /// @param tag The tag to receive the data with.
  void recv(void* data, int size, int peer, int tag) override;

  /// Gather data from all processes.
  ///
  /// When called by rank `r`, this sends data from `allData[r * size]` to `allData[(r + 1) * size - 1]` to all other
  /// ranks. The data sent by rank `r` is received into `allData[r * size]` of other ranks.
  ///
  /// @param allData The buffer to write the received data to.
  /// @param size The size of the data each rank sends.
  void allGather(void* allData, int size) override;

  /// Synchronize all processes.
  void barrier() override;

 private:
  // The interal implementation.
  class Impl;

  // Pointer to the internal implementation.
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_BOOTSTRAP_HPP_
