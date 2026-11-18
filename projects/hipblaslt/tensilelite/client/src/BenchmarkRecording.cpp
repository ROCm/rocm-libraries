// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BenchmarkRecording.hpp"

#include <cerrno>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef __linux__
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif

using namespace TensileLite;
using namespace TensileLite::Client;

DeviceLostError::DeviceLostError(int32_t exitCode)
    : std::runtime_error("device lost")
    , m_exitCode(exitCode)
{
}

int32_t DeviceLostError::exitCode() const
{
    return m_exitCode;
}

BenchmarkSession::BenchmarkSession(std::shared_ptr<BenchmarkDatabase> database,
                                   int64_t                            pid,
                                   int64_t                            seed,
                                   IndirectFailurePolicy              policy)
    : m_database(std::move(database))
    , m_uncaughtExceptions(std::uncaught_exceptions())
{
    m_database->beginSession(pid, seed, policy);
}

BenchmarkSession::~BenchmarkSession() noexcept
{
    if(m_ended || std::uncaught_exceptions() > m_uncaughtExceptions)
        return;

    try
    {
        end();
    }
    catch(std::exception const& error)
    {
        std::cerr << "failed to end the benchmark session: " << error.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "failed to end the benchmark session" << std::endl;
    }
}

void BenchmarkSession::end()
{
    if(m_ended)
        return;
    m_database->endSession();
    m_ended = true;
}

PairRecording Client::classifyPair(BenchmarkDatabase const* database,
                                   int64_t                  problemIndex,
                                   int64_t                  solutionIndex)
{
    if(!database)
        return PairRecording::Disabled;
    if(database->isPending(problemIndex, solutionIndex))
        return PairRecording::Record;
    return PairRecording::SkippedByDatabase;
}

PairRecorder::PairRecorder(DatabaseReporter& reporter, int64_t problemIndex, int64_t solutionIndex)
    : m_reporter(reporter)
    , m_problemIndex(problemIndex)
    , m_solutionIndex(solutionIndex)
{
    m_reporter.beginRun(m_problemIndex, m_solutionIndex);
}

void PairRecorder::finish()
{
    m_reporter.endRun(m_problemIndex, m_solutionIndex);
}

static int32_t reportDatabaseError(std::exception const& error)
{
    std::cerr << "benchmark database error: " << error.what() << std::endl;
    return 1;
}

int32_t Client::runCatchingBenchmarkFailures(std::function<int32_t()> const& client)
{
    try
    {
        return client();
    }
    catch(DeviceLostError const& error)
    {
        return error.exitCode();
    }
    catch(BenchmarkDatabaseError const& error)
    {
        return reportDatabaseError(error);
    }
    catch(SqliteError const& error)
    {
        return reportDatabaseError(error);
    }
}

bool Client::dieWithParent()
{
#ifdef __linux__
    // The parent is captured first: if it dies before the prctl, no signal is ever sent.
    pid_t const parent = getppid();
    if(prctl(PR_SET_PDEATHSIG, SIGKILL) != 0)
        throw std::runtime_error(std::string("prctl(PR_SET_PDEATHSIG) failed: ")
                                 + std::strerror(errno));
    return getppid() == parent;
#else
    return true;
#endif
}
