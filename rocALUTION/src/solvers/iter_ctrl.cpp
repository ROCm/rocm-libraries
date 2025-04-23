/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "iter_ctrl.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"

#include <complex>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <math.h>

namespace rocalution
{

    IterationControl::IterationControl()
    {
        this->residual_history_.clear();
        this->iteration_ = 0;

        this->init_res_ = false;
        this->rec_      = false;
        this->verb_     = 1;
        this->reached_  = 0;

        this->initial_residual_ = 0.0;
        this->current_res_      = 0.0;
        this->current_index_    = -1;

        this->absolute_tol_   = 1e-15;
        this->relative_tol_   = 1e-6;
        this->divergence_tol_ = 1e+8;
        this->minimum_iter_   = 0;
        this->maximum_iter_   = 1000000;
    }

    IterationControl::~IterationControl()
    {
        this->Clear();
    }

    void IterationControl::Clear(void)
    {
        this->residual_history_.clear();
        this->iteration_ = 0;

        this->init_res_ = false;
        this->reached_  = 0;

        this->current_res_   = 0.0;
        this->current_index_ = -1;
    }

    void IterationControl::Init(double abs, double rel, double div, int max)
    {
        this->InitTolerance(abs, rel, div);
        this->InitMaximumIterations(max);
    }

    void IterationControl::Init(double abs, double rel, double div, int min, int max)
    {
        this->InitTolerance(abs, rel, div);
        this->InitMinimumIterations(min);
        this->InitMaximumIterations(max);
    }

    bool IterationControl::InitResidual(double res)
    {
        this->init_res_         = true;
        this->initial_residual_ = res;

        this->reached_   = 0;
        this->iteration_ = 0;

        if(this->verb_ > 0)
        {
            LOG_INFO("IterationControl initial residual = " << res);
        }

        if(this->rec_ == true)
        {
            this->residual_history_.push_back(res);
        }

        if((std::abs(res) == std::numeric_limits<double>::infinity()) || // infinity
           (res != res))
        { // not a number (NaN)

            LOG_INFO("Residual = " << res << " !!!");
            return false;
        }

        if(std::abs(res) <= this->absolute_tol_)
        {
            this->reached_ = 1;
            return false;
        }

        return true;
    }

    void IterationControl::InitTolerance(double abs, double rel, double div)
    {
        this->absolute_tol_   = abs;
        this->relative_tol_   = rel;
        this->divergence_tol_ = div;

        if((std::abs(abs) == std::numeric_limits<double>::infinity()) || // infinity
           (abs != abs)) // not a number (NaN)
        {
            LOG_INFO("Abs tol = " << abs << " !!!");
        }

        if((std::abs(rel) == std::numeric_limits<double>::infinity()) || // infinity
           (rel != rel)) // not a number (NaN)
        {
            LOG_INFO("Rel tol = " << rel << " !!!");
        }

        if((std::abs(div) == std::numeric_limits<double>::infinity()) || // infinity
           (div != div)) // not a number (NaN)
        {
            LOG_INFO("Div tol = " << div << " !!!");
        }
    }

    void IterationControl::InitMinimumIterations(int min)
    {
        assert(min >= 0);
        assert(min <= this->maximum_iter_);

        this->minimum_iter_ = min;
    }

    void IterationControl::InitMaximumIterations(int max)
    {
        assert(max >= 0);
        assert(max >= this->minimum_iter_);

        this->maximum_iter_ = max;
    }

    int IterationControl::GetMinimumIterations(void) const
    {
        return this->minimum_iter_;
    }

    int IterationControl::GetMaximumIterations(void) const
    {
        return this->maximum_iter_;
    }

    int IterationControl::GetIterationCount(void) const
    {
        return this->iteration_;
    }

    double IterationControl::GetCurrentResidual(void) const
    {
        return this->current_res_;
    }

    int64_t IterationControl::GetAmaxResidualIndex(void) const
    {
        return this->current_index_;
    }

    int IterationControl::GetSolverStatus(void) const
    {
        return this->reached_;
    }

    bool IterationControl::CheckResidual(double res)
    {
        assert(this->init_res_ == true);

        this->iteration_++;
        this->current_res_ = res;

        if(this->verb_ > 1)
        {
            LOG_INFO("IterationControl iter=" << this->iteration_ << "; residual=" << res);
        }

        if(this->rec_ == true)
        {
            this->residual_history_.push_back(res);
        }

        if((std::abs(res) == std::numeric_limits<double>::infinity()) || // infinity
           (res != res))
        { // not a number (NaN)

            LOG_INFO("Residual = " << res << " !!!");
            return true;
        }

        if(this->iteration_ >= this->minimum_iter_)
        {
            if(std::abs(res) <= this->absolute_tol_)
            {
                this->reached_ = 1;
                return true;
            }

            if(res / this->initial_residual_ <= this->relative_tol_)
            {
                this->reached_ = 2;
                return true;
            }

            if(this->iteration_ >= this->maximum_iter_)
            {
                this->reached_ = 4;
                return true;
            }
        }

        if(res / this->initial_residual_ >= this->divergence_tol_)
        {
            this->reached_ = 3;
            return true;
        }

        return false;
    }

    bool IterationControl::CheckResidual(double res, int64_t index)
    {
        this->current_index_ = index;
        return this->CheckResidual(res);
    }

    bool IterationControl::CheckResidualNoCount(double res)
    {
        assert(this->init_res_ == true);

        if((std::abs(res) == std::numeric_limits<double>::infinity()) || // infinity
           (res != res))
        { // not a number (NaN)

            LOG_INFO("Residual = " << res << " !!!");
            return true;
        }

        if(std::abs(res) <= this->absolute_tol_)
        {
            this->reached_ = 1;
            return true;
        }

        if(res / this->initial_residual_ <= this->relative_tol_)
        {
            this->reached_ = 2;
            return true;
        }

        if(res / this->initial_residual_ >= this->divergence_tol_)
        {
            this->reached_ = 3;
            return true;
        }

        if(this->iteration_ >= this->maximum_iter_)
        {
            this->reached_ = 4;
            return true;
        }

        return false;
    }

    bool IterationControl::CheckMaximumIterNoCount(void)
    {
        assert(this->init_res_ == true);

        if(this->iteration_ + 1 >= this->maximum_iter_)
        {
            this->reached_ = 4;
            return true;
        }

        return false;
    }

    void IterationControl::RecordHistory(void)
    {
        this->rec_ = true;
    }

    void IterationControl::Verbose(int verb)
    {
        this->verb_ = verb;
    }

    void IterationControl::WriteHistoryToFile(const std::string& filename) const
    {
        std::ofstream file;

        assert(this->residual_history_.size() > 0);
        assert(this->iteration_ > 0);

        LOG_INFO("Writing residual history to filename = " << filename << "; writing...");

        file.open(filename.c_str(), std::ifstream::out);

        if(!file.is_open())
        {
            LOG_INFO("Can not open file [write]:" << filename);
            FATAL_ERROR(__FILE__, __LINE__);
        }

        file.setf(std::ios::scientific);

        for(int n = 0; n < this->iteration_; n++)
        {
            file << this->residual_history_[n] << std::endl;
        }

        file.close();

        LOG_INFO("Writing residual history to filename = " << filename << "; done");
    }

    // LCOV_EXCL_START
    void IterationControl::PrintInit(void) const
    {
        if(this->minimum_iter_ > 0)
        {
            LOG_INFO("IterationControl criteria: "
                     << "abs tol=" << this->absolute_tol_ << "; "
                     << "rel tol=" << this->relative_tol_ << "; "
                     << "div tol=" << this->divergence_tol_ << "; "
                     << "min iter=" << this->minimum_iter_ << "; "
                     << "max iter=" << this->maximum_iter_);
        }
        else
        {
            LOG_INFO("IterationControl criteria: "
                     << "abs tol=" << this->absolute_tol_ << "; "
                     << "rel tol=" << this->relative_tol_ << "; "
                     << "div tol=" << this->divergence_tol_ << "; "
                     << "max iter=" << this->maximum_iter_);
        }
    }

    void IterationControl::PrintStatus(void)
    {
        switch(reached_)
        {
        case 1:
            LOG_INFO("IterationControl ABSOLUTE criteria has been reached: "
                     << "res norm=" << std::abs(this->current_res_) << "; "
                     << "rel val=" << this->current_res_ / this->initial_residual_ << "; "
                     << "iter=" << this->iteration_);
            break;

        case 2:
            LOG_INFO("IterationControl RELATIVE criteria has been reached: "
                     << "res norm=" << std::abs(this->current_res_) << "; "
                     << "rel val=" << this->current_res_ / this->initial_residual_ << "; "
                     << "iter=" << this->iteration_);
            break;

        case 3:
            LOG_INFO("IterationControl DIVERGENCE criteria has been reached: "
                     << "res norm=" << std::abs(this->current_res_) << "; "
                     << "rel val=" << this->current_res_ / this->initial_residual_ << "; "
                     << "iter=" << this->iteration_);
            break;

        case 4:
            LOG_INFO("IterationControl MAX ITER criteria has been reached: "
                     << "res norm=" << std::abs(this->current_res_) << "; "
                     << "rel val=" << this->current_res_ / this->initial_residual_ << "; "
                     << "iter=" << this->iteration_);
            break;

        default:
            LOG_INFO("IterationControl NO criteria has been reached: "
                     << "res norm=" << std::abs(this->current_res_) << "; "
                     << "rel val=" << this->current_res_ / this->initial_residual_ << "; "
                     << "iter=" << this->iteration_);
        }
    }
    // LCOV_EXCL_STOP

} // namespace rocalution
