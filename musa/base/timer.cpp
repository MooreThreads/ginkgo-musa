// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/timer.hpp"

#include <musa.h>
#include <musa_runtime_api.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "musa/base/scoped_device_id.hpp"


namespace gko {


CudaTimer::CudaTimer(std::shared_ptr<const CudaExecutor> exec)
    : device_id_{exec->get_device_id()}, stream_{exec->get_stream()}
{}


void CudaTimer::init_time_point(time_point& time)
{
    detail::musa_scoped_device_id_guard guard{device_id_};
    time.type_ = time_point::type::musa;
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventCreate(&time.data_.musa_event));
}


void CudaTimer::record(time_point& time)
{
    detail::musa_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(time.type_ == time_point::type::musa);
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventRecord(time.data_.musa_event, stream_));
}


void CudaTimer::wait(time_point& time)
{
    detail::musa_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(time.type_ == time_point::type::musa);
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventSynchronize(time.data_.musa_event));
}


std::chrono::nanoseconds CudaTimer::difference_async(const time_point& start,
                                                     const time_point& stop)
{
    detail::musa_scoped_device_id_guard guard{device_id_};
    GKO_ASSERT(start.type_ == time_point::type::musa);
    GKO_ASSERT(stop.type_ == time_point::type::musa);
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventSynchronize(stop.data_.musa_event));
    float ms{};
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventElapsedTime(&ms, start.data_.musa_event,
                                                   stop.data_.musa_event));
    return std::chrono::nanoseconds{static_cast<int64>(ms * double{1e6})};
}


}  // namespace gko
