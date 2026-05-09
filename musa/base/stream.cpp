// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/stream.hpp"

#include <musa_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "musa/base/scoped_device_id.hpp"


namespace gko {


cuda_stream::cuda_stream() : stream_{nullptr}, device_id_{} {}


cuda_stream::cuda_stream(int device_id) : stream_{}, device_id_(device_id)
{
    detail::musa_scoped_device_id_guard g(device_id_);
    GKO_ASSERT_NO_CUDA_ERRORS(musaStreamCreate(&stream_));
}


cuda_stream::~cuda_stream()
{
    if (stream_) {
        detail::musa_scoped_device_id_guard g(device_id_);
        musaStreamDestroy(stream_);
    }
}


cuda_stream::cuda_stream(cuda_stream&& other)
    : stream_{std::exchange(other.stream_, nullptr)},
      device_id_(std::exchange(other.device_id_, 0))
{}


CUstream_st* cuda_stream::get() const { return stream_; }


}  // namespace gko
