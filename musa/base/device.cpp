// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "musa/base/device.hpp"

#include <musa_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "musa/base/scoped_device_id.hpp"


namespace gko {
namespace kernels {
namespace musa {


void reset_device(int device_id)
{
    gko::detail::musa_scoped_device_id_guard guard{device_id};
    musaDeviceReset();
}


void destroy_event(MUevent_st* event)
{
    GKO_ASSERT_NO_CUDA_ERRORS(musaEventDestroy(event));
}


std::string get_device_name(int device_id)
{
    musaDeviceProp prop;
    GKO_ASSERT_NO_CUDA_ERRORS(musaGetDeviceProperties(&prop, device_id));
    return {prop.name};
}


}  // namespace musa
}  // namespace kernels
}  // namespace gko
