// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_DEVICE_HPP_
#define GKO_MUSA_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace musa {


/** calls musaDeviceReset on the given device. */
void reset_device(int device_id);


/** calls musaEventDestroy on the given event. */
void destroy_event(MUevent_st* event);


/** returns musaDeviceProp.name for the given device */
std::string get_device_name(int device_id);


}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_DEVICE_HPP_
