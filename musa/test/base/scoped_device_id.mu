// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "musa/base/scoped_device_id.hpp"

#include <musa_runtime.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>

#include "musa/test/utils.hpp"


namespace {


class ScopedDeviceIdGuard : public CudaTestFixture {};


TEST_F(ScopedDeviceIdGuard, SetsId)
{
    auto new_device_id = std::max(exec->get_num_devices() - 1, 0);

    gko::detail::musa_scoped_device_id_guard g{new_device_id};

    int device_id;
    musaGetDevice(&device_id);
    ASSERT_EQ(device_id, new_device_id);
}


TEST_F(ScopedDeviceIdGuard, ResetsId)
{
    auto old_device_id = exec->get_device_id();

    {
        auto new_device_id = std::max(exec->get_num_devices() - 1, 0);
        gko::detail::musa_scoped_device_id_guard g{new_device_id};
    }

    int device_id;
    musaGetDevice(&device_id);
    ASSERT_EQ(device_id, old_device_id);
}


}  // namespace
