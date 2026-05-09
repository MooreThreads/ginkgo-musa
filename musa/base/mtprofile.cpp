// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <musa_runtime.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {


constexpr static uint32 category_magic_offset = 0x676B6FU;


void init_nvtx() {}


std::function<void(const char*, profile_event_category)> begin_nvtx_fn(
    uint32_t color_argb)
{
    return [](const char*, profile_event_category) {};
}


void end_nvtx(const char* name, profile_event_category) {}


}  // namespace log
}  // namespace gko
