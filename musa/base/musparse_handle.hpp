// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_CUSPARSE_HANDLE_HPP_
#define GKO_MUSA_BASE_CUSPARSE_HANDLE_HPP_


#include <musa.h>
#include <musparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace musa {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {


inline musparseHandle_t init(musaStream_t stream)
{
    musparseHandle_t handle{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreate(&handle));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSetPointerMode(handle, MUSPARSE_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSetStream(handle, stream));
    return handle;
}


inline void destroy(musparseHandle_t handle)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroy(handle));
}


}  // namespace cusparse
}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_CUSPARSE_HANDLE_HPP_
