// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_CUBLAS_HANDLE_HPP_
#define GKO_MUSA_BASE_CUBLAS_HANDLE_HPP_


#include <mublas.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace musa {
namespace cublas {


inline mublasHandle_t init(musaStream_t stream)
{
    mublasHandle_t handle;
    GKO_ASSERT_NO_CUBLAS_ERRORS(mublasCreate(&handle));
    GKO_ASSERT_NO_CUBLAS_ERRORS(
        mublasSetPointerMode(handle, MUBLAS_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_CUBLAS_ERRORS(mublasSetStream(handle, stream));
    return handle;
}


inline void destroy(mublasHandle_t handle)
{
    GKO_ASSERT_NO_CUBLAS_ERRORS(mublasDestroy(handle));
}


}  // namespace cublas
}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_CUBLAS_HANDLE_HPP_
