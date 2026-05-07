// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <musa_runtime.h>
#include <mublas.h>
#include <mufft.h>
#include <murand.h>
#include <musparse.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


TEST(AssertNoCudaErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUDA_ERRORS(1), gko::CudaError);
}


TEST(AssertNoCudaErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUDA_ERRORS(musaSuccess));
}


TEST(AssertNoCublasErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUBLAS_ERRORS(1), gko::CublasError);
}


TEST(AssertNoCublasErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUBLAS_ERRORS(MUBLAS_STATUS_SUCCESS));
}


TEST(AssertNoCurandErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CURAND_ERRORS(1), gko::CurandError);
}


TEST(AssertNoCurandErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CURAND_ERRORS(MURAND_STATUS_SUCCESS));
}


TEST(AssertNoCusparseErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(1), gko::CusparseError);
}


TEST(AssertNoCusparseErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(MUSPARSE_STATUS_SUCCESS));
}


TEST(AssertNoCufftErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUFFT_ERRORS(1), gko::CufftError);
}


TEST(AssertNoCufftErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUFFT_ERRORS(MUFFT_SUCCESS));
}


}  // namespace
