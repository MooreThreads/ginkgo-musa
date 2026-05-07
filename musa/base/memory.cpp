// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/memory.hpp"

#include <musa.h>
#include <musa_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "musa/base/scoped_device_id.hpp"


namespace gko {


#define GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(_operation, _size)       \
    {                                                                 \
        auto error_code = _operation;                                 \
        if (error_code == musaErrorMemoryAllocation) {                \
            throw AllocationError(__FILE__, __LINE__, "musa", _size); \
        } else {                                                      \
            GKO_ASSERT_NO_CUDA_ERRORS(error_code);                    \
        }                                                             \
    }


#if GKO_VERBOSE_LEVEL >= 1
#define GKO_EXIT_ON_CUDA_ERROR(_operation)                                  \
    {                                                                       \
        const auto error_code = _operation;                                 \
        if (error_code != musaSuccess) {                                    \
            int device_id{-1};                                              \
            musaGetDevice(&device_id);                                      \
            std::cerr << "Unrecoverable CUDA error on device " << device_id \
                      << " in " << __func__ << ":" << __LINE__ << ": "      \
                      << musaGetErrorName(error_code) << ": "               \
                      << musaGetErrorString(error_code) << std::endl        \
                      << "Exiting program" << std::endl;                    \
            std::exit(error_code);                                          \
        }                                                                   \
    }
#else
#define GKO_EXIT_ON_CUDA_ERROR(_operation)  \
    {                                       \
        const auto error_code = _operation; \
        if (error_code != musaSuccess) {    \
            std::exit(error_code);          \
        }                                   \
    }
#endif


void* CudaAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(musaMalloc(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(musaFree(ptr));
}


#if CUDA_VERSION >= 11020


CudaAsyncAllocator::CudaAsyncAllocator(CUstream_st* stream) : stream_{stream} {}


void* CudaAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(
        musaMallocAsync(&ptr, num_bytes, stream_), num_bytes);
    return ptr;
}


void CudaAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(musaFreeAsync(ptr, stream_));
}


#else  // Fall back to regular allocation


CudaAsyncAllocator::CudaAsyncAllocator(CUstream_st* stream) : stream_{stream}
{
#if GKO_VERBOSE_LEVEL >= 1
    std::cerr << "This version of CUDA does not support musaMallocAsync, "
                 "please use CudaAllocator instead of CudaAsyncAllocator.\n";
#endif
}


void* CudaAsyncAllocator::allocate(size_type num_bytes)
{
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(musaMalloc(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaAsyncAllocator::deallocate(void* ptr)
{
    GKO_EXIT_ON_CUDA_ERROR(musaFree(ptr));
}


#endif


bool CudaAsyncAllocator::check_environment(int device_id,
                                           CUstream_st* stream) const
{
    return stream == stream_;
}


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id)
    : CudaUnifiedAllocator{device_id, musaMemAttachGlobal}
{}


CudaUnifiedAllocator::CudaUnifiedAllocator(int device_id, unsigned int flags)
    : device_id_{device_id}, flags_{flags}
{}


void* CudaUnifiedAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::musa_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(
        musaMallocManaged(&ptr, num_bytes, flags_), num_bytes);
    return ptr;
}


void CudaUnifiedAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::musa_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_CUDA_ERROR(musaFree(ptr));
}


bool CudaUnifiedAllocator::check_environment(int device_id,
                                             CUstream_st* stream) const
{
    return device_id == device_id_;
}


CudaHostAllocator::CudaHostAllocator(int device_id) : device_id_{device_id} {}


void* CudaHostAllocator::allocate(size_type num_bytes)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::musa_scoped_device_id_guard g(device_id_);
    void* ptr{};
    GKO_ASSERT_NO_CUDA_ALLOCATION_ERRORS(musaMallocHost(&ptr, num_bytes),
                                         num_bytes);
    return ptr;
}


void CudaHostAllocator::deallocate(void* ptr)
{
    // we need to set the device ID in case this gets used in a host executor
    detail::musa_scoped_device_id_guard g(device_id_);
    GKO_EXIT_ON_CUDA_ERROR(musaFreeHost(ptr));
}


bool CudaHostAllocator::check_environment(int device_id,
                                          CUstream_st* stream) const
{
    return device_id == device_id_;
}


}  // namespace gko
