// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_
#define GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_


#include <ginkgo/config.hpp>


#if defined(GKO_COMPILING_MUSA)
struct MUstream_st;
struct MUevent_st;
typedef struct MUstream_st CUstream_st;
typedef struct MUevent_st CUevent_st;
struct _mublasHandle_t;
typedef struct _mublasHandle_t cublasContext;
struct _musparse_handle;
typedef struct _musparse_handle cusparseContext;
#else
struct CUstream_st;
struct CUevent_st;
struct cublasContext;
struct cusparseContext;
#endif

struct hipblasContext;

struct hipsparseContext;

#if GINKGO_HIP_PLATFORM_HCC
struct ihipStream_t;
struct ihipEvent_t;
#define GKO_HIP_STREAM_STRUCT ihipStream_t
#define GKO_HIP_EVENT_STRUCT ihipEvent_t
#else
#define GKO_HIP_STREAM_STRUCT CUstream_st
#define GKO_HIP_EVENT_STRUCT CUevent_st
#endif


namespace sycl {
inline namespace _V1 {


class queue;
class event;


}  // namespace _V1
}  // namespace sycl


#endif  // GKO_PUBLIC_CORE_BASE_FWD_DECLS_HPP_
